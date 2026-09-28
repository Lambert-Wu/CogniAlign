import torch
import numpy as np
import random
import yaml
import os
from dotmap import DotMap
import wandb
from tqdm import tqdm
import time
import copy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def get_config(config_file):
    """Load configuration from a YAML file and ensure log directories exist."""
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_yaml = yaml.safe_load(f)
    
    config = DotMap(config_yaml)
    config.path_name = f"{config.model_name}_{config.model.pooling}"
    log_path = os.path.join('logs', config.path_name)
    
    # os.makedirs(log_path, exist_ok=True)
    
    # config_file_path = os.path.join(log_path, 'config.yaml')
    
    # with open(config_file_path, 'w') as f:
        # yaml.dump(config_yaml, f, default_flow_style=False)
    
    return config

def save_config(config):
    """Save the configuration to a YAML file, ensuring log directories exist."""

    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''

    textual_data = config.model.textual_model + '_' if config.model.textual_model != '' else ''
    audio_data = config.model.audio_model + '_' if config.model.audio_model != '' else ''
    pauses_data = 'P_' if config.model.pauses else ''

    config.model_name = f"{textual_data}{audio_data}{pauses_data}{config.model.fusion}"
    config.model.model_name = config.model_name

    config.path_name = f"{config.model_name}_{config.model.pooling}"

    # 允许用环境变量指定「结果文件夹的名字」（run_train.sh -o 走的就是这条路）。
    # ⚠️ **只改 path_name**（日志/权重目录），**绝不能动 model_name** ——
    # model_name 会被模型类拿去做两件事：① 判断要不要挂 mel/egemaps 的 ResNet；
    # ② forward() 里 `model_name.split('_')[1]` 取出音频模型名。
    # 动它就是直接改模型结构，权重会跟之前完全对不上。
    _run_name = os.environ.get("COGNIALIGN_RUN_NAME", "").strip()
    if _run_name:
        config.path_name = _run_name

    # ⚠️ log_path / config_file_path 必须在 path_name 更新**之后**才算。
    # 原代码把这两行放在函数开头，而那时 path_name 还是 get_config() 里那个
    # f"{空的 model_name}_{pooling}" = "_mean"，于是：
    #   · 凭空建出一个 logs/_mean/ 垃圾目录
    #   · config.yaml 被写进 logs/_mean/，真正的结果目录里反而没有配置文件
    log_path = os.path.join('logs', config.path_name)
    os.makedirs(log_path, exist_ok=True)
    config_file_path = os.path.join(log_path, 'config.yaml')

    # Convert DotMap to a standard dictionary
    config_dict = config.toDict()

    with open(config_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_metrics_classification(true_labels, pred_labels):
    """Compute classification metrics safely."""
    zero_div = 1 if len(set(true_labels)) == 1 else 0  # Avoid zero division warnings
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    
    return accuracy, f1, recall, precision

def train(model, train_dataloader, valid_dataloader, lossfn, optimizer, lr_scheduler, num_epochs, model_name, early_stopping, early_stopping_patience, cross_val=False, num_cross_val=0):
    """Train the model with early stopping."""
    wandb.init(project="WordLevelFusion", config={"epochs": num_epochs})
    wandb.watch(model)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    log_path = f'logs/{model_name}/train_stats_{num_cross_val}.txt' if cross_val else f'logs/{model_name}/train_stats.txt'
    
    # best_value 必须从 -inf 起（原代码是 0）：
    # 验证准确率有可能整轮都是 0（样本少 / 模型没学到东西），
    # 那时 `validation_value > best_value` 永远不成立 → best_weights 始终是 None
    # → 结尾 load_state_dict(None) 直接 TypeError 崩掉（实测踩过）。
    best_value, patience = -float("inf"), 0
    best_epoch, best_weights, rest_best_values = 0, None, []
    
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    # buffering=1 = 行缓冲，每写一行就落盘。
    # 默认是块缓冲（8KB），而训练日志每 epoch 才几百字节，要攒约 30 个 epoch
    # 才写一次盘 —— 另开一个终端 tail -f 会一直看不到内容，看起来像卡住了。
    with open(log_path, "w", encoding='utf-8', buffering=1) as log:
        for epoch in range(num_epochs):
            model.train()
            total_true, total_pred, total_loss = [], [], 0
            
            progress_bar.set_description(f"Epoch {epoch + 1}")
            log.write(f'Epoch {epoch + 1}:\n')
            
            for features, labels in train_dataloader:        
                outputs = model(features).squeeze(-1)
                loss = lossfn(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predictions = torch.round(probs)
                
                if torch.isnan(predictions).any():
                    print("⚠️ Warning: NaN detected in predictions! Skipping batch.")
                    continue
                
                predictions = predictions.detach().cpu().numpy().astype(int)
                labels = labels.detach().cpu().numpy().astype(int)
                
                total_true.extend(labels)
                total_pred.extend(predictions)
                progress_bar.update(1)
            
            accuracy, f1, recall, precision = get_metrics_classification(total_true, total_pred)
            avg_loss = total_loss / len(train_dataloader)
            
            log.write(f'Training completed in: {time.time()} seconds\n')
            log.write(f'Loss: {avg_loss}\nAccuracy: {accuracy}\nF1 Score: {f1}\nRecall: {recall}\nPrecision: {precision}\n')
            wandb.log({"train_loss": avg_loss, "train_ACC": accuracy, "train_F1": f1})
            
            validation_value, rest_values = evaluation(model, valid_dataloader, lossfn, log)
            
            if validation_value > best_value:
                best_epoch, best_weights = epoch + 1, copy.deepcopy(model.state_dict())
                best_value, rest_best_values = validation_value, rest_values
                patience = 0
            else:
                patience += 1
            
            if patience == early_stopping_patience and early_stopping:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        if not rest_best_values:
            rest_best_values = [0, 0, 0]
        
        log.write(f'Best validation accuracy: {best_value}\n')
        log.write(f'Best validation F1: {rest_best_values[0]}\nBest validation Recall: {rest_best_values[1]}\nBest validation Precision: {rest_best_values[2]}\n')
        log.write(f'Best epoch: {best_epoch}\n')
    
    # 兜底：理论上 best_weights 不会是 None（best_value 从 -inf 起，
    # 第一个 epoch 必定会保存一次），但白跑一整折再崩不值得，这里再加一道保护。
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, best_value, rest_best_values

def evaluation(model, dataloader, lossfn, log, test=False):
    """Evaluate the model on a given dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_true, total_pred, total_loss = [], [], 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features).squeeze(-1)
            loss = lossfn(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            predictions = torch.round(probs)
            
            if torch.isnan(predictions).any():
                print("⚠️ Warning: NaN detected in predictions! Skipping batch.")
                continue
            
            predictions = predictions.detach().cpu().numpy().astype(int)
            labels = labels.detach().cpu().numpy().astype(int)
            
            total_true.extend(labels)
            total_pred.extend(predictions)
    
    accuracy, f1, recall, precision = get_metrics_classification(total_true, total_pred)
    avg_loss = total_loss / len(dataloader)
    
    log.write(f'Loss: {avg_loss}\nAccuracy: {accuracy}\nF1 Score: {f1}\nRecall: {recall}\nPrecision: {precision}\n')
    wandb.log({"test_loss": avg_loss, "test_UAR": accuracy, "test_F1": f1} if test else {"validation_loss": avg_loss, "validation_ACC": accuracy, "validation_F1": f1})
    
    return accuracy, [f1, recall, precision]


def get_model_statistics(model='all'):
    directory = 'logs/'
    folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # Ordered structure
    grouped_results = {}
    models_used = set()

    for folder_name in folder_names:
        try:
            model_name, pooling = folder_name.split('_')  # Expected: "distilbert_base_cls"
        except ValueError:
            print(f"Warning: Unexpected folder name format {folder_name}, skipping.")
            continue
        
        if model != 'all' and model not in model_name:
            continue
        
        file_path = os.path.join(directory, folder_name, 'cross_fold_summary.txt')
        
        if not os.path.exists(file_path):
            print(f"Warning: Missing file {file_path}")
            continue
        
        try:
            with open(file_path, "r", encoding='utf-8') as result_file:
                lines = result_file.readlines()
            
            if not lines:
                print(f"Warning: Empty file {file_path}")
                continue

            metrics = {'acc': [], 'f1': [], 'recall': [], 'precision': []}
            
            for i in range(0, len(lines), 4):
                try:
                    metrics['acc'].append(float(lines[i].split()[-1]) * 100)
                    metrics['f1'].append(float(lines[i+1].split()[-1]) * 100)
                    metrics['recall'].append(float(lines[i+2].split()[-1]) * 100)
                    metrics['precision'].append(float(lines[i+3].split()[-1]) * 100)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Malformed line in {file_path} - {e}")
                    continue

            if not all(metrics[key] for key in metrics):
                print(f"Warning: Incomplete statistics in {file_path}")
                continue

            means = np.array([np.mean(metrics[key]) for key in metrics])
            stds = np.array([np.std(metrics[key]) for key in metrics])

            if model_name not in grouped_results:
                grouped_results[model_name] = {}
            
            grouped_results[model_name][pooling] = (
                round(means[0], 2), round(stds[0], 1),
                round(means[1], 2), round(stds[1], 1),
                round(means[2], 2), round(stds[2], 1),
                round(means[3], 2), round(stds[3], 1)
            )
            models_used.add(model_name)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Print LaTeX formatted table
    for model_name, poolings in grouped_results.items():
        print("\n\n\\begin{table}[H]")
        print("\\centering")
        print("\\begin{tabular}{l|cccc}")
        print("\\hline")
        print("Pooling & Acc & F1 & Recall & Precision \\\\")
        print("\\Xhline{1pt}")
        
        for pooling in sorted(poolings.keys()):  # Ensure consistent order
            values = poolings[pooling]
            print(f"{pooling}  &  {values[0]}  $\\pm$  {values[1]}  &  {values[2]}  $\\pm$  {values[3]}  &  {values[4]}  $\\pm$  {values[5]}  &  {values[6]}  $\\pm$  {values[7]} \\\\")
        print("\\hline")
        
        print("\\end{tabular}")
        print(f"\\caption{{{model_name}}}")
        print("\\end{table}")
