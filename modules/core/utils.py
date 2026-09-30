import torch
import numpy as np
import random
import yaml
import os
from dotmap import DotMap

from core import feature_spec
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
    
    # 把音频编码器的输出维度带进 model 段（网络结构据此决定要不要挂 ResNet）
    apply_encoder_meta(config)
    return config

def apply_encoder_meta(config):
    """把当前编码器的属性投影到 config.model，供网络结构使用。

    目前只带一个 `audio_dim`：音频编码器输出维度 ≠ hidden_size 时，
    `networks/model.py` 会据此挂一层 ResNet 升维（以前是拿模型名字猜的：
    `if 'mel' in model_name or 'egemaps' in model_name`）。

    维度只在 configs/*.yaml 的 `encoders` 段定义一次，这里不重复写值。
    训练走 get_config()，评估在 build_config() 里各自调一次。
    """
    spec = feature_spec.from_config(config)
    config.model.audio_dim = spec.dim('audio') if config.model.audio_model else 0
    return config


def save_config(config):
    """Save the configuration to a YAML file, ensuring log directories exist."""

    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''

    textual_data = config.model.textual_model + '_' if config.model.textual_model != '' else ''
    audio_data = config.model.audio_model + '_' if config.model.audio_model != '' else ''
    # 停顿开关统一从配置的 dataset 段读（以前 model.pauses 是另一份，同一个语义
    # 写两遍，改一处忘一处结果目录名就跟特征文件名对不上了）
    pauses_data = 'P_' if feature_spec.from_config(config).pauses else ''

    config.model_name = f"{textual_data}{audio_data}{pauses_data}{config.model.fusion}"
    config.model.model_name = config.model_name

    config.path_name = f"{config.model_name}_{config.model.pooling}"

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

def train(model, train_dataloader, valid_dataloader, lossfn, optimizer, lr_scheduler, num_epochs, model_name, early_stopping, early_stopping_patience, cross_val=False, num_cross_val=0, early_stopping_metric='loss', early_stopping_min_delta=0.0):
    """Train the model with early stopping."""
    wandb.init(project="WordLevelFusion", config={"epochs": num_epochs})
    wandb.watch(model)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    log_path = f'logs/{model_name}/train_stats_{num_cross_val}.txt' if cross_val else f'logs/{model_name}/train_stats.txt'
    
    # ── 早停判据 ────────────────────────────────────────────────────────────
    # `early_stopping_metric`（配置项 train.early_stopping_metric）：
    #   'loss'     —— 验证损失，**越小越好**（默认）。损失是连续值，不会像准确率
    #                 那样卡在台阶上（验证集只有 47 条 → 准确率只有 48 个取值），
    #                 所以能真实反映"模型还在不在进步"。
    #   'accuracy' —— 验证准确率，越大越好（旧行为）。
    #
    # 为什么默认改成 loss（实测依据）：
    #   旧的 accuracy 判据下，第 2 折 ep6 摸到 59.6% 后连续 20 轮没超过 → ep26 停，
    #   整折只训 26 轮；第 3 折 ep2 摸到 61.7% → ep22 停，只训 22 轮。
    #   验证集只有 47 条，准确率严格大于的判据几乎等于"原地抖动就消耗耐心"，
    #   模型刚起步就被判死刑。改看 loss 后判据是连续的，不会出现这种假停。
    # ────────────────────────────────────────────────────────────────────────
    higher_is_better = (early_stopping_metric == 'accuracy')
    best_value, patience = (-float("inf") if higher_is_better else float("inf")), 0
    best_epoch, best_weights, rest_best_values = 0, None, []
    best_metric_extra = {}
    
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
            
            validation_value, rest_values, val_extra = evaluation(model, valid_dataloader, lossfn, log)
            
            # 按配置的判据取这一轮用来比较的数
            #   'loss'     —— 越小越好；'accuracy' —— 越大越好（旧行为）
            current = (val_extra['accuracy'] if higher_is_better else val_extra['loss'])
            # 只有改进量 > min_delta 才算"刷新"。
            # 注意 loss 方向：current 比 best 小得够多才算改进
            # （min_delta=0 时退化和旧行为一致，仍是"严格更好"，不会因此少训）。
            improved = (current > best_value + early_stopping_min_delta
                        if higher_is_better
                        else current < best_value - early_stopping_min_delta)
            
            if improved:
                best_epoch, best_weights = epoch + 1, copy.deepcopy(model.state_dict())
                best_value, rest_best_values = current, rest_values
                best_metric_extra = val_extra
                patience = 0
            else:
                patience += 1
            
            if patience == early_stopping_patience and early_stopping:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        if not rest_best_values:
            rest_best_values = [0, 0, 0]
        
        # 汇总行按判据改名，避免出现 "Best validation accuracy: 0.43" 这种
        # 把 loss 当 accuracy 写的误导（旧版本写死了 accuracy 这个词）。
        best_metric_name = 'accuracy' if higher_is_better else 'loss'
        log.write(f'Best validation {best_metric_name}: {best_value}\n')
        log.write(f'Best validation accuracy: {best_metric_extra.get("accuracy", 0)}\n')
        log.write(f'Best validation F1: {rest_best_values[0]}\nBest validation Recall: {rest_best_values[1]}\nBest validation Precision: {rest_best_values[2]}\n')
        log.write(f'Best epoch: {best_epoch}\n')
    
    # 兜底：理论上 best_weights 不会是 None（best_value 从 ±inf 起，
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
    
    # 返回值从二元组变成三元组：多带一个 metrics 字典，把 avg_loss 交出去，
    # 早停才能按"验证 loss"判（见 train() 的 early_stopping_metric）。
    # 只有 train() 内部调用它，已同步改成解包 3 个值。
    return accuracy, [f1, recall, precision], {'loss': avg_loss, 'accuracy': accuracy}


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
