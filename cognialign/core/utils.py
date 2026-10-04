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


def load_init_weights(model, config, fold, device):
    """可选：从已有权重继续训练（微调），而不是随机初始化。

    为什么需要：train.py 原来只会 `model_module.build()` 随机初始化再训。
    做少样本微调时，我们要的是「**拿英文训好的权重当起点，再用几条中文继续训**」，
    而不是从零学 —— 8 条样本从零训没有意义。

    配置项 `train.init_checkpoint`：
      · 空 / 不写        —— 不加载，保持随机初始化（原行为，不影响任何已有实验）
      · 指向 .pth 文件   —— 直接加载这个文件，所有折都用同一份起点
      · 指向目录         —— 取目录里的 `model_fold_<fold>.pth`；没有才退回 `model.pt`
                            （和 train.py 存权重的命名一致，方便"第 k 折起点配第 k 折"）

    相对路径按 `cognialign/` 解析（train.py 就在那里，也和各处 `--config` 的口径一致）。
    权重结构和当前配置不符时会当场报错（strict=True），不静默半加载。
    """
    t = config.get('train', {}) or {}
    raw = str(t.get('init_checkpoint', '') or '').strip()
    if not raw:
        return model

    modules_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = raw if os.path.isabs(raw) else os.path.join(modules_dir, raw)
    if os.path.isdir(path):
        cand = os.path.join(path, 'model_fold_%d.pth' % fold)
        if not os.path.exists(cand):
            cand = os.path.join(path, 'model.pt')
        path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(
            'train.init_checkpoint 指向的权重不存在：\n    %s\n'
            '（相对路径按 cognialign/ 解析；目录里应有 model_fold_<折>.pth 或 model.pt）'
            % path)

    sd = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print('[微调] 初始权重：%s' % path)
    print('[微调] 已加载 %d 个参数块（缺失 %d / 多余 %d）'
          % (len(sd), len(missing), len(unexpected)))
    return model


def save_config(config):
    """Save the configuration to a YAML file, ensuring log directories exist."""

    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''

    # 结果目录名**不在这里拼** —— 规则只有一处，在 core/feature_spec.result_names()。
    # 以前这里和 run_train.sh 各写了一份（Python 一份、bash 一份），
    # 加一个"实验标签"要改两个地方，漏一处就会出现"脚本显示的结果目录"
    # 和"模型真正写进去的目录"不是一个地方。
    # ⚠️ 停顿开关也从那里读（它统一从配置的 dataset 段取，不再看 model.pauses）。
    config.model_name, config.path_name = feature_spec.result_names(config)
    config.model.model_name = config.model_name

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

def train(model, train_dataloader, valid_dataloader, lossfn, optimizer, lr_scheduler, num_epochs, model_name, early_stopping, early_stopping_patience, cross_val=False, num_cross_val=0, early_stopping_metric='loss', early_stopping_min_delta=0.0, select_best=True):
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
    # select_best=False 时用这两个记"最后一个 epoch"的结果（不做任何选点）
    last_value, last_rest_values, last_metric_extra = None, [0, 0, 0], {}
    
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
            # 记录"最后一个 epoch"的结果，供 select_best=False 时使用
            last_value, last_rest_values, last_metric_extra = current, rest_values, val_extra
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
        
        # ── 收尾：选点（默认）还是用最后一个 epoch ────────────────────────────
        # select_best=False：**不做任何基于验证集的选点** —— 固定 epoch 数训练，
        # 直接返回最后一个 epoch 的权重。用于"防止在评测集上选点造成泄漏"的
        # 严谨评估（见 docs/RESULTS.md 的中文少样本微调章节）。
        if select_best:
            if best_weights is not None:
                model.load_state_dict(best_weights)
            final_value, final_rest = best_value, rest_best_values
            final_extra, final_epoch = best_metric_extra, best_epoch
        else:
            final_value = last_value if last_value is not None else best_value
            final_rest = last_rest_values
            final_extra, final_epoch = last_metric_extra, num_epochs
            log.write('[select_best=False] 不选点，返回最后一个 epoch 的权重\n')

        if not final_rest:
            final_rest = [0, 0, 0]

        # 汇总行按判据改名，避免出现 "Best validation accuracy: 0.43" 这种
        # 把 loss 当 accuracy 写的误导（旧版本写死了 accuracy 这个词）。
        best_metric_name = 'accuracy' if higher_is_better else 'loss'
        log.write(f'Best validation {best_metric_name}: {final_value}\n')
        log.write(f'Best validation accuracy: {final_extra.get("accuracy", 0)}\n')
        log.write(f'Best validation F1: {final_rest[0]}\nBest validation Recall: {final_rest[1]}\nBest validation Precision: {final_rest[2]}\n')
        log.write(f'Best epoch: {final_epoch}\n')

    # 兜底：理论上 best_weights 不会是 None（best_value 从 ±inf 起，
    # 第一个 epoch 必定会保存一次），但白跑一整折再崩不值得，这里再加一道保护。
    if select_best and best_weights is not None:
        model.load_state_dict(best_weights)
    return model, final_value, final_rest

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
        # 目录名本身就是实验标识（{文本}_{音频}_{pause|nopause}[_tag]）。
        # 池化默认 mean、已不写进目录名，这里**不再**按 '_' 切成两段
        # （旧写法 folder_name.split('_') 对现役目录全部会跳过）。
        model_name = folder_name
        pooling = 'mean'
        
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
