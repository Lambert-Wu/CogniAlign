from dataset.dataset import get_dataloaders
from core.utils import set_seed, get_config, train, save_config, load_init_weights
from core import feature_spec
from networks import model as model_module
import torch
import wandb
import sys
import torch.nn as nn
from transformers import get_scheduler
from torch.optim import AdamW
wandb.login()
import os


def set_up(config, train_dataloader, device, fold=0):
    """Set up model, optimizer, loss function, and scheduler."""
    # 随机种子：COGNIALIGN_SEED > config.train.seed > 42（见 feature_spec.seed_of）。
    # 多随机种子重复实验靠它；非默认种子会写进结果目录名的 _seed<N> 后缀。
    set_seed(feature_spec.seed_of(config))
    
    # 用哪个网络结构由配置的 model.architecture 决定，训练和评估共用同一份实现
    # （见 networks/model.py 的 build）。以前这里和 evaluate.py 各写一份
    # `if 'cross' in fusion` 的字符串判断，改一处忘一处就会训评不一致。
    model = model_module.build(config.model).to(device)

    # 可选：从已有权重继续微调（配置 train.init_checkpoint，见 core.utils.load_init_weights）。
    # 不写这一项时行为完全不变（随机初始化）—— 所有已有实验不受影响。
    model = load_init_weights(model, config, fold, device)


    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    lossfn = nn.BCEWithLogitsLoss()
    
    num_training_steps = config.train.num_epochs * len(train_dataloader)

    # ── warmup 步数做成配置项（train.warmup_steps，默认 20 = 旧行为）──────────
    # ⚠️ 为什么必须可配：样本极少、batch 很大时，每 epoch 只有 1 个 step。
    #    warmup=20 会让"第 1 个 step 的学习率为 0、前 20 个 step 都几乎不学"，
    #    若验证 loss 恰好没低于初始值，早停就会把**等于初始权重**的 epoch 1
    #    当成 best 存下来（实测：8 条样本 + batch=8 时，5 个 seed 里 3 个中招）。
    #    few-shot 配置把它设成 0（第一个 step 就用满学习率）。
    #    ⚠️ 注意 HF 的 warmup 语义：step 0 的 lr = base * 0/warmup，
    #    所以 warmup_steps=1 时第 1 个 step 的学习率**仍然是 0**，只有 0 才真正跳过。
    num_warmup_steps = int(config.train.get('warmup_steps', 20) or 0)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    if num_warmup_steps != 20:
        print('[warmup] num_warmup_steps=%d（配置 train.warmup_steps）' % num_warmup_steps)

    
    wandb.init(
        project="WordLevelFusion",
        name=f"{config.model_name}_{fold}" if config.train.cross_validation else config.model_name,
        config={
            "learning_rate": config.train.learning_rate,
            "architecture": config.model_name,
            "dataset": "ADReSSo",
            "epochs": config.train.num_epochs,
            "batch_size": config.train.batch_size,
            "seed": feature_spec.seed_of(config),
        }
    )
    
    wandb.watch(model)
    return model, optimizer, lossfn, lr_scheduler

def _resume_enabled():
    """是否开启续跑。设 COGNIALIGN_RESUME=1 时，已经存过权重的折直接跳过。

    训练没有"从某个 epoch 接着训"的能力（权重只在每折结束时才存盘），
    所以"续跑"的粒度是**折**：中断时正在跑的那一折白跑，已跑完的折保留。
    """
    return os.environ.get('COGNIALIGN_RESUME', '0').strip().lower() in ('1', 'true', 'yes')


def main(config):
    """Main function to train and save model, supporting cross-validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.join('logs', config.path_name)
    os.makedirs(log_path, exist_ok=True)

    resume = _resume_enabled()
    if resume:
        print('[续跑] 已开启：存在 model_fold_<n>.pth 的折会被跳过')

    if config.train.cross_validation:
        log_file = os.path.join(log_path, 'cross_fold_summary.txt')
        # buffering=1 = 行缓冲：每折写完立刻落盘，另开终端 tail -f 能实时看到
        # 续跑用追加（不清掉上次的记录），全新跑用覆盖（避免和旧结果混在一起）
        mode = 'a' if resume else 'w'
        skipped = 0
        with open(log_file, mode, encoding='utf-8', buffering=1) as log:
            for fold in range(config.train.cross_validation_folds):
                ckpt = os.path.join(log_path, f'model_fold_{fold}.pth')
                if resume and os.path.exists(ckpt):
                    print(f'[续跑] 第 {fold} 折的权重已存在，跳过')
                    skipped += 1
                    continue

                train_dataloader, validation_dataloader = get_dataloaders(config, kfold_number=fold)
                
                model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device, fold)
                model, best_value, rest_best_values = train(
                    model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler,
                    config.train.num_epochs, config.path_name, config.train.early_stopping, 
                    config.train.early_stopping_patience, config.train.cross_validation, fold,
                    early_stopping_metric=config.train.get('early_stopping_metric', 'loss'),
                    early_stopping_min_delta=float(config.train.get('early_stopping_min_delta', 0.0))
                )
                
                log.write(f'Fold {fold}: Best Value = {best_value}\n')
                log.write(f'Best F1: {rest_best_values[0]}\nBest Recall: {rest_best_values[1]}\nBest Precision: {rest_best_values[2]}\n')
                
                torch.save(model.state_dict(), ckpt)
                print(f'Model for fold {fold} saved')
                wandb.log({
                    "best_value": best_value,
                    "best_f1": rest_best_values[0],
                    "best_recall": rest_best_values[1],
                    "best_precision": rest_best_values[2],
                })
                wandb.finish()

        if resume and skipped == config.train.cross_validation_folds:
            print(f'[续跑] 5 折的权重都在，没有要跑的了 —— 结果见 {log_file}')
    else:
        model_save_path = os.path.join(log_path, 'model.pt')
        if resume and os.path.exists(model_save_path):
            print(f'[续跑] {model_save_path} 已存在，跳过（不跑交叉验证时只有一个模型）')
            return

        train_dataloader, validation_dataloader = get_dataloaders(config)
        
        model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device)
        model, best_value, rest_best_values = train(
            model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler, 
            config.train.num_epochs, config.path_name, config.train.early_stopping, 
            config.train.early_stopping_patience,
            early_stopping_metric=config.train.get('early_stopping_metric', 'loss'),
            early_stopping_min_delta=float(config.train.get('early_stopping_min_delta', 0.0))
        )
        
        torch.save(model.state_dict(), model_save_path)
        print('Model saved')
        wandb.finish()


if __name__ == '__main__':

    config_path = sys.argv[sys.argv.index('--config') + 1]
    config = get_config(config_path)
    """
    for model_name in ['qwen']:
            config.model_name = model_name
            config.model.model_name = config.model_name

            for fusion in ['crossgated']:
                config.model.fusion = fusion

                for pooling in ['mean', 'cls']:
                    config.model.pooling = pooling
    """
    save_config(config)    
    main(config)