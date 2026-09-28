from dataset import get_dataloaders
from utils import set_seed, get_config, train, save_config
from model import CrossAttentionTransformerEncoder, MyTransformerEncoder, BidirectionalCrossAttentionTransformerEncoder, ElementWiseFusionEncoder
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
    set_seed(42)
    
    if config.model.multimodality:
        if 'bicross' in config.model.fusion:
            model = BidirectionalCrossAttentionTransformerEncoder(config.model).to(device)
        elif 'cross' in config.model.fusion:
            model = CrossAttentionTransformerEncoder(config.model).to(device)
        else:
            model = ElementWiseFusionEncoder(config.model).to(device)
    else:
         model = MyTransformerEncoder(config.model).to(device)


    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    lossfn = nn.BCEWithLogitsLoss()
    
    num_training_steps = config.train.num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=num_training_steps
    )

    
    wandb.init(
        project="WordLevelFusion",
        name=f"{config.model_name}_{fold}" if config.train.cross_validation else config.model_name,
        config={
            "learning_rate": config.train.learning_rate,
            "architecture": config.model_name,
            "dataset": "ADReSSo",
            "epochs": config.train.num_epochs,
            "batch_size": config.train.batch_size,
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
                    config.train.early_stopping_patience, config.train.cross_validation, fold
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
            config.train.early_stopping_patience
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