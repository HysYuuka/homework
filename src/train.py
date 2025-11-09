import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import math
import argparse
from tqdm import tqdm
from model.encoder import Encoder
from data.data_loader import get_data_loader, create_padding_mask
from utils import save_model, plot_curves

# 硬编码配置参数
CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "d_ff": 512,
    "n_layers": 4,
    "max_seq_len": 256,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "checkpoints",
    "log_dir": "results",
    "seed": 42,
    "use_pos_encoding": True
}

# 固定随机种子
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG["seed"])


def train_one_epoch(model, data_loader, dataset, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(data_loader, desc="Training (PTB)")
    pad_token_id = dataset.pad_token_id

    for input_ids, label_ids in pbar:
        input_ids, label_ids = input_ids.to(device), label_ids.to(device)
        batch_size = input_ids.size(0)

        mask = create_padding_mask(input_ids, pad_token_id)
        logits = model(input_ids, mask)

        loss = criterion(logits.reshape(-1, logits.size(-1)), label_ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / len(data_loader.dataset)


def evaluate(model, data_loader, dataset, criterion, device):
    model.eval()
    total_loss = 0.0
    pad_token_id = dataset.pad_token_id
    with torch.no_grad():
        for input_ids, label_ids in tqdm(data_loader, desc="Evaluating (PTB)"):
            input_ids, label_ids = input_ids.to(device), label_ids.to(device)
            mask = create_padding_mask(input_ids, pad_token_id)
            logits = model(input_ids, mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), label_ids.reshape(-1))
            total_loss += loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss, math.exp(avg_loss)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true", help="仅评估已训练模型，不进行训练")
    parser.add_argument("--model_path", type=str, help="评估时使用的模型路径（配合--eval_only使用）")
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    # 加载数据（训练/验证集共用同一批配置）
    val_loader, val_dataset = get_data_loader(
        split="validation",
        batch_size=CONFIG["batch_size"],
        seq_len=CONFIG["max_seq_len"],
        shuffle=False
    )
    vocab_size = val_dataset.tokenizer.vocab_size  # 从验证集获取词汇表大小

    # 初始化模型
    model = Encoder(
        vocab_size=vocab_size,
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
        use_pos_encoding=CONFIG["use_pos_encoding"]
    ).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss(ignore_index=val_dataset.pad_token_id)

    # 仅评估模式
    if args.eval_only:
        if not args.model_path:
            raise ValueError("请通过 --model_path 指定模型权重路径（例如：checkpoints/best_model_ptb.pth）")
        # 加载模型权重
        checkpoint = torch.load(args.model_path, map_location=CONFIG["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载模型：{args.model_path}")

        # 执行评估
        val_loss, val_ppl = evaluate(model, val_loader, val_dataset, criterion, CONFIG["device"])
        print(f"评估结果：Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        return

    # 训练模式（默认流程）
    train_loader, train_dataset = get_data_loader(
        split="train",
        batch_size=CONFIG["batch_size"],
        seq_len=CONFIG["max_seq_len"]
    )
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # 训练循环
    best_val_loss = float("inf")
    train_losses, val_losses, val_ppls = [], [], []
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"配置：头数={CONFIG['n_heads']}，位置编码={'启用' if CONFIG['use_pos_encoding'] else '禁用'}")

        train_loss = train_one_epoch(model, train_loader, train_dataset, criterion, optimizer, CONFIG["device"])
        val_loss, val_ppl = evaluate(model, val_loader, val_dataset, criterion, CONFIG["device"])
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, best_val_loss, CONFIG["save_dir"])

    # 训练结束后绘制曲线
    plot_curves(train_losses, val_losses, val_ppls, CONFIG["log_dir"])
    print(f"训练完成！结果保存至 {CONFIG['log_dir']}")


if __name__ == "__main__":
    main()
