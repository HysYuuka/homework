import torch
import matplotlib.pyplot as plt
import os

from data.data_loader import create_padding_mask


def save_model(model, optimizer, epoch, val_loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, "best_model_ptb.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": {"d_model": model.d_model, "n_heads": model.encoder_blocks[0].self_attn.n_heads}  # 保存关键超参
    }, save_path)
    print(f"PTB最优模型已保存至: {save_path}")


def plot_curves(train_losses, val_losses, val_ppls, log_dir):
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    epochs = range(1, len(train_losses) + 1)
    # 创建子图（1行2列：损失曲线+困惑度曲线）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 子图1：训练/验证损失曲线
    ax1.plot(epochs, train_losses, label="PTB训练损失", color="blue", linewidth=2)
    ax1.plot(epochs, val_losses, label="PTB验证损失", color="red", linewidth=2)
    ax1.set_xlabel("训练轮次（Epoch）")
    ax1.set_ylabel("损失值（Loss）")
    ax1.set_title("PTB数据集训练与验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：验证集困惑度曲线
    ax2.plot(epochs, val_ppls, label="PTB验证困惑度", color="green", linewidth=2)
    ax2.set_xlabel("训练轮次（Epoch）")
    ax2.set_ylabel("困惑度（Perplexity）")
    ax2.set_title("PTB数据集验证集困惑度曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 保存图片
    save_path = os.path.join(log_dir, "ptb_train_val_curves.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"PTB训练曲线已保存至: {save_path}")


def predict_from_validation_set(model, val_dataset, idx=0, max_gen_len=10, device="cuda"):
    """
    从验证集中选取指定索引的样本作为输入，生成后续文本
    Args:
        model: 训练好的模型
        val_dataset: 验证集数据集（PTBDataset实例）
        idx: 验证集中的样本索引
        max_gen_len: 生成的后续token数量
        device: 运行设备
    Returns:
        input_text: 输入的文本片段
        generated_text: 生成的完整文本（输入+预测）
    """
    model.eval()
    with torch.no_grad():
        # 1. 从验证集中获取输入token（取样本的input_ids作为起始文本）
        input_ids, _ = val_dataset[idx]  # input_ids是长度为seq_len的张量
        input_ids = input_ids.unsqueeze(0).to(device)  # 形状: (1, seq_len)

        # 2. 解码原始输入为文本（取前50个token作为示例输入）
        input_tokens = input_ids[0, :50].cpu().numpy()  # 取前50个token
        input_text = val_dataset.tokenizer.decode(input_tokens, skip_special_tokens=True)

        # 3. 自回归生成后续token
        current_ids = input_ids[:, :50]  # 用前50个token作为起始输入
        for _ in range(max_gen_len):
            # 生成padding mask
            mask = create_padding_mask(current_ids, val_dataset.pad_token_id).to(device)
            # 模型预测
            logits = model(current_ids, mask)
            temperature = 0.7
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)  # 按概率采样
            # 拼接新token
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

        # 4. 解码生成的完整文本
        generated_ids = current_ids.squeeze(0).cpu().numpy()
        generated_text = val_dataset.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # 提取生成的部分（去除输入前缀）
        generated_part = generated_text[len(input_text):].strip()
        return input_text, generated_part
