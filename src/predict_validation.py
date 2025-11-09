import torch
from model.encoder import Encoder
from data.data_loader import get_data_loader
from utils import predict_from_validation_set

# 配置（与训练时一致）
CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "d_ff": 512,
    "n_layers": 2,
    "max_seq_len": 256,
    "dropout": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "checkpoints",
    "use_pos_encoding": True
}


def main():
    # 1. 加载验证集（用于获取数据和tokenizer）
    _, val_dataset = get_data_loader(
        split="validation",
        batch_size=1,
        seq_len=CONFIG["max_seq_len"],
        shuffle=False
    )
    vocab_size = val_dataset.tokenizer.vocab_size

    # 2. 加载训练好的模型
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

    # 加载模型权重
    checkpoint = torch.load(f"{CONFIG['save_dir']}/best_model_ptb.pth", map_location=CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    print("模型加载完成！")

    # 3. 从验证集中选取第0个样本进行预测
    input_text, generated_part = predict_from_validation_set(
        model=model,
        val_dataset=val_dataset,
        idx=0,  # 验证集中的样本索引
        max_gen_len=10,  # 生成10个后续token
        device=CONFIG["device"]
    )

    # 4. 打印结果
    print("\n===== 预测结果 =====")
    print(f"输入（前50个token）：{input_text}")
    print(f"预测（后续10个token）：{generated_part}")


if __name__ == "__main__":
    main()
