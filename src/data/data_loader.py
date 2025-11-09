from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset


class PTBDataset(Dataset):
    def __init__(self, split="train", seq_len=256):
        super().__init__()
        self.seq_len = seq_len
        # 加载PTB数据集
        self.dataset = load_dataset("ptb_text_only", split=split, trust_remote_code=True)
        # 初始化tokenizer并设置pad_token
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 用eos_token作为pad_token
        self.pad_token_id = self.tokenizer.pad_token_id  # 缓存pad_token_id

        # 收集所有token
        self.full_tokens = []
        for sample in self.dataset:
            sentence = sample["sentence"].strip()
            if not sentence:
                continue
            # 分词（不添加特殊token）
            sent_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            self.full_tokens.extend(sent_tokens)

        # 截断到整数倍seq_len
        self.num_samples = len(self.full_tokens) // self.seq_len
        self.full_tokens = self.full_tokens[:self.num_samples * self.seq_len]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.full_tokens[start:end]
        label_ids = self.full_tokens[start + 1: end + 1]  # 偏移一位作为标签

        # 确保标签长度正确（最后一个样本可能不足）
        if len(label_ids) < self.seq_len:
            label_ids += [self.pad_token_id] * (self.seq_len - len(label_ids))

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def create_padding_mask(x, pad_token_id):

    # 生成 (batch_size, seq_len) 的布尔张量（True表示非pad）
    mask = (x != pad_token_id)
    # 扩展维度至 (batch_size, 1, 1, seq_len)，再扩展为 (batch_size, 1, seq_len, seq_len)
    mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(1), -1)
    return mask.float()


def get_data_loader(split="train", batch_size=32, seq_len=256, shuffle=True):
    """返回数据加载器和对应的dataset（用于获取tokenizer）"""
    dataset = PTBDataset(split=split, seq_len=seq_len)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    return data_loader, dataset  # 同时返回data_loader和dataset
