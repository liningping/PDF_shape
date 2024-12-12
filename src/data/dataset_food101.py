import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from collections import Counter

from transformers import BertTokenizer

from src.data.vocab import Vocab

class Args:
    def __init__(self, max_seq_len, labels):
        self.max_seq_len = max_seq_len
        self.labels = labels

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.46777044, 0.44531429, 0.40661017],
            std=[0.12221994, 0.12145835, 0.14380469],
        ),
    ])

def get_vocab():
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    # print(path)
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs

class Food101Dataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.vocab = vocab
        self.args = args
        self.text_start_token = ["[CLS]"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        tokenized_text = self.tokenizer(data["text"])
        sentence = self.text_start_token + tokenized_text[:self.args.max_seq_len - 1]
        segment = torch.zeros(len(sentence))
        
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        label = torch.LongTensor(
            [self.args.labels.index(self.data[index]["label"])]
        )

        img = None

        if self.data[index]["img"]:
            img_path = os.path.join("/home/zrb/study/PDF/datasets/food101", self.data[index]["img"])
            image = Image.open(
                img_path
            ).convert("RGB")
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = self.transforms(image)

        return sentence, segment, image, label, torch.LongTensor([index])


# 假设你有一个分词器和词汇表
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True).tokenize
vocab = get_vocab()

transforms = get_transforms()

# 数据集路径
data_path = '/home/zrb/study/PDF/datasets/food101/train.jsonl'
labels, label_freqs = get_labels_and_frequencies(
        os.path.join(data_path)
    )
# 其他参数
args = Args(max_seq_len=128, labels=labels)

# 创建数据集实例
dataset = Food101Dataset(data_path, tokenizer, transforms, vocab, args)

# 获取数据集的长度
print(len(dataset))

# 获取数据集的一个样本
sample = dataset[0]
print(sample)
