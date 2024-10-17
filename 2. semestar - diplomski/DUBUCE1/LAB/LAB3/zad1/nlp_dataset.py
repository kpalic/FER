from torch.utils.data import Dataset
from zad1.instance import Instance
from zad1.vocab import Vocab
from typing import List

class NLPDataset(Dataset):
    def __init__(self, instances: List[Instance], text_vocab: Vocab, label_vocab: Vocab):
        self.instances = instances
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        instance = self.instances[idx]
        text = self.text_vocab.encode(instance.text)
        label = self.label_vocab.stoi.get(instance.label, -1)
        return text, label

    @staticmethod
    def from_file(filepath: str, text_vocab: Vocab, label_vocab: Vocab):
        instances = []
        with open(filepath, 'r') as file:
            for line in file:
                text, label = line.strip().split(',')
                text = text.strip()  
                label = label.strip()
                instances.append(Instance(text.split(), label))
        return NLPDataset(instances, text_vocab, label_vocab)
