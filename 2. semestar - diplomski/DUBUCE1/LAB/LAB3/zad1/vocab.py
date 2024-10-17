import torch
from typing import List, Union, Dict

class Vocab:
    def __init__(self, frequencies: Dict[str, int], max_size=-1, min_freq=1, specials=True):
        self.itos = []
        self.stoi = {}
        if specials:
            self.itos = ['<PAD>', '<UNK>']
            self.stoi = {'<PAD>': 0, '<UNK>': 1}

        sorted_tokens = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]), reverse=False)
        for token, freq in sorted_tokens:
            if freq < min_freq:
                continue
            if 0 < max_size <= len(self.itos):
                break
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, tokens: Union[List[str], str]) -> torch.Tensor:
        if isinstance(tokens, list):
            return torch.tensor([self.stoi.get(token, self.stoi['<UNK>']) for token in tokens])
        elif isinstance(tokens, str):
            return torch.tensor([self.stoi.get(tokens, self.stoi['<UNK>'])])

    def decode(self, indices: Union[List[int], int]) -> Union[List[str], str]:
        if isinstance(indices, list):
            return [self.itos[index] if index < len(self.itos) else '<UNK>' for index in indices]
        elif isinstance(indices, int):
            return self.itos[indices] if indices < len(self.itos) else '<UNK>'

# import torch
# from typing import List, Union

# class Vocab:
#     def __init__(self, frequencies, max_size=-1, min_freq=1, specials = True):
#         self.itos = []
#         self.stoi = {}
#         if (specials):
#             self.itos = ['<PAD>', '<UNK>']
#             self.stoi = {'<PAD>': 0, '<UNK>': 1}

#         sorted_tokens = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]), reverse=False)
#         for token, freq in sorted_tokens:
#             if freq < min_freq:
#                 break
#             if len(self.itos) >= max_size > 0:
#                 break
#             self.stoi[token] = len(self.itos)
#             self.itos.append(token)

#     def encode(self, tokens: Union[List[str], str]) -> torch.Tensor:
#         if isinstance(tokens, list):
#             return torch.tensor([self.stoi.get(token, 1) for token in tokens])
#         elif isinstance(tokens, str):
#             return torch.tensor([self.stoi.get(tokens, -1)])


#     def decode(self, indices: Union[List[int], int]) -> Union[List[str], str]:
#         if isinstance(indices, list):
#             return [self.itos[index] for index in indices]
#         elif isinstance(indices, int):
#             return self.itos[indices]
