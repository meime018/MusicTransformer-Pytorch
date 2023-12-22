from dataset.e_piano import process_midi
import torch

raw_mid = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
max_seq = 3
random_seq = False
print(raw_mid)
x1, x2, tgt = process_midi(raw_mid, max_seq, random_seq)
print(x1, x2, tgt)

