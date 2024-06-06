import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

def check_shape(x, y):
    if x.shape != y.shape:
        raise RuntimeError(f"mismatched shape between tensors ({x.shape} vs {y.shape})")


def length_to_attention_mask(lens, reverse: bool = False):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) < lens.view(bsz, 1).expand(-1, max_lens)
    if reverse:
        mask = (~mask).to(torch.long)
    else:
        mask = mask.to(torch.long)

    return mask

def length_to_4d_attention_mask(lens):
    # Assuming src_lens is your length matrix of shape (B,)
    B = lens.size(0)
    max_len = lens.max()

    # Create a range tensor of shape (max_len,)
    range_tensor = torch.arange(max_len, device=lens.device).expand(B, max_len)

    # Compare the range tensor with src_lens to create the mask
    mask = range_tensor < lens.unsqueeze(1)

    # Use broadcasting to create the full 4D attention mask
    attention_mask = mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(-1)

    attention_mask = (1 - attention_mask.float()) * -1000
    return attention_mask
