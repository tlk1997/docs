import torch
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Union

logger = logging.getLogger(__name__)

__all__ = [
    'manual_seed',
    'seq_len_to_mask',
    'to_one_hot',
]


def manual_seed(seed: int = 1) -> None:
    """
        Set seeds
        Args : 
            seed(int): The number of setting
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #if torch.cuda.CUDA_ENABLED and use_deterministic_cudnn:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def seq_len_to_mask(seq_len: Union[List, np.ndarray, torch.Tensor], max_len=None, mask_pos_to_true=True):
    """
    Convert a one-dimensional array representing sequence length to a two-dimensional mask, the default position of the pad is 1.
    Convert 1-d seq_len to 2-d mask.
    Args :
        seq_len (list, np.ndarray, torch.LongTensor) : shape will be (B,)
        max_len (int): Pad the length to this length. The default (None) uses the longest length in seq_len. But in the scenario of nn.DataParallel, the seq_len of different cards may be different, so you need to pass in a max_len so that the length of the mask is from pad to that length.
    Return: 
        mask (np.ndarray, torch.Tensor) : shape will be (B, max_length),the element is similar to bool or torch.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


def to_one_hot(x: torch.Tensor, length: int) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor):[B] ,is generally the value of target
        length (int) : L ,is generally the relationship tree.
    Return:
        x_one_hot.to(device=x.device) (torch.Tensor) : [B, L] ,In each row, only the corresponding position is 1, and the rest are 0.
    """
    B = x.size(0)
    x_one_hot = torch.zeros(B, length)
    for i in range(B):
        x_one_hot[i, x[i]] = 1.0

    return x_one_hot.to(device=x.device)