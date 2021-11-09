import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CNN(nn.Module):
    """
    In order to ensure that the output sentence length = the input sentence length in nlp, an odd kernel_size is generally used, such as [3, 5, 7, 9]. 
    Of course, it can also be output with different lengths. 
    Keep_length is set to False. 
    At this time, padding = k / / 2, stride is generally 1
    """
    def __init__(self, config):
        """
        Args:
            in_channels: Generally it is the dimension of word embedding, or the dimension of hidden size
            out_channels: int
            kernel_sizes: In order to ensure that the output length = the input length, list must be an odd number: 3, 5, 7...
            activation: [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]
            pooling_strategy : [max, avg, cls]
            dropout: float
        """
        super(CNN, self).__init__()

        # self.xxx = config.xxx
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_sizes = config.kernel_sizes
        self.activation = config.activation
        self.pooling_strategy = config.pooling_strategy
        self.dropout = config.dropout
        self.keep_length = config.keep_length
        for kernel_size in self.kernel_sizes:
            assert kernel_size % 2 == 1, "kernel size has to be odd numbers."

        # convolution
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      stride=1,
                      padding=k // 2 if self.keep_length else 0,
                      dilation=1,
                      groups=1,
                      bias=False) for k in self.kernel_sizes
        ])

        # activation function
        assert self.activation in ['relu', 'lrelu', 'prelu', 'selu', 'celu', 'gelu', 'sigmoid', 'tanh'], \
            'activation function must choose from [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]'
        self.activations = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['selu', nn.SELU()],
            ['celu', nn.CELU()],
            ['gelu', GELU()],
            ['sigmoid', nn.Sigmoid()],
            ['tanh', nn.Tanh()],
        ])

        # pooling
        assert self.pooling_strategy in ['max', 'avg', 'cls'], 'pooling strategy must choose from [max, avg, cls]'

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H] Generally the value after embedding
            mask: [batch_size, max_len], The sentence length part is 0, and the padding part is 1. Does not affect the convolution operation, max-pool will not pool to the position where pad is 0
        """
        # [B, L, H] -> [B, H, L] 
        x = torch.transpose(x, 1, 2)

        # convolution + activation  [[B, H, L], ... ]
        act_fn = self.activations[self.activation]

        x = [act_fn(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)

        # mask
        if mask is not None:
            # [B, L] -> [B, 1, L]
            mask = mask.unsqueeze(1)
            x = x.masked_fill_(mask, 1e-12)

        # pooling
        # [[B, H, L], ... ] -> [[B, H], ... ]
        if self.pooling_strategy == 'max':
            xp = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)

        elif self.pooling_strategy == 'avg':
            x_len = mask.squeeze().eq(0).sum(-1).unsqueeze(-1).to(torch.float).to(device=mask.device)
            xp = torch.sum(x, dim=-1) / x_len

        else:
            # self.pooling_strategy == 'cls'
            xp = x[:, :, 0]

        x = x.transpose(1, 2)
        x = self.dropout(x)
        xp = self.dropout(xp)

        return x, xp  # [B, L, Hs], [B, Hs]
