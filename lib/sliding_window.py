import torch
from torch import nn as nn


class SlidingWindow(nn.Module):
    """
    from https://discuss.pytorch.org/t/unlabeled-object-localization/1709

    Slides a window of over the input feature map more efficiently than with for-loops
    """
    def __init__(self, window_size, stride):
        """

        :param window_size (int): Size of the sliding window
        :param stride (int): Stride of the sliding window
        """
        super(SlidingWindow, self).__init__()
        self.window_size = window_size
        self.stride = stride

    def forward(self, fm):
        """
        :param fm (torch.FloatTensor): Tensor of size (batch_size x channels x height x width)
        :return torch.FloatTensor: Tensor of size (batch_size x num_windows x channels x windows_size x windows_size)
        """
        batch_size, channels = fm.size()[0], fm.size()[1]
        fm = fm.unfold(2, self.window_size, self.stride)
        fm = fm.unfold(3, self.window_size, self.stride)
        fm = torch.transpose(fm, 1, 2)
        fm = torch.transpose(fm, 2, 3)
        fm = fm.contiguous()
        fm = fm.view(batch_size, -1, channels, self.window_size, self.window_size)
        return fm