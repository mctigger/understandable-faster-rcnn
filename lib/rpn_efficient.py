from torch import nn as nn


class RPN(nn.Module):
    """
    The Region Proposal Network finds interesting regions in an image, which will then be further inspected by the RCNN.
    """
    def __init__(self, in_channels, k, window_size):
        """
        :param in_channels: Integer. Number of channels of the given feature map.
        :param k: Integer. Number of anchors.
        :param window_size: Integer. The size of the sliding window the RPN uses.
        """
        super(RPN, self).__init__()

        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=window_size, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, (4+1)*k, kernel_size=1)
        )

    def forward(self, x):
        """
        :param x: torch.FloatTensor of size (B x C x window_size x window_size).
        :return:    torch.FloatTensor of size (B x C x window_size x 4*k), torch.FloatTensor of size (B x C x window_size x k)
                    representing the bbox regression and the confidence of the proposal.
        """
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, 5)
        reg = x[:, :4]
        cls = x[:, 4]

        return reg, cls