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

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=window_size, padding=0),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
        )

        self.reg = nn.Linear(in_channels, 4*k)
        self.cls = nn.Linear(in_channels, 1*k)

    def forward(self, x):
        """
        :param x: torch.FloatTensor of size (B x C x window_size x window_size).
        :return:    torch.FloatTensor of size (B x C x window_size x 4*k), torch.FloatTensor of size (B x C x window_size x k)
                    representing the bbox regression and the confidence of the proposal.
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        reg = self.reg(x)
        cls = self.cls(x)

        return reg, cls