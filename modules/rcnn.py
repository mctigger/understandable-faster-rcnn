from torch import nn as nn


class RCNN(nn.Module):
    """
    Refines the predictions of the RPN.
    """
    def __init__(self, in_channels, num_classes, in_size=(7, 7), num_channels=1024):
        """
        :param in_channels: Number of channels of the input feature map.
        :param num_classes: Number of target classes.
        :param in_size: Size of the input feauture map after RoI-Pooling
        :param num_channels: Number of channels of the RCNN.
        """
        super(RCNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels*in_size[0]*in_size[1], num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
        )

        self.reg = nn.Linear(num_channels, 4)
        self.cls = nn.Linear(num_channels, num_classes + 1)

    def forward(self, x):
        """
        :param x: torch.FloatTensor of size ()
        :return:
        """
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        reg = self.reg(x)
        cls = self.cls(x)

        return reg, cls