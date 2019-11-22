from torch import nn as nn
from torchvision import models as models


class CNN(nn.Module):
    """
    The CNN base network.

    :param torch.FloatTensor x: Tensor representing a batch of images of size (BxCxHxW)
    :return torch.FloatTensor: Embedding with a size of (BxOCxOHxOW) depending on the CNN
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.base = models.resnet50(pretrained=False)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        #x = self.base.layer4(x)

        return x