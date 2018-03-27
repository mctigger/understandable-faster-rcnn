import torch
from torch import nn as nn

from roi_pooling.modules.roi_pool import _RoIPooling


class RoIPooling(nn.Module):
    """
    Wrapper for cuda-RoI-Pooling from https://github.com/jwyang/faster-rcnn.pytorch/tree/master/lib/model/roi_pooling.
    To make this work you have to compile https://github.com/jwyang/faster-rcnn.pytorch and then copy lib/model/roi_pooling
    to ./
    The reduction parameter is the factor of size reduction from the input to the output of the cnn_base.
    For example for a standard resnet with input 224 the last feature map is of size 7. This means the reduction factor
    is 32.
    """
    def __init__(self, reduction, out_size=(7, 7)):
        """
        :param reduction: Float scaling factor from the input of the base CNN to the output feature map of the base CNN.
        :param out_size: Output size of the RoI-pooled feature map.
        """
        super(RoIPooling, self).__init__()

        self.roi_pooling = _RoIPooling(out_size[0], out_size[1], 1.0/reduction)

    def forward(self, image, image_ids, rois):
        batch_size = rois.size()[0]
        num_rois = rois.size()[1]
        image_ids = torch.unsqueeze(image_ids, dim=1).expand(batch_size, num_rois).contiguous().view(-1, 1)
        rois = rois.view(-1, 4)
        rois = torch.cat([image_ids, rois], dim=1)
        output = self.roi_pooling(image, rois)

        return output