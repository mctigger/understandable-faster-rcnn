import torch
from torch import nn as nn

top = 0
left = 1
bottom = 2
right = 3


class FasterRCNNPredictor(nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNPredictor, self).__init__()
        self.faster_rcnn = faster_rcnn

    def forward(self, img, img_id):
        rpn_reg, rpn_cls, nms_reg, nms_cls, rcnn_reg, rcnn_cls, anchors = self.faster_rcnn(img, img_id)

        nms_reg_rounded = torch.round(nms_reg * self.faster_rcnn.reduction) // self.faster_rcnn.reduction

        nms_height = torch.abs(nms_reg_rounded[:, :, top] - nms_reg_rounded[:, :, bottom])
        nms_width = torch.abs(nms_reg_rounded[:, :, left] - nms_reg_rounded[:, :, right])

        rcnn_reg[:, :, top] = rcnn_reg[:, :, top] * nms_height
        rcnn_reg[:, :, left] = rcnn_reg[:, :, left] * nms_width
        rcnn_reg[:, :, bottom] = rcnn_reg[:, :, bottom] * nms_height
        rcnn_reg[:, :, right] = rcnn_reg[:, :, right] * nms_width

        rcnn_reg[:, :, top] = rcnn_reg[:, :, top] + nms_reg_rounded[:, :, top]
        rcnn_reg[:, :, left] = rcnn_reg[:, :, left] + nms_reg_rounded[:, :, left]
        rcnn_reg[:, :, bottom] = rcnn_reg[:, :, bottom] + nms_reg_rounded[:, :, bottom]
        rcnn_reg[:, :, right] = rcnn_reg[:, :, right] + nms_reg_rounded[:, :, right]

        return nms_reg, nms_cls, rcnn_reg, rcnn_cls