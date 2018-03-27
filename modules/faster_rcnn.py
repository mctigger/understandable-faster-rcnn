import torch
from torch import nn as nn

from modules.anchor_generator import AnchorGenerator
from modules.cnn_base import CNN
from modules.non_maximum_suppression import NonMaximumSuppression
from modules.rcnn import RCNN
from modules.roi_pooling import RoIPooling
from modules.rpn import RPN
from modules.sliding_window import SlidingWindow


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, anchor_boxes):
        super(FasterRCNN, self).__init__()
        fm_channels = 2048 // 2
        window_size = 3
        self.reduction = 16
        self.cnn = CNN()
        self.rpn = RPN(fm_channels, len(anchor_boxes), window_size)
        self.rcnn = RCNN(fm_channels, num_classes)
        self.sliding_window = SlidingWindow(window_size, stride=1)
        self.nms = NonMaximumSuppression(iou_threshold=0.3, top=20)
        self.agn = AnchorGenerator(anchor_boxes, window_size)
        self.roi_pooling = RoIPooling(reduction=self.reduction)

    def forward(self, img, img_id):
        # 1. Apply CNN base for feature extraction
        fm = self.cnn(img)
        batch_size, fm_channels, _, _ = fm.size()

        # 2. + 3. together equal to sliding the RPN over the feature map
        # 2. Apply sliding window to get slices
        slices = self.sliding_window(fm)

        # 3. Apply RPN for proposal bbox regression and binary classification (roi/non-roi)
        slices = slices.view(-1, fm_channels, slices.size()[-2], slices.size()[-1])
        rpn_reg, rpn_cls = self.rpn(slices)

        rpn_reg = rpn_reg.view(batch_size, -1, 4)
        rpn_cls = rpn_cls.view(batch_size, -1)

        # rpn_reg are bbox predictions of the RPN relative to some anchors
        # For non-maximum suppression we need the absolute position
        anchors = self.agn(img, fm)
        #anchors = anchors[:, 0:2]
        #anchors = anchors.repeat(1, 2)
        anchors = torch.unsqueeze(anchors, dim=0).expand_as(rpn_reg)

        # 4. Apply Non-Maximum-Suppression
        nms_reg, nms_cls = self.nms(rpn_reg + anchors, rpn_cls)

        # 5. Apply Region of Interest-Pooling
        roi_pooled = self.roi_pooling(fm, img_id, nms_reg)
        # 6. Apply the RCNN for bbox regression and classification
        rcnn_reg, rcnn_cls = self.rcnn(roi_pooled)

        # Restore tensor sizes to (batch_size, num_predictions, ...)
        rcnn_reg = rcnn_reg.view(batch_size, -1, 4)
        rcnn_cls = rcnn_cls.view(batch_size, -1, rcnn_cls.size()[-1])

        anchors = self.agn(img, fm)

        return rpn_reg, rpn_cls, nms_reg, nms_cls, rcnn_reg, rcnn_cls, anchors