import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.ops import RoIPool, batched_nms, nms

from lib.anchor_generator import AnchorGenerator
from lib.cnn_base import CNN
from lib.rcnn import RCNN
from lib.rpn import RPN
from lib.sliding_window import SlidingWindow

import timy


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, anchor_boxes):
        super(FasterRCNN, self).__init__()
        fm_channels = 2048 // 2
        window_size = 3
        self.reduction = 1/16
        self.n_proposals = 300

        self.cnn = CNN()
        self.rpn = RPN(fm_channels, len(anchor_boxes), window_size)
        self.rcnn = RCNN(fm_channels, num_classes)
        self.sliding_window = SlidingWindow(window_size, stride=1)
        self.agn = AnchorGenerator(anchor_boxes, window_size)
        self.roi_pooling = RoIPool(output_size=(7, 7), spatial_scale=self.reduction)

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
        anchors = torch.unsqueeze(anchors, dim=0).expand_as(rpn_reg)

        # 4. Apply Non-Maximum-Suppression
        rpn_reg_absolute = (rpn_reg + anchors)

        nms_reg = []
        nms_cls = []
        for b in range(batch_size):
            b_rpn_reg_absolute = rpn_reg_absolute[b]
            b_rpn_cls = rpn_cls[b]
            
            # NMS for RPN for all bboxes with IoU overlap > 0.7
            keep = nms(b_rpn_reg_absolute, torch.sigmoid(b_rpn_cls), 0.7)

            r = b_rpn_reg_absolute[keep][:self.n_proposals]
            c = b_rpn_cls[keep][:self.n_proposals]

            c_sorted, c_indices = torch.sort(b_rpn_cls, descending=True)
            r_sorted = b_rpn_reg_absolute[c_indices]

            c_sorted = c_sorted[:max(0, self.n_proposals - len(c))]
            r_sorted = r_sorted[:max(0, self.n_proposals - len(c))]

            nms_reg.append(torch.cat([r, r_sorted], 0))
            nms_cls.append(torch.cat([c, c_sorted], 0))
        
        nms_reg = torch.stack(nms_reg, 0)
        nms_cls = torch.stack(nms_cls, 0)
        
        # 5. Apply Region of Interest-Pooling
        nms_reg_rounded = nms_reg[:, :, [1, 0, 3, 2]]
        nms_reg_rounded = torch.cat([torch.floor(nms_reg_rounded[:, :, [0, 1]] * self.reduction), torch.ceil(nms_reg_rounded[:, :, [2, 3]] * self.reduction)], dim=2) / self.reduction
        roi_pooled = self.roi_pooling(fm, [r for r in nms_reg_rounded])

        # 6. Apply the RCNN for bbox regression and classification
        rcnn_reg, rcnn_cls = self.rcnn(roi_pooled)
        anchors = self.agn(img, fm)

        rcnn_reg = rcnn_reg.view(batch_size, -1, rcnn_reg.shape[1])
        rcnn_cls = rcnn_cls.view(batch_size, -1, rcnn_cls.shape[1])

        return rpn_reg, rpn_cls, nms_reg, nms_cls, rcnn_reg, rcnn_cls, anchors