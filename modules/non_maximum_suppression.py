import torch
from torch import nn as nn, autograd as autograd

import helper


class NonMaximumSuppression(nn.Module):
    """
    Implements Non-Maximum-Suppression for bounding boxes.
    Calculates the IoU between every bounding box and removes all bounding boxes that overlap.
    Bounding boxes overlap if there IoU is greater than the specified iou_threshold.
    This is an inefficient implementation. A custom gpu/cuda implementation would be faster.
    """
    def __init__(self, iou_threshold=0.5, top=100):
        """
        :param iou_threshold: Float between 0 and 1.
        :param top: Integer greater 0. After nms take top best bounding boxes.
        """
        super(NonMaximumSuppression, self).__init__()
        self.iou_threshold = iou_threshold
        self.top = top

    def forward(self, reg, cls):
        """
        :param reg: torch.FloatTensor of size (BxPx4). RPN bbox regression.
        :param cls: torch.FloatTensor of size (BxP). RPN proposal confidence.
        :return: reg, cls but with removed bboxes all four reg values and the cls score set to 0.
        """
        batch_size = cls.size()[0]
        sorted_cls, sorted_cls_indices = torch.sort(cls, dim=1, descending=True)
        sorted_reg_indices = torch.unsqueeze(sorted_cls_indices, dim=2).expand_as(reg)
        sorted_reg = torch.gather(reg, dim=1, index=sorted_reg_indices)

        sorted_cls = sorted_cls[:, :self.top]
        sorted_reg = sorted_reg[:, :self.top, :]

        num_anchors = sorted_cls.size()[1]

        reg_a = torch.unsqueeze(sorted_reg, dim=2).expand(batch_size, num_anchors, num_anchors, 4).contiguous()
        reg_b = torch.unsqueeze(sorted_reg, dim=1).expand(batch_size, num_anchors, num_anchors, 4).contiguous()

        iou = helper.calculate_iou(reg_a.view(-1, 4), reg_b.view(-1, 4)).view(batch_size, num_anchors, num_anchors)

        mask = autograd.Variable(torch.ones((sorted_cls.size()[0], sorted_cls.size()[1])).byte().cuda())
        for i in range(sorted_cls.size()[1]):
            it_mask = torch.unsqueeze(mask[:, i], dim=1).expand_as(mask)
            tmp_mask = iou[:, i, :] < self.iou_threshold

            updated_mask = mask & tmp_mask
            updated_mask[:, i] = 1
            mask = updated_mask & it_mask | mask & (it_mask ^ 1)

        mask_reg = torch.unsqueeze(mask, dim=2).expand_as(sorted_reg).contiguous()

        sorted_reg = sorted_reg.clone()
        sorted_cls = sorted_cls.clone()
        sorted_reg[mask_reg ^ 1] = 0
        sorted_cls[mask ^ 1] = 0

        return sorted_reg, sorted_cls