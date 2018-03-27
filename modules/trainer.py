import torch
from torch import nn as nn, autograd as autograd

import helper
from metrics import CategoricalAccuracy

top = 0
left = 1
bottom = 2
right = 3


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn

        self.rpn_trainer = RPNTrainer()
        self.rcnn_trainer = RCNNTrainer(self.faster_rcnn.reduction)

    def forward(self, img, img_id, bboxes, classes):
        rpn_reg, rpn_cls, nms_reg, nms_cls, rcnn_reg, rcnn_cls, anchors = self.faster_rcnn(img, img_id)

        rpn_cls_loss, rpn_reg_loss = self.rpn_trainer(rpn_reg, rpn_cls, anchors, bboxes)
        rcnn_cls_loss, rcnn_reg_loss, accuracy = self.rcnn_trainer(nms_reg, nms_cls, rcnn_reg, rcnn_cls, bboxes, classes)
        rcnn_reg_loss = autograd.Variable(rpn_reg.new([0]))
        rcnn_cls_loss = autograd.Variable(rpn_reg.new([0]))
        accuracy = autograd.Variable(rpn_reg.new([0]))

        return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, accuracy


class RCNNTrainer(nn.Module):
    def __init__(self, reduction):
        super(RCNNTrainer, self).__init__()
        self.reduction = reduction

        self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

    def forward(self, nms_reg, nms_cls, rcnn_reg, rcnn_cls, bboxes, classes):
        batch_size = bboxes.size()[0]
        num_targets = bboxes.size()[1]
        num_reg = rcnn_reg.size()[1]
        num_classes = rcnn_cls.size()[-1]

        reg_expanded = torch.unsqueeze(rcnn_reg, dim=1).expand(
            batch_size,
            num_targets,
            num_reg,
            4
        )
        reg_expanded = reg_expanded.contiguous().view(-1, 4)

        nms_reg_expanded = torch.unsqueeze(nms_reg, dim=1).expand(
            batch_size,
            num_targets,
            num_reg,
            4
        )
        nms_reg_expanded = nms_reg_expanded.contiguous().view(-1, 4)

        bboxes_expanded = torch.unsqueeze(bboxes, dim=2).expand(
            batch_size,
            num_targets,
            num_reg,
            4
        )
        bboxes_expanded = bboxes_expanded.contiguous().view(-1, 4)

        rcnn_cls_expanded = torch.unsqueeze(rcnn_cls, dim=1).expand(
            batch_size,
            num_targets,
            num_reg,
            num_classes
        )
        rcnn_cls_expanded = rcnn_cls_expanded.contiguous().view(-1, num_classes)

        classes_expanded = torch.unsqueeze(classes, dim=2).expand(
            batch_size,
            num_targets,
            num_reg
        )
        classes_expanded = classes_expanded.contiguous().view(-1)

        # Calculate IoU
        iou = helper.calculate_iou(nms_reg_expanded, bboxes_expanded.float())
        iou = iou.view(
            batch_size,
            num_targets,
            num_reg
        )

        sorted_iou, indices = torch.sort(iou, dim=1)
        indices_expanded = torch.unsqueeze(indices, dim=-1).expand(-1, -1, -1, 4)
        indices_expanded_cls = torch.unsqueeze(indices, dim=-1).expand(-1, -1, -1, num_classes)

        max_reg = torch.gather(reg_expanded.view(batch_size, num_targets, num_reg, 4), 1, indices_expanded)[:, -1, :, :]
        max_bboxes = torch.gather(bboxes_expanded.view(batch_size, num_targets, num_reg, 4), 1, indices_expanded)[:, -1, :, :]

        max_rcnn_cls = torch.gather(rcnn_cls_expanded.view(batch_size, num_targets, num_reg, num_classes), 1, indices_expanded_cls)[:, -1, :, :]
        max_classes = torch.gather(classes_expanded.view(batch_size, num_targets, num_reg), 1, indices)[:, -1, :]

        max_iou = sorted_iou[:, -1, :]

        mask_positive = max_iou > 0.7

        reg_mask = torch.unsqueeze(mask_positive, dim=2).expand(-1, -1, 4)
        cls_mask = torch.unsqueeze(mask_positive, dim=2).expand(-1, -1, num_classes)

        masked_reg = max_reg[reg_mask].view(-1, 4)
        masked_bboxes = max_bboxes[reg_mask].view(-1, 4).float()

        reg_loss = autograd.Variable(masked_reg.new([0]))
        cls_loss = autograd.Variable(masked_reg.new([0]))
        accuracy = autograd.Variable(masked_reg.new([0]))

        if len(mask_positive.nonzero()) > 0:
            max_classes = max_classes[mask_positive].contiguous().view(-1)
            max_rcnn_cls = max_rcnn_cls[cls_mask].contiguous().view(-1, num_classes)

            cls_loss += self.cls_criterion(
                max_rcnn_cls,
                max_classes
            )

            accuracy += self.accuracy(
                max_rcnn_cls,
                max_classes
            )

            rounded_masked_bboxes = torch.round(masked_bboxes / self.reduction) * self.reduction

            nms_height = rounded_masked_bboxes[:, bottom] - rounded_masked_bboxes[:, top]
            nms_width = rounded_masked_bboxes[:, right] - rounded_masked_bboxes[:, left]

            reg_loss += self.reg_criterion(masked_reg[:, top], (masked_bboxes[:, top] - rounded_masked_bboxes[:, top]) / nms_height)
            reg_loss += self.reg_criterion(masked_reg[:, left], (masked_bboxes[:, left] - rounded_masked_bboxes[:, left]) / nms_width)
            reg_loss += self.reg_criterion(masked_reg[:, bottom], (masked_bboxes[:, bottom] - rounded_masked_bboxes[:, bottom]) / nms_height)
            reg_loss += self.reg_criterion(masked_reg[:, right], (masked_bboxes[:, right] - rounded_masked_bboxes[:, right]) / nms_width)

        return cls_loss, reg_loss, accuracy


class RPNTrainer(nn.Module):
    def __init__(self):
        super(RPNTrainer, self).__init__()

        self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def forward(self, reg, cls, anchors, targets):
        # Monitoring
        batch_size = targets.size()[0]
        num_targets = targets.size()[1]
        num_anchors = anchors.size()[0]

        targets = targets.view(-1, 4)

        anchors_expanded = torch.unsqueeze(anchors, dim=0).expand(targets.size()[0], anchors.size()[0],
                                                                  anchors.size()[1])
        anchors_expanded = anchors_expanded.contiguous().view(-1, 4)

        targets_expanded = torch.unsqueeze(targets, dim=1).expand(targets.size()[0], anchors.size()[0], 4)
        targets_expanded = targets_expanded.contiguous().view(-1, 4)

        iou = helper.calculate_iou(anchors_expanded, targets_expanded)
        iou = iou.view(batch_size, -1, anchors.size()[0])

        # Choose positive/negative samples as described in the paper
        # For each anchor find the target with the highest IoU
        # 1. Sort the targets (dim 1)
        sorted_iou, indices = torch.sort(iou, dim=1)
        indices_expanded = torch.unsqueeze(indices, dim=-1).expand(-1, -1, -1, 4)

        max_target = torch.gather(targets_expanded.view(batch_size, num_targets, num_anchors, 4), 1, indices_expanded)[
                     :, -1, :, :]
        max_iou = sorted_iou[:, -1, :].contiguous()

        anchors = torch.unsqueeze(anchors, dim=0).expand(batch_size, -1, -1)

        mask_positive = max_iou > 0.5
        mask_negative = max_iou < 0.5
        anchor_cls_positive = torch.unsqueeze(mask_positive, dim=2).expand(-1, -1, 4)

        masked_anchor = anchors[anchor_cls_positive].view(-1, 4)
        masked_targets = max_target[anchor_cls_positive].view(-1, 4)
        masked_reg = reg[anchor_cls_positive.cuda()].view(-1, 4)

        # Loss
        cls_loss = autograd.Variable(masked_reg.new([0]))
        reg_loss = autograd.Variable(masked_reg.new([0]))
        if anchor_cls_positive.nonzero().size()[0] > 0:
            reg_loss += self.reg_criterion(masked_reg[:, 0], (masked_targets[:, 0] - masked_anchor[:, 0]).float())
            reg_loss += self.reg_criterion(masked_reg[:, 1], (masked_targets[:, 1] - masked_anchor[:, 1]).float())
            reg_loss += self.reg_criterion(masked_reg[:, 2], (masked_targets[:, 2] - masked_anchor[:, 2]).float())
            reg_loss += self.reg_criterion(masked_reg[:, 3], (masked_targets[:, 3] - masked_anchor[:, 3]).float())

        cls = torch.cat([
            cls[mask_positive],
            cls[mask_negative],
        ])

        anchor_cls = torch.cat([
            mask_positive[mask_positive],
            mask_positive[mask_negative],
        ])

        cls_loss += self.cls_criterion(cls, anchor_cls.float())

        return cls_loss, reg_loss / num_anchors