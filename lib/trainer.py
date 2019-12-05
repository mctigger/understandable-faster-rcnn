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

        return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, accuracy


class RCNNTrainer(nn.Module):
    def __init__(self, reduction):
        super(RCNNTrainer, self).__init__()
        self.reduction = reduction

        self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

        self.rois_per_image = 64

    def forward(self, nms_reg, nms_cls, rcnn_reg, rcnn_cls, bboxes, classes):
        batch_size = bboxes.shape[0]
        num_targets = bboxes.shape[1]
        num_reg = rcnn_reg.shape[1]
        num_classes = rcnn_cls.shape[-1]

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

        # Calculate IoU
        iou = helper.calculate_iou(nms_reg_expanded, bboxes_expanded.float())
        iou = iou.view(
            batch_size,
            num_targets,
            num_reg
        )

        iou, indices = torch.max(iou, dim=1)

        # Take highest IoU-overlap target for each RoI
        mask_positive = (iou > 0.5).view(-1)
        mask_negative = (iou <= 0.5).view(-1)

        reg_loss = iou.new([0])
        cls_loss = iou.new([0])
        accuracy = iou.new([0])

        rcnn_cls_sampled = torch.cat([
            rcnn_cls.view(-1, num_classes)[mask_positive, :][:self.rois_per_image*batch_size * 1 // 4],
            rcnn_cls.view(-1, num_classes)[mask_negative, :][:self.rois_per_image*batch_size * 3 // 4]
        ], dim=0)

        classes_sampled = torch.cat([
            torch.gather(classes, 1, indices).view(-1)[mask_positive][:self.rois_per_image*batch_size * 1 // 4],
            torch.zeros(self.rois_per_image*batch_size * 3 // 4, dtype=torch.long).to(classes.device)
        ], dim=0)

        cls_loss += self.cls_criterion(rcnn_cls_sampled, classes_sampled)

        accuracy += self.accuracy(rcnn_cls_sampled, classes_sampled)
        
        if len(mask_positive.nonzero()) > 0:
            masked_bboxes = torch.gather(bboxes, 1, indices.unsqueeze(2).repeat(1, 1, 4)).view(-1, 4)[mask_positive, :][:self.rois_per_image*batch_size * 1 // 4]
            masked_reg = rcnn_reg.view(-1, 4)[mask_positive, :][:self.rois_per_image*batch_size * 1 // 4]
            masked_nms_reg = nms_reg.view(-1, 4)[mask_positive, :][:self.rois_per_image*batch_size * 1 // 4]

            rounded_masked_bboxes = torch.cat([torch.floor(masked_nms_reg[:, [0, 1]] * self.reduction), torch.ceil(masked_nms_reg[:, [2, 3]] * self.reduction)], dim=1) / self.reduction

            roi_height = torch.abs(rounded_masked_bboxes[:, bottom] - rounded_masked_bboxes[:, top])
            roi_width = torch.abs(rounded_masked_bboxes[:, right] - rounded_masked_bboxes[:, left])

            reg_loss += self.reg_criterion(masked_reg[:, top], (masked_bboxes[:, top] - rounded_masked_bboxes[:, top]))
            reg_loss += self.reg_criterion(masked_reg[:, left], (masked_bboxes[:, left] - rounded_masked_bboxes[:, left]))
            reg_loss += self.reg_criterion(masked_reg[:, bottom], (masked_bboxes[:, bottom] - rounded_masked_bboxes[:, bottom]))
            reg_loss += self.reg_criterion(masked_reg[:, right], (masked_bboxes[:, right] - rounded_masked_bboxes[:, right]))

        return cls_loss, reg_loss / 4, accuracy


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

        anchors_expanded = torch.unsqueeze(anchors, dim=0).expand(
            targets.size()[0],
            anchors.size()[0],
            anchors.size()[1]
        )
        anchors_expanded = anchors_expanded.contiguous().view(-1, 4)

        targets_expanded = torch.unsqueeze(targets, dim=1).expand(
            targets.size()[0],
            anchors.size()[0],
            4
        )
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

        mask_positive = (max_iou > 0.5)[:128]
        mask_negative = (max_iou <= 0.5)[:256 - len(mask_positive)]
        anchor_cls_positive = torch.unsqueeze(mask_positive, dim=2).expand(-1, -1, 4)

        masked_anchor = anchors[anchor_cls_positive].view(-1, 4)
        masked_targets = max_target[anchor_cls_positive].view(-1, 4)
        masked_reg = reg[anchor_cls_positive.cuda()].view(-1, 4)

        # Loss
        cls_loss = masked_reg.new([0])
        reg_loss = masked_reg.new([0])
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

        return cls_loss, reg_loss / 4