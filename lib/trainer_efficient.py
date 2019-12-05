import torch
from torch import nn as nn, autograd as autograd

import helper
from metrics import CategoricalAccuracy

top = 0
left = 1
bottom = 2
right = 3


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, n_anchors=256):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn

        self.rpn_trainer = RPNTrainer()
        self.rcnn_trainer = RCNNTrainer(self.faster_rcnn.reduction, faster_rcnn, n_anchors)

    def forward(self, img, bboxes, classes):
        # 1. Apply CNN base for feature extraction
        feature_maps = self.faster_rcnn.forward_backbone(img)
        batch_size, fm_channels, _, _ = feature_maps.size()

        # 2. + 3. together equal to sliding the RPN over the feature map
        # 2. Apply sliding window to get slices
        anchors = self.faster_rcnn.agn(img, feature_maps)
        rpn_reg, rpn_cls, rpn_reg_absolute = self.faster_rcnn.forward_rpn(feature_maps, anchors)

        # 4. Apply Non-Maximum-Suppression
        nms_reg, nms_cls = self.faster_rcnn.forward_nms(rpn_reg_absolute, rpn_cls)
        
        rpn_cls_loss, rpn_reg_loss = self.rpn_trainer(rpn_reg, rpn_cls, anchors, bboxes)
        rcnn_cls_loss, rcnn_reg_loss, accuracy, offset = self.rcnn_trainer(nms_reg, nms_cls, feature_maps, bboxes, classes)

        return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, accuracy, offset


class RCNNTrainer(nn.Module):
    def __init__(self, reduction, faster_rcnn, n_anchors):
        super(RCNNTrainer, self).__init__()
        self.reduction = reduction
        self.faster_rcnn = faster_rcnn

        self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()
        self.offset = nn.L1Loss()

        self.rois_per_image = 64

    def forward(self, nms_reg, nms_cls, feature_maps, bboxes, classes):
        batch_size = bboxes.shape[0]
        num_targets = bboxes.shape[1]
        num_reg = nms_reg.shape[1]

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

        # Take highest IoU-overlap target for each RoI
        iou, indices = torch.max(iou, dim=1)

        mask_positive = (iou > 0.5).view(-1)
        mask_negative = (iou <= 0.5).view(-1)
        sorted_iou, sorted_indices = torch.sort(iou[iou <= 0.5], descending=True)

        reg_loss = iou.new([0])
        cls_loss = iou.new([0])
        accuracy = iou.new([0])
        offset = iou.new([0])

        classes_sampled = torch.cat([
            torch.gather(classes, 1, indices).view(-1)[mask_positive][:self.rois_per_image*batch_size * 1 // 4],
            torch.zeros(self.rois_per_image*batch_size * 3 // 4, dtype=torch.long).to(classes.device)
        ], dim=0)

        img_ids = torch.arange(batch_size, device=nms_reg.device, dtype=torch.float).unsqueeze(1).unsqueeze(2).expand(*nms_reg.shape[:2], 1)
        nms_reg_img_ids = torch.cat([img_ids, nms_reg], dim=2)
        nms_reg_positive = nms_reg_img_ids.view(-1, 5)[mask_positive, :][:self.rois_per_image*batch_size * 1 // 4]
        nms_reg_negative = nms_reg_img_ids.view(-1, 5)[mask_negative, :][sorted_indices][:self.rois_per_image*batch_size * 3 // 4]
        nms_reg_sampled = torch.cat([nms_reg_positive, nms_reg_negative], dim=0)

        nms_reg_positive = nms_reg_positive[:, 1:]

        # 5. Apply Region of Interest-Pooling
        rois = self.faster_rcnn.forward_roi_pooling(feature_maps, nms_reg_sampled)

        # 6. Apply the RCNN for bbox regression and classification
        rcnn_reg, rcnn_cls = self.faster_rcnn.forward_rcnn(rois)

        cls_loss += self.cls_criterion(rcnn_cls, classes_sampled)
      
        if len(mask_positive.nonzero()) > 0:
            accuracy += self.accuracy(rcnn_cls[:len(nms_reg_positive)][:self.rois_per_image*batch_size * 1 // 4], classes_sampled[:len(nms_reg_positive)][:self.rois_per_image*batch_size * 1 // 4])
            
            rcnn_reg = rcnn_reg[:len(nms_reg_positive)][:self.rois_per_image*batch_size * 1 // 4]
            masked_bboxes = torch.gather(bboxes, 1, indices.unsqueeze(2).repeat(1, 1, 4)).view(-1, 4)[mask_positive, :][:self.rois_per_image*batch_size * 1 // 4]
            rounded_masked_bboxes = torch.cat([torch.floor(nms_reg_positive[:, [0, 1]] * self.reduction), torch.ceil(nms_reg_positive[:, [2, 3]] * self.reduction)], dim=1) / self.reduction

            roi_height = torch.abs(rounded_masked_bboxes[:, bottom] - rounded_masked_bboxes[:, top])
            roi_width = torch.abs(rounded_masked_bboxes[:, right] - rounded_masked_bboxes[:, left])

            reg_loss += self.reg_criterion(rcnn_reg[:, top], (masked_bboxes[:, top] - rounded_masked_bboxes[:, top]))
            reg_loss += self.reg_criterion(rcnn_reg[:, left], (masked_bboxes[:, left] - rounded_masked_bboxes[:, left]))
            reg_loss += self.reg_criterion(rcnn_reg[:, bottom], (masked_bboxes[:, bottom] - rounded_masked_bboxes[:, bottom]))
            reg_loss += self.reg_criterion(rcnn_reg[:, right], (masked_bboxes[:, right] - rounded_masked_bboxes[:, right]))

            offset += self.offset(rcnn_reg[:, top], (masked_bboxes[:, top] - rounded_masked_bboxes[:, top]))
            offset += self.offset(rcnn_reg[:, left], (masked_bboxes[:, left] - rounded_masked_bboxes[:, left]))
            offset += self.offset(rcnn_reg[:, bottom], (masked_bboxes[:, bottom] - rounded_masked_bboxes[:, bottom]))
            offset += self.offset(rcnn_reg[:, right], (masked_bboxes[:, right] - rounded_masked_bboxes[:, right]))

        return cls_loss, reg_loss / 4, accuracy, offset / 4


class RPNTrainer(nn.Module):
    def __init__(self):
        super(RPNTrainer, self).__init__()

        self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def match_target_to_anchor(self, anchors, targets, iou_threshold=0.5):
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
            batch_size*num_targets,
            num_anchors,
            4
        )
        targets_expanded = targets_expanded.contiguous().view(-1, 4)

        iou = helper.calculate_iou(anchors_expanded, targets_expanded)
        iou = iou.view(batch_size, -1, num_anchors)

        max_iou, indices = torch.max(iou, dim=1)
        max_target = torch.gather(targets.view(batch_size, num_targets, 4), 1, indices.unsqueeze(2).repeat(1, 1, 4))

        anchors = torch.unsqueeze(anchors, dim=0).expand(batch_size, num_anchors, 4)

        mask = max_iou > iou_threshold

        masked_anchor = anchors[mask].view(-1, 4)
        masked_targets = max_target[mask].view(-1, 4)

        return masked_anchor, masked_targets, mask  

    def forward(self, reg, cls, anchors, targets):
        anchors_matched, targets_matched, mask_matched = self.match_target_to_anchor(anchors, targets, iou_threshold=0.5)

        cls_matched_positive = cls[mask_matched][:128]
        cls_matched_negative = cls[~mask_matched][:256 - len(mask_matched)]

        cls = torch.cat([cls_matched_positive, cls_matched_negative], dim=0)

        anchor_cls = torch.cat([
            torch.ones(cls_matched_positive.shape[0], device=cls_matched_positive.device),
            torch.zeros(cls_matched_negative.shape[0], device=cls_matched_negative.device)
        ])

        anchors_matched = anchors_matched[:128]
        targets_matched = targets_matched[:128]
        reg_matched = reg[mask_matched].view(-1, 4)[:128]

        # Loss
        cls_loss = self.cls_criterion(cls, anchor_cls.float())
        reg_loss = self.reg_criterion(reg_matched.view(-1, 4), (targets_matched.view(-1, 4) - anchors_matched.view(-1, 4)).float())

        return cls_loss, reg_loss / 4