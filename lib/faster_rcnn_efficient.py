import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.ops import RoIPool, batched_nms, nms

from lib.anchor_generator import AnchorGenerator
from lib.cnn_base import CNN
from lib.rcnn_efficient import RCNN
from lib.rpn_efficient import RPN
from lib.sliding_window import SlidingWindow


import timy


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, anchor_boxes, n_proposals):
        super(FasterRCNN, self).__init__()
        fm_channels = 2048 // 2
        window_size = 3
        self.reduction = 1/16
        self.n_proposals = n_proposals

        self.cnn = CNN()
        self.rpn = RPN(fm_channels, len(anchor_boxes), window_size)
        self.rcnn = RCNN(fm_channels, num_classes)
        self.agn = AnchorGenerator(anchor_boxes, window_size)
        self.roi_pooling = RoIPool(output_size=(7, 7), spatial_scale=self.reduction)


    def forward_backbone(self, img):
        """
        Applies a large backbone CNN to the input image. Extracts high-level 
        features from the input.
        
        Args:
            img (Tensor[N, 3, H, W]): A tensor representating a batch of RGB 
                images of height H and width W.
        
        Returns:
            Tensor[N, C, H_out, W_out]: A batch of feature maps that represent 
                high-level features. 
        """
        feature_maps = self.cnn(img)
        return feature_maps

    def forward_rpn(self, feature_maps, anchors):
        """
        Slides the region proposal network over the given feature maps.
        
        Args:
            feature_maps (Tensor[N, C, H, W]): Usually the result from a CNN 
                backbone.
            anchors (Tensor[A, 4]): A list of anchors.
        
        Returns:
            Tuple[Tensor[N, A, 4], Tensor[N, A], Tensor[N, A, 4]]: Bounding box
                regression (offset to the anchor) and classification for each
                anchor and batch. Also returns the bounding regression absolute.
        """
        batch_size, num_channels = feature_maps.shape[:2]
        
        # Apply RPN for proposal bbox regression and binary classification (roi/non-roi).
        rpn_reg, rpn_cls = self.rpn(feature_maps)

        # Unpack slices from batch dimension back into second dimension 
        # resulting in a Tensor[N, A, 4] for regression and Tensor[N, A] for
        # classification results.
        rpn_reg = rpn_reg.view(batch_size, -1, 4)
        rpn_cls = rpn_cls.view(batch_size, -1)

        anchors = torch.unsqueeze(anchors, dim=0).expand_as(rpn_reg)
        rpn_reg_absolute = (rpn_reg + anchors)

        return rpn_reg, rpn_cls, rpn_reg_absolute

    def forward_nms(self, rpn_reg_absolute, rpn_cls):
        """
        Applies non-maximum suppression to the region proposals. Also pads 
        results if less than self.n_proposals survive non-maximum suppression.
        
        Args:
            rpn_reg_absolute (Tensor[N, A, 4]): Region proposals relative to 
                image coordinate system.
            rpn_cls (Tensor[N, A]): Region proposal classification prediction.
        
        Returns:
            Tuple[Tensor[B*N_proposals, 4], Tensor[B*N_proposals]]: Post-nms 
                regression and classification scores.
        """
        nms_reg = []
        nms_cls = []

        for b in range(rpn_reg_absolute.shape[0]):
            b_rpn_reg_absolute = rpn_reg_absolute[b]
            b_rpn_cls = rpn_cls[b]
            # NMS for RPN for all bboxes with IoU overlap > 0.7
            keep = nms(b_rpn_reg_absolute[:, [1, 0, 3, 2]], torch.sigmoid(b_rpn_cls), 0.7)

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

        return nms_reg, nms_cls

    def forward_roi_pooling(self, feature_maps, nms_reg):
        """
        Applies region of interest pooling. This method assumes that the roi 
        coordinates are not given batch-wise but merged in a single dimension.
        This is done for efficiency since we do not need to compute rois for 
        padding coordinates.
        
        Args:
            feature_maps (Tensor[N, C, H, W]): Feature maps from the CNN backbone.
            nms_reg (Tensor[P?, 5]): The region of interests coordinates. The first column 
                in the second dimension represents the batch index that the roi belongs to. 
        
        Returns:
            Tensor[P?, C, O, O]: A tensor of regions of interests.
        """

        # Reorder second dimension. We assume [y1, x1, y2, x2] format for 
        # bounding box coordinates, but torchvision roi_pooling uses [x1, y1, x2, y2].
        nms_reg_rounded = nms_reg[:, [0, 2, 1, 4, 3]]

        # Roi-pooling only works with full pixels, so we have to round the 
        # coordinates. We floor for the top-left corner and ceil for the 
        # bottom-right corner to include more context into a roi. Else we would 
        # lose some information at the boundary of objects.
        nms_reg_rounded = torch.cat([
            nms_reg_rounded[:, [0]],
            torch.floor(nms_reg_rounded[:, [1, 2]] * self.reduction) / self.reduction,
            torch.ceil(nms_reg_rounded[:, [3, 4]] * self.reduction) / self.reduction
        ], dim=1)

        # Apply roi-pooling.
        rois = self.roi_pooling(feature_maps, nms_reg_rounded)

        return rois

    def forward_rcnn(self, rois):
        """
        Runs each roi through the RCNN.
        
        Args:
            rois (Tensor[P?, C, O, O]): A tensor of P? RoIs.
        
        Returns:
            [Tuple[Tensor[P?, 4], Tensor[P?, N_cl]]]: The first tensor 
                represents the bbox offset from the RoI bbox to the real object
                bbox. The second tensor classifies the bbox into one of N_cl 
                classes with the first entry representing the background class.
        """
        rcnn_reg, rcnn_cls = self.rcnn(rois)

        return rcnn_reg, rcnn_cls

    def forward(self, img):
        
        # 1. Apply CNN base for feature extraction
        feature_maps = self.forward_backbone(img)
        batch_size, fm_channels, _, _ = feature_maps.size()

        # 2. + 3. together equal to sliding the RPN over the feature map
        # 2. Apply sliding window to get slices
        anchors = self.agn(img, feature_maps)
        rpn_reg, rpn_cls, rpn_reg_absolute = self.forward_rpn(feature_maps, anchors)

        # 4. Apply Non-Maximum-Suppression
        nms_reg, nms_cls = self.forward_nms(rpn_reg_absolute, rpn_cls)
        
        # 5. Apply Region of Interest-Pooling
        img_ids = torch.arange(batch_size, device=nms_reg.device, dtype=torch.float).unsqueeze(1).unsqueeze(2).expand(*nms_reg.shape[:2], 1)
        nms_reg_img_ids = torch.cat([img_ids, nms_reg], dim=2)
        rois = self.forward_roi_pooling(feature_maps, nms_reg_img_ids.view(-1, 5))

        # 6. Apply the RCNN for bbox regression and classification
        rcnn_reg, rcnn_cls = self.forward_rcnn(rois)

        return rpn_reg, rpn_cls, nms_reg, nms_cls, rcnn_reg, rcnn_cls, anchors




