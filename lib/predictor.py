import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.ops import nms

top = 0
left = 1
bottom = 2
right = 3


class FasterRCNNPredictor(nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNPredictor, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.reduction = self.faster_rcnn.reduction

    def forward(self, img, img_id):
        rpn_reg, rpn_cls, nms_reg, nms_cls, rcnn_reg, rcnn_cls, anchors = self.faster_rcnn(img, img_id)

        nms_reg_rounded = torch.cat([torch.floor(nms_reg[:, :, [0, 1]] * self.reduction), torch.ceil(nms_reg[:, :, [2, 3]] * self.reduction)], dim=2) / self.reduction

        nms_height = torch.abs(nms_reg_rounded[:, :, top] - nms_reg_rounded[:, :, bottom])
        nms_width = torch.abs(nms_reg_rounded[:, :, left] - nms_reg_rounded[:, :, right])

        rcnn_reg[:, :, top] = rcnn_reg[:, :, top]
        rcnn_reg[:, :, left] = rcnn_reg[:, :, left]
        rcnn_reg[:, :, bottom] = rcnn_reg[:, :, bottom]
        rcnn_reg[:, :, right] = rcnn_reg[:, :, right]

        rcnn_reg[:, :, top] = rcnn_reg[:, :, top] + nms_reg_rounded[:, :, top]
        rcnn_reg[:, :, left] = rcnn_reg[:, :, left] + nms_reg_rounded[:, :, left]
        rcnn_reg[:, :, bottom] = rcnn_reg[:, :, bottom] + nms_reg_rounded[:, :, bottom]
        rcnn_reg[:, :, right] = rcnn_reg[:, :, right] + nms_reg_rounded[:, :, right]

        rcnn_cls = F.softmax(rcnn_cls, dim=2)

        reg = []
        cls = []
        batch_size = img.shape[0]
        num_class = rcnn_cls.shape[2]

        for b in range(batch_size):
            b_rcnn_reg = rcnn_reg[b]
            b_rcnn_cls = rcnn_cls[b]
            
            b_rcnn_cls_argmax = torch.argmax(b_rcnn_cls, dim=1)
            
            keep = b_rcnn_cls_argmax != 0

            r = b_rcnn_reg[keep, :]
            c = b_rcnn_cls[keep, :]
            b_rcnn_cls_argmax = b_rcnn_cls_argmax[keep]

            # NMS for RPN for all bboxes with IoU overlap > 0.7
            
            b_reg = []
            b_cls = []
            
            for class_id in range(1, num_class):
                class_c = c[b_rcnn_cls_argmax == class_id, :]
                class_r = r[b_rcnn_cls_argmax == class_id, :]
                
                keep = nms(class_r, class_c[:, class_id], 0.5)  

                if class_c[keep].shape[0] > 0:
                    b_cls.append(class_c[keep])
                    b_reg.append(class_r[keep])

            c = torch.cat(b_cls, dim=0)
            r = torch.cat(b_reg, dim=0)

            reg.append(r)
            cls.append(c)


        return nms_reg, nms_cls, rcnn_reg, rcnn_cls, reg, cls