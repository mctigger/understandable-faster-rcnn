import torch
import numpy as np
from scipy.misc import imshow
from torch.nn import functional as F

top = 0
left = 1
bottom = 2
right = 3


def calculate_iou(bbox_a, bbox_b):
    yA = torch.max(bbox_a[:, top], bbox_b[:, top])
    xA = torch.max(bbox_a[:, left], bbox_b[:, left])

    yB = torch.min(bbox_a[:, bottom], bbox_b[:, bottom])
    xB = torch.min(bbox_a[:, right], bbox_b[:, right])

    interArea = (xB - xA) * (yB - yA)

    mask_a = (xB - xA) > 0
    mask_b = (yB - yA) > 0

    no_intersect_mask = (mask_a & mask_b) ^ 1

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bbox_a[:, top] - bbox_a[:, bottom]) * (bbox_a[:, left] - bbox_a[:, right])
    boxBArea = (bbox_b[:, top] - bbox_b[:, bottom]) * (bbox_b[:, left] - bbox_b[:, right])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    iou = interArea.float() / (boxAArea + boxBArea - interArea).float()

    iou[no_intersect_mask] = 0

    # return the intersection over union value
    return iou


def img_to_np(img):
    img = img.data.cpu().numpy()
    img = np.copy(img)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)

    return img


def draw_bbox(img, bbox, color, mode='replace'):
    image_size = img.shape[0], img.shape[1]

    top, left, bottom, right = bbox

    top = np.clip(top, 0, image_size[0]-1)
    left = np.clip(left, 0, image_size[1]-1)
    bottom = np.clip(bottom, 0, image_size[0]-1)
    right = np.clip(right, 0, image_size[1]-1)

    if mode == 'replace':
        img[top, left:right] = color
        img[bottom, left:right] = color
        img[top:bottom, left] = color
        img[top:bottom, right] = color

    return img


def visualize_both(nms_reg, nms_cls, rcnn_reg, rcnn_cls, img, color_map):
    nms_cls = F.sigmoid(nms_cls)
    rcnn_cls = F.softmax(rcnn_cls, dim=1)

    for i, (batch_image, batch_reg, batch_cls, batch_rcnn_reg, batch_rcnn_cls) in enumerate(zip(img, nms_reg, nms_cls, rcnn_reg, rcnn_cls)):
        print('batch', i)
        np_img = img_to_np(batch_image)

        for r, c, rcnn_r, rcnn_c in zip(batch_reg, batch_cls, batch_rcnn_reg, batch_rcnn_cls):
            if c >= 0.5:
                cls, index = torch.max(rcnn_c, dim=0)

                cls_color = color_map[0]
                if int(index) in color_map:
                    cls_color = color_map[int(index)]

                np_img = draw_bbox(np_img, torch.round(r).long(), [1, 0, 0])
                np_img = draw_bbox(np_img, torch.round(rcnn_r).long(), np.array(cls_color)/255)

        imshow(np_img)