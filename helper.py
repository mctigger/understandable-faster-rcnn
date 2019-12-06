import time
import functools
import collections

import torch
import numpy as np
from matplotlib import patches, text
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
    img = img.cpu().numpy()
    img = np.copy(img)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)

    return img

    
def add_bbox(ax, bbox, color, alpha=1, text=""):
    top, left, bottom, right = bbox
    rect = patches.Rectangle((left,top), right-left, bottom-top, linewidth=2, edgecolor=color, facecolor='none', alpha=alpha)
    ax.add_patch(rect)
    ax.text(
        left,
        top,
        text,
        fontsize=12,
        bbox=dict(
            boxstyle="square",
            fc=color,
        )
    )
    
def visualize_anchors(ax, anchors):
    for a in anchors:
        add_bbox(ax, torch.round(a).long(), [1.0, 1.0, 1.0], 0.2)
        
def visualize_rpn(ax, nms_reg, nms_cls, img, color=[1.0, 1.0, 1.0], draw_all=False):
    np_img = img_to_np(img)

    for r, c in zip(nms_reg, nms_cls):
        if c >= 0.5 or draw_all:
            add_bbox(ax, r, color=color, text="c={:.2f}".format(c))

            
def visualize_rcnn(ax, rcnn_reg, rcnn_cls, color_map):
    for rcnn_r, rcnn_c in zip(rcnn_reg, rcnn_cls):
        cls, index = torch.max(rcnn_c, dim=0)
        
        if index == 0:
            continue
        
        cls_color = color_map[0]
        if int(index) in color_map:
            cls_color = color_map[int(index)]

        add_bbox(ax, rcnn_r, color=cls_color, text="class={}: {:.2f}".format(index, cls))


class GPURuntimeProfiler:
    def __init__(self):
        self.stats = collections.defaultdict(int)

    def measure_gpu(self, name):
        def decorator(f):
            @functools.wraps(f)
            def measure(*args, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()

                result = f(*args, **kwargs)

                torch.cuda.synchronize() # wait for mm to finish
                end = time.perf_counter()

                duration = end - start

                self.stats[name] += duration

                return result

            return measure
        
        return decorator
