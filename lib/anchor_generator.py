import itertools

import torch
from torch import nn as nn, autograd as autograd


class AnchorGenerator(nn.Module):
    def __init__(self, boxes, window_size):
        super(AnchorGenerator, self).__init__()
        self.boxes = boxes
        self.window_size = window_size

    def forward(self, img, fm):
        _, _, height_fm, width_fm = fm.size()
        _, _, height_img, width_img = img.size()

        height_reduction = height_img / height_fm
        width_reduction = width_img / width_fm

        anchors = []
        for y, x in itertools.product(range(height_fm - self.window_size+1), range(width_fm - self.window_size+1)):

            anchor_center_v = int(height_reduction * y) + int(height_reduction * self.window_size / 2)
            anchor_center_h = int(width_reduction * x) + int(width_reduction * self.window_size / 2)

            # Generate anchors
            for box in self.boxes:
                anchor_top, anchor_bottom = anchor_center_v - box[0] // 2, anchor_center_v + box[0] // 2
                anchor_left, anchor_right = anchor_center_h - box[1] // 2, anchor_center_h + box[1] // 2

                anchors.append((anchor_top, anchor_left, anchor_bottom, anchor_right))

        return torch.tensor(anchors, requires_grad=False, dtype=torch.float, device=fm.device)