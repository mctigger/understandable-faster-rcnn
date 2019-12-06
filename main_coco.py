import itertools
import numpy as np

import torch
from torch import autograd, nn, optim, utils

import torchvision.transforms as transforms

from tqdm import tqdm
from dataset_coco import CocoDataset, detection_collate
from lib.faster_rcnn_efficient import FasterRCNN
from lib.predictor import FasterRCNNPredictor
from lib.trainer_efficient import FasterRCNNTrainer

top = 0
left = 1
bottom = 2
right = 3

def train():
    baseline_boxes = [
        (1, 1),
        (2, 1),
        (1, 2)
    ]

    scales = [64, 128, 256, 512]

    anchor_boxes = [(bbox[0] * scale, bbox[1] * scale) for bbox, scale in itertools.product(baseline_boxes, scales)]

    faster_rcnn = FasterRCNN(num_classes=81, anchor_boxes=anchor_boxes, n_proposals=2000)

    dataset = CocoDataset("data/coco/train2017", "data/coco/annotations/instances_train2017.json")
    dataloader = utils.data.DataLoader(dataset, batch_size=16, num_workers=12, shuffle=True, collate_fn=detection_collate)

    params = [param for param in faster_rcnn.parameters() if param.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = nn.DataParallel(FasterRCNNTrainer(
        faster_rcnn
    )).cuda()

    for epoch in range(1):
        losses = []
        rpn_cls_losses = []
        rpn_reg_losses = []
        rcnn_cls_losses = []
        rcnn_reg_losses = []
        accuracy = []
        offsets = []

        with tqdm(total=len(dataloader), leave=True, smoothing=1) as pbar:
            pbar.set_description('Epoch {}'.format(epoch))

            for i, (img, bboxes, classes) in enumerate(dataloader):
                img = img.float()
                bboxes = bboxes.detach()
                classes = classes.detach()

                rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, acc, offset = trainer(img, bboxes, classes)
                loss = rpn_cls_loss + rpn_reg_loss * 10 + rcnn_cls_loss + rcnn_reg_loss * 10
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.data.cpu().numpy())
                rpn_cls_losses.append(rpn_cls_loss.data.cpu().numpy())
                rpn_reg_losses.append(rpn_reg_loss.data.cpu().numpy())
                rcnn_cls_losses.append(rcnn_cls_loss.data.cpu().numpy())
                rcnn_reg_losses.append(rcnn_reg_loss.data.cpu().numpy())
                accuracy.append(acc.data.cpu().numpy())
                offsets.append(offset.data.cpu().numpy())

                pbar.set_postfix({
                    'loss': np.mean(losses),
                    'rpn_cls_loss': np.mean(rpn_cls_losses),
                    'rpn_reg_loss': np.mean(rpn_reg_losses),
                    'rcnn_cls_loss': np.mean(rcnn_cls_losses),
                    'rcnn_reg_loss': np.mean(rcnn_reg_losses),
                    'accuracy': np.mean(accuracy),
                    'offsets': np.mean(offsets)
                })

                pbar.update()

        scheduler.step()
        
    return faster_rcnn, dataset


if __name__ == "__main__":
    train()