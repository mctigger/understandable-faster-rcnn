import itertools
import numpy as np

import torch
from torch import autograd, nn, optim, utils

import torchvision.transforms as transforms

from tqdm import tqdm
from dataset_test import Dataset, color_map
from helper import visualize_both
from modules.faster_rcnn import FasterRCNN
from modules.predictor import FasterRCNNPredictor
from modules.trainer import FasterRCNNTrainer

top = 0
left = 1
bottom = 2
right = 3

if __name__ == "__main__":
    baseline_boxes = [
        (1, 1),
        (2, 1),
        (1, 2),
        (1, 1.5),
        (1.5, 1)
    ]

    scales = [32, 64, 128, 256]

    anchor_boxes = [(bbox[0] * scale, bbox[1] * scale) for bbox, scale in itertools.product(baseline_boxes, scales)]

    faster_rcnn = FasterRCNN(num_classes=3, anchor_boxes=anchor_boxes)

    dataset = Dataset(transforms.Compose([transforms.ToTensor()]))
    dataloader = utils.data.DataLoader(dataset, batch_size=32, num_workers=12, shuffle=True)

    params = [param for param in faster_rcnn.parameters() if param.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    trainer = nn.DataParallel(FasterRCNNTrainer(
        faster_rcnn
    )).cuda()

    for epoch in range(5):
        losses = []
        rpn_cls_losses = []
        rpn_reg_losses = []
        rcnn_cls_losses = []
        rcnn_reg_losses = []
        accuracy = []

        scheduler.step()
        with tqdm(total=len(dataloader), leave=True, smoothing=1) as pbar:
            pbar.set_description('Epoch {}'.format(epoch))

            for i, (img, bboxes, classes) in enumerate(dataloader):
                img = autograd.Variable(img)
                bboxes = autograd.Variable(bboxes, requires_grad=False)
                classes = autograd.Variable(classes, requires_grad=False)
                img_id = autograd.Variable(torch.arange(0, img.size()[0]))

                rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, acc = trainer(img, img_id, bboxes, classes)
                loss = rpn_cls_loss + rpn_reg_loss * 10 + rcnn_cls_loss + rcnn_reg_loss * 1000
                loss = torch.sum(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.data.cpu().numpy())
                rpn_cls_losses.append(rpn_cls_loss.data.cpu().numpy())
                rpn_reg_losses.append(rpn_reg_loss.data.cpu().numpy())
                rcnn_cls_losses.append(rcnn_cls_loss.data.cpu().numpy())
                rcnn_reg_losses.append(rcnn_reg_loss.data.cpu().numpy())
                accuracy.append(acc.data.cpu().numpy())

                pbar.set_postfix({
                    'loss': np.mean(losses),
                    'rpn_cls_loss': np.mean(rpn_cls_losses),
                    'rpn_reg_loss': np.mean(rpn_reg_losses),
                    'rcnn_cls_loss': np.mean(rcnn_cls_losses),
                    'rcnn_reg_loss': np.mean(rcnn_reg_losses),
                    'accuracy': np.mean(accuracy)
                })

                pbar.update()

    faster_rcnn_predictor = nn.DataParallel(FasterRCNNPredictor(faster_rcnn)).cuda()

    for i, (img, targets, classes) in enumerate(dataloader):
        img = autograd.Variable(img)
        targets = autograd.Variable(targets, requires_grad=False)
        img_id = autograd.Variable(torch.arange(0, img.size()[0]))

        nms_reg, nms_cls, rcnn_reg, rcnn_cls = faster_rcnn_predictor(img, img_id)

        visualize_both(nms_reg, nms_cls, rcnn_reg, rcnn_cls, img, color_map)
