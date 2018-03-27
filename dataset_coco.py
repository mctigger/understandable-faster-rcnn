from pprint import pprint

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor, Resize, Compose
from pycocotools.coco import COCO
from sklearn.preprocessing import LabelEncoder

import helper


class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CocoDataset, self).__init__(root, annFile, transform, target_transform)

        self.transforms = Compose([
            Resize((240, 320)),
            ToTensor()
        ])

        ids = self.coco.getCatIds()
        le = LabelEncoder()
        le.fit(ids)
        self.le = le

    def __getitem__(self, index):
        item = super(CocoDataset, self).__getitem__(index)

        image = item[0]
        width, height = image.size
        x_factor = 320/width
        y_factor = 240/height

        bboxes = []
        classes = []

        for instance in item[1]:
            x, y, width, height = instance['bbox']

            bboxes.append((y*y_factor, x*x_factor, (y+height)*y_factor, (x+width)*x_factor))
            classes.append(instance['category_id'])

        for i in range(100 - len(item[1])):
            while True:
                bboxes.append([0, 0, 0, 0])
                classes.append(1)

                break

        image = self.transforms(image)

        return image, torch.FloatTensor(bboxes), torch.LongTensor(self.le.transform(classes))


if __name__ == "__main__":
    dataset = CocoDataset('./data/coco/train2017', './data/coco/annotations/instances_train2017.json')

    for image, bboxes, classes in dataset:
        image = helper.img_to_np(image)
        for bbox in bboxes:
            image = helper.draw_bbox(image, bbox.long(), [0, 0, 1])

        helper.imshow(image)