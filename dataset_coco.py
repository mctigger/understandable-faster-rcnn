from pprint import pprint

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor, Resize, Compose
from pycocotools.coco import COCO
from sklearn.preprocessing import LabelEncoder

import helper


HEIGHT = 600
WIDTH = 800

class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CocoDataset, self).__init__(root, annFile, transform, target_transform)

        self.transforms_ = Compose([
            Resize((HEIGHT, WIDTH)),
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
        x_factor = WIDTH/width
        y_factor = HEIGHT/height

        bboxes = []
        classes = []

        for instance in item[1]:
            x, y, width, height = instance['bbox']

            bboxes.append(torch.FloatTensor([y*y_factor, x*x_factor, (y+height)*y_factor, (x+width)*x_factor]))
            classes.append(torch.LongTensor(self.le.transform([instance['category_id']])))

        image = self.transforms_(image)

        return image, bboxes, classes


def detection_collate(batch):
    batch_img = []
    batch_bboxes = []
    batch_classes = []

    max_num_object = max([len(bboxes) for img, bboxes, classes in batch])

    for img, bboxes, classes in batch:
        batch_img.append(img)
        batch_bboxes.append(torch.stack(bboxes + [torch.FloatTensor([0, 0, 0, 0])]*(max_num_object - len(bboxes)), dim=0))
        batch_classes.append(torch.cat(classes + [torch.LongTensor([0])]*(max_num_object - len(bboxes)), dim=0))
    
    batch_img = torch.stack(batch_img, 0)
    batch_bboxes = torch.stack(batch_bboxes, 0)
    batch_classes = torch.stack(batch_classes, 0)

    return batch_img, batch_bboxes, batch_classes



if __name__ == "__main__":
    dataset = CocoDataset('./data/coco/train2017', './data/coco/annotations/instances_train2017.json')

    for image, bboxes, classes in dataset:
        image = helper.img_to_np(image)
        for bbox in bboxes:
            image = helper.draw_bbox(image, bbox.long(), [0, 0, 1])

        helper.imshow(image)