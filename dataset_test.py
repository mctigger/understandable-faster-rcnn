import numpy as np
from skimage.draw import ellipse, polygon

import torch
import torch.utils.data as data


color_map = {
    0: [1, 1, 1],
    1: [0, 0, 1],
    2: [0, 1, 0],
    3: [1, 0, 0],
    4: [0, 1, 1],
}
num_classes = 3
image_size = (256, 256)


class Dataset(data.Dataset):
    def __init__(self, transform, min_bboxes=0, max_bboxes=3):
        super(Dataset, self).__init__()
        self.transform = transform
        self.min_bboxes = min_bboxes
        self.max_bboxes = max_bboxes

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        rs = np.random.RandomState(index)
        img = np.zeros(shape=(image_size[0], image_size[1], 3), dtype=np.float32)
        bboxes = []
        classes = []

        num_targets = np.random.randint(self.min_bboxes, self.max_bboxes)
        for i in range(num_targets):
            height = rs.randint(64, 128)
            width = rs.randint(64, 128)
            cls = rs.randint(1, num_classes + 1)

            border_offset = 0

            top = rs.random_integers(border_offset, image_size[0] - height - border_offset)
            left = rs.random_integers(border_offset, image_size[1] - width - border_offset)
            bottom = top + height
            right = left + width

            rr, cc = ellipse(top + height // 2, left + width//2, height // 2, width // 2)

            img[rr, cc, :] = color_map[cls]
            bboxes.append([int(top), int(left), int(bottom), int(right)])
            classes.append(int(cls))

            
        for i in range(10 - num_targets):
            while True:
                bboxes.append([0, 0, 0, 0])
                classes.append(0)

                break

        img = self.transform(img)

        bboxes = torch.FloatTensor(bboxes)
        classes = torch.LongTensor(classes)
        return img, bboxes, classes