import numpy as np
from skimage.draw import ellipse, polygon

import torch
import torch.utils.data as data


color_map = {
    0: [255, 255, 255],
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [0, 255, 255],
}
num_classes = 3
image_size = (256, 256)


class Dataset(data.Dataset):
    def __init__(self, transform):
        super(Dataset, self).__init__()
        self.transform = transform

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        rs = np.random.RandomState(index)
        #img = rs.random_integers(low=0, high=255, size=(image_size[0], image_size[1], 3))
        img = np.zeros(shape=(image_size[0], image_size[1], 3))
        bboxes = []
        classes = []

        num_targets = 2
        for i in range(num_targets):
            while True:
                height = rs.random_integers(64, 128)
                width = rs.random_integers(64, 128)
                cls = rs.random_integers(1, num_classes)

                border_offset = 0

                top = rs.random_integers(border_offset, image_size[0] - height - border_offset)
                left = rs.random_integers(border_offset, image_size[1] - width - border_offset)
                bottom = top + height
                right = left + width

                rr, cc = ellipse(top + height // 2, left + width//2, height // 2, width // 2)

                img[rr, cc, :] = color_map[cls]
                bboxes.append([int(top), int(left), int(bottom), int(right)])
                classes.append(int(cls))

                break
        for i in range(10 - num_targets):
            while True:
                bboxes.append([0, 0, 0, 0])
                classes.append(0)

                break

        img = self.transform(img)

        bboxes = torch.FloatTensor(bboxes)
        classes = torch.LongTensor(classes)
        return img, bboxes, classes