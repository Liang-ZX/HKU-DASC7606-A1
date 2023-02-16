import os
import json

import random
import numpy as np

import torch
import torch.utils.data as data

import cv2


CAR_CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck',
               'Tram']

COLORS = {'Pedestrian': (0, 0, 0),
          'Cyclist': (128, 0, 0),
          'Car': (0, 128, 0),
          'Truck': (128, 128, 0),
          'Tram': (0, 0, 128)}


class Dataset(data.Dataset):
    image_size = 448

    def __init__(self, args, split, transform):
        print('DATASET INITIALIZATION')
        self.args = args
        root = args.dataset_root
        self.root_images = os.path.join(root, split, 'image')
        if split == "train":
            self.train = True
        else:
            self.train = False

        self.transform = transform
        self.f_names, self.boxes, self.labels = [], [], []
        self.mean = [123.675, 116.280, 103.530]  # RGB
        self.std = [58.395, 57.120, 57.375]
        annotation_path = os.path.join(root, 'annotations', 'instance_' + split + '.json')
        annotations = load_json(annotation_path)

        for annotation in annotations['annotations']:
            if annotation['image_name'] not in self.f_names:
                if len(self.f_names) != 0:
                    self.boxes.append(torch.Tensor(box))
                    self.labels.append(torch.LongTensor(label))
                box, label = [], []
                self.f_names.append(annotation['image_name'])

            bbox = annotation['bbox']
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[0] + bbox[2]), float(bbox[1] + bbox[3])
            box.append([x1, y1, x2, y2])
            label.append(int(annotation['category_id']))

        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        f_name = self.f_names[idx]
        img = cv2.imread(os.path.join(self.root_images, f_name))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            # img = self.random_bright(img)
            img, boxes = random_flip(img, boxes)
            img, boxes = randomScale(img, boxes)
            img = randomBlur(img)
            img = RandomBrightness(img)
            img = RandomHue(img)
            img = RandomSaturation(img)
            img, boxes, labels = randomShift(img, boxes, labels)
            img, boxes, labels = randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = BGR2RGB(img)
        img = subMeanDividedStd(img, self.mean, self.std)
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(boxes, labels)  # S*S*(B*5+C)
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        S, B, C = self.args.yolo_S, self.args.yolo_B, self.args.yolo_C
        grid_num = S
        target = torch.zeros((grid_num, grid_num, B * 5 + C))
        cell_size = 1.0 / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1
            for kk in range(B):
              target[int(ij[1]), int(ij[0]), kk*5 + 4] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + (B-1)*5+4] = 1
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target


def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def BGR2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def HSV2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def RandomBrightness(bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def RandomSaturation(bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def RandomHue(bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def randomBlur(bgr):
    if random.random() < 0.5:
        bgr = cv2.blur(bgr, (5, 5))
    return bgr


def randomShift(bgr, boxes, labels):
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    if random.random() < 0.5:
        height, width, c = bgr.shape
        after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
        after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)

        if shift_x >= 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                 :]
        elif shift_x >= 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                          :]
        elif shift_x < 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                         :]
        elif shift_x < 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                  -int(shift_x):, :]

        shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
        center = center + shift_xy
        mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
        mask = (mask1 & mask2).view(-1, 1)
        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
            boxes_in)
        boxes_in = boxes_in + box_shift
        labels_in = labels[mask.view(-1)]
        return after_shfit_image, boxes_in, labels_in
    return bgr, boxes, labels


def randomScale(bgr, boxes):
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        height, width, c = bgr.shape
        bgr = cv2.resize(bgr, (int(width * scale), height))
        scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        return bgr, boxes
    return bgr, boxes


def randomCrop(bgr, boxes, labels):
    if random.random() < 0.5:
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        height, width, c = bgr.shape
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        x = random.uniform(0, width - w)
        y = random.uniform(0, height - h)
        x, y, h, w = int(x), int(y), int(h), int(w)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
        mask = (mask1 & mask2).view(-1, 1)

        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

        boxes_in = boxes_in - box_shift
        boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
        boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
        boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
        boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

        labels_in = labels[mask.view(-1)]
        img_croped = bgr[y:y + h, x:x + w, :]
        return img_croped, boxes_in, labels_in
    return bgr, boxes, labels


def subMean(bgr, mean):
    mean = np.array(mean, dtype=np.float32)
    bgr = bgr - mean
    return bgr


def subMeanDividedStd(rgb, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb


def random_flip(im, boxes):
    if random.random() < 0.5:
        im_lr = np.fliplr(im).copy()
        h, w, _ = im.shape
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
        return im_lr, boxes
    return im, boxes


def random_bright(im, delta=16):
    alpha = random.random()
    if alpha > 0.3:
        im = im * alpha + random.randrange(-delta, delta)
        im = im.clip(min=0, max=255).astype(np.uint8)
    return im


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = './ass1_dataset'
    train_dataset = Dataset(root=file_root, split='train',
                            transform=[transforms.ToTensor()])
#     img,target = train_dataset[0]
#     exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count() - 2)
    train_iter = iter(train_loader)
    for i in range(10):
        img, target = next(train_iter)
        print(img, target)


if __name__ == '__main__':
    main()  # for debug
