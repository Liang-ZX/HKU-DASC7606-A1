import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from data.dataset import CAR_CLASSES


def non_maximum_suppression(boxes, scores, threshold=0.5):
    """
    Input:
        - boxes: (bs, 4)  4: [x1, y1, x2, y2] left top and right bottom
        - scores: (bs, )   confidence score
        - threshold: int    delete bounding box with IoU greater than threshold
    Return:
        - A long int tensor whose size is (bs, )
    """
    ###################################################################
    # TODO: Please fill the codes below to calculate the iou of the two boxes
    # Hint: You can refer to the nms part implemented in loss.py but the input shapes are different here
    ##################################################################
    pass

    ##################################################################


def pred2box(args, prediction):
    """
    This function calls non_maximum_suppression to transfer predictions to predicted boxes.
    """
    S, B, C = args.yolo_S, args.yolo_B, args.yolo_C
    
    boxes, cls_indexes, confidences = [], [], []
    prediction = prediction.data.squeeze(0)  # SxSx(B*5+C)
    
    contain = []
    for b in range(B):
        tmp_contain = prediction[:, :, b * 5 + 4].unsqueeze(2)
        contain.append(tmp_contain)

    contain = torch.cat(contain, 2)
    mask1 = contain > 0.1
    mask2 = (contain == contain.max())
    mask = mask1 + mask2
    for i in range(S):
        for j in range(S):
            for b in range(B):
                if mask[i, j, b] == 1:
                    box = prediction[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([prediction[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * 1.0 / S
                    box[:2] = box[:2] * 1.0 / S + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(prediction[i, j, B*5:], 0)
                    cls_index = torch.LongTensor([cls_index])
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexes.append(cls_index)
                        confidences.append(contain_prob * max_prob)

    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        confidences = torch.zeros(1)
        cls_indexes = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)
        confidences = torch.cat(confidences, 0)
        cls_indexes = torch.cat(cls_indexes, 0)
    keep = non_maximum_suppression(boxes, confidences, threshold=args.nms_threshold)
    return boxes[keep], cls_indexes[keep], confidences[keep]


def inference(args, model, img_path):
    """
    Inference the image with trained model to get the predicted bounding boxes
    """
    results = []
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img = cv2.resize(img, (args.image_size, args.image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123.675, 116.280, 103.530)  # RGB
    std = (58.395, 57.120, 57.375)
    ###################################################################
    # TODO: Please fill the codes here to do the image normalization
    ##################################################################
    pass
    ##################################################################

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img).unsqueeze(0)
    img = img.cuda()

    with torch.no_grad():
        prediction = model(img).cpu()  # 1xSxSx(B*5+C)
        boxes, cls_indices, confidences = pred2box(args, prediction)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indices[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        conf = confidences[i]
        conf = float(conf)
        results.append([(x1, y1), (x2, y2), CAR_CLASSES[cls_index], img_path.split('/')[-1], conf])
    return results
