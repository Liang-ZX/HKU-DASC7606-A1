import cv2
import argparse
import os
import torch

from data.dataset import COLORS, CAR_CLASSES
from model.hkudetector import resnet50
from utils.util import inference


def predict(args, model):
    image_path = args.image_path
    image = cv2.imread(image_path)

    print('PREDICTING...')
    result = inference(args, model, image_path)

    for x1y1, x2y2, class_name, _, prob in result:
        color = COLORS[class_name]
        cv2.rectangle(image, x1y1, x2y2, color, 2)

        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        p1 = (x1y1[0], x1y1[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    if not args.unsave_img:
        vis_dir = args.vis_dir
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        save_path = os.path.join(vis_dir, image_path.split('/')[-1])
        cv2.imwrite(save_path, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')
    parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num')
    parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

    parser.add_argument('--image_path', default="./ass1_dataset/val/image/000001.jpg", help='Path to Image file')
    parser.add_argument('--model_path', default="./checkpoints/hku_mmdetector_best.pth", help='Pretrained Model Path')
    parser.add_argument('--unsave_img', action='store_true', help='Do not save the image after detection')
    parser.add_argument('--vis_dir', default="./vis_results", help='Dir for Visualization')

    parser.add_argument('--nms_threshold', default=0.5, type=float, help='Threshold for non maximum suppression')
    args = parser.parse_args()

    ####################################################################
    # Prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(args=args).to(device)

    print('LOADING MODEL...')
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)

    # If you have single gpu then please modify model loading process
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.eval()
    predict(args, model)

    # # If you want to predict multiple images at one time, please uncomment the following codes
    # image_root = "./ass1_dataset/val/image/"
    # for i in range(10):
    #     image_path = os.path.join(image_root, "%06d.jpg" % i)
    #     args.image_path = image_path
    #     predict(args, model)
