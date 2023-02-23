from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import pickle

from model.hkudetector import resnet50
from utils.util import *

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')
    parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num')
    parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

    parser.add_argument('--dataset_root', default='./ass1_dataset', type=str, help='dataset root')
    parser.add_argument('--split', default='test', type=str, help="dataset split in ['val', 'test']")
    parser.add_argument('--model_path', default="./checkpoints/hku_mmdetector_best.pth", help='Pretrained Model Path')
    parser.add_argument('--output_file', default="./result.pkl", help='PKL for evaluation')
    parser.add_argument('--pos_threshold', default=0.1, type=float, help='Confidence threshold for positive prediction')
    parser.add_argument('--nms_threshold', default=0.5, type=float, help='Threshold for non maximum suppression')
    parser.add_argument('--image_size', default=448, type=int, help='Image Size')
    args = parser.parse_args()

    targets = defaultdict(list)
    predictions = defaultdict(list)
    image_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('START EVALUATION...')

    model = resnet50(args=args).to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.eval()

    test_root = os.path.join(args.dataset_root, args.split, 'image')
    for image_name in tqdm(os.listdir(test_root)):
        image_path = os.path.join(test_root, image_name)
        result = inference(args, model, image_path)

        for (x1, y1), (x2, y2), class_name, image_name, conf in result:
            predictions[class_name].append([image_name, conf, x1, y1, x2, y2])

    # write the prediction result
    f = open(args.output_file, 'wb')
    pickle.dump(args, f)
    pickle.dump(predictions, f)
    f.close()

    print('DONE.')
