from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import pickle

from data.dataset import CAR_CLASSES, COLORS, load_json
from model.hkudetector import resnet50
from utils.util import *


class Evaluation:
    def __init__(self, predictions, targets, threshold):
        super(Evaluation, self).__init__()
        self.predictions = predictions
        self.targets = targets
        self.threshold = threshold

    @staticmethod
    def compute_ap(recall, precision):
        # average precision calculation
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))

        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        ap = 0.0  # average precision (AUC of the precision-recall curve).
        for i in range(precision.size - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        return ap

    def evaluate(self):
        aps = []
        print('CLASS'.ljust(25, ' '), 'AP')
        for class_name in CAR_CLASSES:
            class_preds = self.predictions[class_name]  # [[image_id,confidence,x1,y1,x2,y2],...]
            if len(class_preds) == 0:
                ap = 0
                print(f'{class_name}'.ljust(25, ' '), f'{ap:.2f}')
                aps.append(ap)
                continue
            image_ids = [x[0] for x in class_preds]
            confidence = np.array([float(x[1]) for x in class_preds])
            BB = np.array([x[2:] for x in class_preds])
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            npos = 0.
            for (key1, key2) in self.targets:
                if key2 == class_name:
                    npos += len(self.targets[(key1, key2)])
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            for d, image_id in enumerate(image_ids):
                bb = BB[d]
                if (image_id, class_name) in self.targets:
                    BBGT = self.targets[(image_id, class_name)]
                    for x1y1_x2y2 in BBGT:
                        # compute overlaps
                        # intersection
                        x_min = np.maximum(x1y1_x2y2[0], bb[0])
                        y_min = np.maximum(x1y1_x2y2[1], bb[1])
                        x_max = np.minimum(x1y1_x2y2[2], bb[2])
                        y_max = np.minimum(x1y1_x2y2[3], bb[3])
                        w = np.maximum(x_max - x_min + 1., 0.)
                        h = np.maximum(y_max - y_min + 1., 0.)
                        intersection = w * h

                        union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (x1y1_x2y2[2] - x1y1_x2y2[0] + 1.) * (
                                x1y1_x2y2[3] - x1y1_x2y2[1] + 1.) - intersection
                        if union == 0:
                            print(bb, x1y1_x2y2)

                        overlaps = intersection / union
                        if overlaps > self.threshold:
                            tp[d] = 1
                            BBGT.remove(x1y1_x2y2)
                            if len(BBGT) == 0:
                                del self.targets[(image_id, class_name)]
                            break
                    fp[d] = 1 - tp[d]
                else:
                    fp[d] = 1
            ###################################################################
            # TODO: Please fill the codes to compute recall and precision
            ##################################################################
            recall = 0.
            precision = 0.

            ##################################################################
            ap = self.compute_ap(recall, precision)

            print(f'{class_name}'.ljust(25, ' '), f'{ap*100:.2f}')
            aps.append(ap)

        return aps


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')
    parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num')
    parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

    parser.add_argument('--dataset_root', default='./ass1_dataset', type=str, help='dataset root')
    parser.add_argument('--split', default='val', type=str, help="dataset split in ['val', 'test']")
    parser.add_argument('--model_path', default="./checkpoints/hku_mmdetector_best.pth", help='Pretrained Model Path')
    parser.add_argument('--output_file', default="./result.pkl", help='PKL for evaluation')
    parser.add_argument('--pos_threshold', default=0.3, type=float, help='Confidence threshold for positive prediction')
    parser.add_argument('--nms_threshold', default=0.5, type=float, help='Threshold for non maximum suppression')
    args = parser.parse_args()

    targets = defaultdict(list)
    predictions = defaultdict(list)
    image_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('DATA PREPARING...')
    annotation_path = os.path.join(args.dataset_root, 'annotations', 'instance_%s.json' % args.split)
    annotations = load_json(annotation_path)

    for annotation in annotations['annotations']:
        image_name = annotation['image_name']
        if image_name not in image_list:
            image_list.append(image_name)
        bbox = annotation['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        c = int(annotation['category_id'])
        class_name = CAR_CLASSES[c-1]
        targets[(image_name, class_name)].append([x1, y1, x2, y2])
    print('DONE.')
    print('START EVALUATION...')

    model = resnet50(args=args).to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.eval()

    for image_name in tqdm(image_list):
        image_path = os.path.join(args.dataset_root, args.split, 'image', image_name)
        result = inference(args, model, image_path)

        for (x1, y1), (x2, y2), class_name, image_name, conf in result:
            predictions[class_name].append([image_name, conf, x1, y1, x2, y2])

    # write the prediction result
    f = open(args.output_file, 'wb')
    pickle.dump(args, f)
    pickle.dump(predictions, f)
    f.close()

    print('BEGIN CALCULATE MAP...')
    aps = Evaluation(predictions, targets, threshold=args.pos_threshold).evaluate()
    print(f'mAP: {np.mean(aps):.2f}')
    print('DONE.')
