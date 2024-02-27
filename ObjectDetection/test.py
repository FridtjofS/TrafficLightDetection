
import argparse
import os
import json

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

#did not import variables because this would trigger training run
BATCH_SIZE = 2
WORKERS = 1
#ROOT_DIR = '/home/jakob/uni/TrafficLightDetection/ObjectDetection/od_train_data/dataset'
ROOT_DIR = '/home/jakob/uni/TrafficLightDetection/ObjectDetection/od_train_data/backup/dataset_60_20_20'
CHECKPOINT_DIR = 'ObjectDetection/checkpoints'

def test_yolo_nas(architecture, run):
    classes = ['traffic light']
    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': ROOT_DIR,
            'images_dir': 'images/test',
            'labels_dir': 'labels/test',
            'classes': classes
        },
        dataloader_params={
            'batch_size':BATCH_SIZE,
            'num_workers':WORKERS
        }
    )
    trainer =  Trainer(
            experiment_name=architecture, 
            ckpt_root_dir=CHECKPOINT_DIR
        )
    run_dir = os.path.join(CHECKPOINT_DIR, architecture, run)
    for average in [True, False]:
        checkpoint_path = os.path.join(run_dir, 'average_model.pth') if average else os.path.join(run_dir, 'ckpt_best.pth') 
        model_best = models.get(architecture, num_classes=len(classes), checkpoint_path=checkpoint_path)

        test_metrics = trainer.test(
            model=model_best,
            test_loader=test_data,
            test_metrics_list=[
                DetectionMetrics_050(
                score_thres=0.1, 
                top_k_predictions=300, 
                num_cls=len(classes), 
                normalize_targets=True, 
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01, 
                    nms_top_k=1000, 
                    max_predictions=300,                                                                              
                    nms_threshold=0.7
                )
                ),
                DetectionMetrics_050_095(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=len(classes),
                    normalize_targets=True,
                    calc_best_score_thresholds = True, 
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7
                    )
                )
            ]
        )
        
        for key in test_metrics.keys():
            prefix = 'avg' if average else 'best'
            print(f'{prefix} {key}: {test_metrics[key]}')
            save_dir = os.path.join(run_dir, 'test_metrics_best.json') if not average else os.path.join(run_dir, 'test_metrics_average.json')
        with open(save_dir, 'w') as file:
            json.dump(test_metrics, file, indent=4)
    


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test yolo nas model dinetuned earlier")
    parser.add_argument("--architecture", required=True, help="model architecture type (folder name 'yolo_nas__')")
    parser.add_argument("--run", required=True, help="Foldername of the training run to test (folder name 'RUN...').")
    return parser.parse_args()

def main():
    args = parse_arguments()
    test_yolo_nas(args.architecture, args.run)
    print("Test completed.")

if __name__ == "__main__":
    main()
