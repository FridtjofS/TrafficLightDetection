
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

with open('ObjectDetection/finetuning/config.json', 'r') as json_file:
    params = json.load(json_file)

BATCH_SIZE = params['batch_size']
WORKERS = params['workers']
ROOT_DIR = params['data_dir']
CHECKPOINT_DIR = params['checkpoint_dir']

def test_yolo_nas(architecture, run):
    classes = ['traffic light']
    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': params['data_dir'],
            'images_dir': params['test_images_dir'],
            'labels_dir': params['test_labels_dir'],
            'classes': params['classes']
        },
        dataloader_params={
            'batch_size':BATCH_SIZE,
            'num_workers':WORKERS
        }
    )
    print(CHECKPOINT_DIR)
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
            save_path = os.path.join(run_dir, 'test_metrics_best.json') if not average else os.path.join(run_dir, 'test_metrics_average.json')
        with open(save_path, 'w') as file:
            json.dump(test_metrics, file, indent=4)
        print(f'saved {prefix} metrics to {save_path}')
    


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
