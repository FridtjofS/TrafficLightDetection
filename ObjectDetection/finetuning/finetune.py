import numpy as np
import json

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

with open('ObjectDetection/finetuning/config.json', 'r') as json_file:
    params = json.load(json_file)

EPOCHS = params['epochs']
BATCH_SIZE = params['batch_size']
WORKERS = params['workers']
MODELS = params['models']
ROOT_DIR = params['data_dir']
CHECKPOINT_DIR = params['checkpoint_dir']
THRESHOLDS = [0.7, 0.75, 0.8]

colors = np.random.uniform(0, 255, size=(len(params['classes']), 3))

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': params['data_dir'],
        'images_dir': params['train_images_dir'],
        'labels_dir': params['train_labels_dir'],
        'classes': params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
)

 
val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': params['data_dir'],
        'images_dir': params['val_images_dir'],
        'labels_dir': params['val_labels_dir'],
        'classes': params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
)

train_data.dataset.plot()
 
for model_to_train in MODELS:
    train_params = {
        'silent_mode': False,
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(params['classes']),
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
                num_cls=len(params['classes']),
                normalize_targets=True,
                calc_best_score_thresholds = True, 
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50:0.95'
    }
    
    trainer = Trainer(
        experiment_name=model_to_train, 
        ckpt_root_dir=CHECKPOINT_DIR
    )

    model = models.get(
        model_to_train, 
        num_classes=len(params['classes']), 
        pretrained_weights="coco"
    )

    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_data, 
        valid_loader=val_data
    )