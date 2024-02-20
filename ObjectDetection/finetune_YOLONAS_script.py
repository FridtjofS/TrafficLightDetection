import argparse
from super_gradients.training import Trainer
#from super_gradients.training import dataloaders
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
from tqdm.auto import tqdm
 
import os
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import random

# Global parameters.
EPOCHS = 5
BATCH_SIZE = 2
WORKERS = 1
CLASSES = ['traffic light']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
MODELS = [
        'yolo_nas_s'#,
        #'yolo_nas_m',
        #'yolo_nas_l'
]
    
# CHECKPOINT_DIR = 'checkpoints'


def train_model(dataset_folder, save_weights):
    train_imgs_dir = 'images/train'
    train_labels_dir = 'labels/train'
    val_imgs_dir = 'images/val'
    val_labels_dir = 'labels/val'
    test_imgs_dir = 'images/test'
    test_labels_dir = 'labels/test'


    dataset_params = {
        'data_dir':dataset_folder,
        'train_images_dir':train_imgs_dir,
        'train_labels_dir':train_labels_dir,
        'val_images_dir':val_imgs_dir,
        'val_labels_dir':val_labels_dir,
        'test_images_dir':test_imgs_dir,
        'test_labels_dir':test_labels_dir,
        'classes':CLASSES 
    }

    # Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
    def yolo2bbox(bboxes):
        xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
        xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
        return xmin, ymin, xmax, ymax

    def plot_box(image, bboxes, labels):
        # Need the image height and width to denormalize bounding box coordinates
        height, width, _ = image.shape
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
        tf = max(lw - 1, 1) # Font thickness.
        for box_num, box in enumerate(bboxes):
            x1, y1, x2, y2 = yolo2bbox(box)
            # denormalize the coordinates
            xmin = int(x1*width)
            ymin = int(y1*height)
            xmax = int(x2*width)
            ymax = int(y2*height)

            p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
            
            class_name = CLASSES[int(labels[box_num])]

            color=COLORS[CLASSES.index(class_name)]
            
            cv2.rectangle(
                image, 
                p1, p2,
                color=color, 
                thickness=lw,
                lineType=cv2.LINE_AA
            ) 

            # For filled rectangle.
            w, h = cv2.getTextSize(
                class_name, 
                0, 
                fontScale=lw / 3, 
                thickness=tf
            )[0]

            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(
                image, 
                p1, p2, 
                color=color, 
                thickness=-1, 
                lineType=cv2.LINE_AA
            )  
            cv2.putText(
                image, 
                class_name, 
                (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw/3.5, 
                color=(255, 255, 255), 
                thickness=tf, 
                lineType=cv2.LINE_AA
            )
        return image

    # Function to plot images with the bounding boxes.
    def plot(image_path, label_path, num_samples, dataset_folder):
        all_training_images = glob.glob(image_path+'/*')
        all_training_labels = glob.glob(label_path+'/*')
        all_training_images.sort()
        all_training_labels.sort()
        
        temp = list(zip(all_training_images, all_training_labels))
        random.shuffle(temp)
        all_training_images, all_training_labels = zip(*temp)
        all_training_images, all_training_labels = list(all_training_images), list(all_training_labels)
        
        num_images = len(all_training_images)
        
        if num_samples == -1:
            num_samples = num_images
            
        plt.figure(figsize=(15, 12))
        for i in range(num_samples):
            image_name = all_training_images[i].split(os.path.sep)[-1]
            image = cv2.imread(all_training_images[i])
            with open(all_training_labels[i], 'r') as f:
                bboxes = []
                labels = []
                label_lines = f.readlines()
                for label_line in label_lines:
                    label, x_c, y_c, w, h = label_line.split(' ')
                    x_c = float(x_c)
                    y_c = float(y_c)
                    w = float(w)
                    h = float(h)
                    bboxes.append([x_c, y_c, w, h])
                    labels.append(label)
            result_image = plot_box(image, bboxes, labels)
            plt.subplot(2, 2, i+1) # Visualize 2x2 grid of images.
            plt.imshow(image[:, :, ::-1])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Visualize a few training images.
    plot(
        image_path=os.path.join(dataset_folder, train_imgs_dir), 
        label_path=os.path.join(dataset_folder, train_labels_dir),
        num_samples=4,
        dataset_folder = dataset_folder
    )

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':BATCH_SIZE,
            'num_workers':WORKERS
        }
    )
    
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':BATCH_SIZE,
            'num_workers':WORKERS
        }
    )

    train_data.dataset.transforms

    train_data.dataset.transforms[0]

    train_data.dataset.transforms.pop(2)

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
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
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
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
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


    #Model Training

    import torchvision.transforms as transforms
    import torch
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model's input size
        transforms.ToTensor(),# Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    for model_to_train in MODELS:
        trainer = Trainer(
            experiment_name=model_to_train, 
            ckpt_root_dir=save_weights
        )

        model = models.get(
            model_to_train, 
            num_classes=len(dataset_params['classes']), 
            pretrained_weights="coco"
        )

        trainer.train(
            model=model, 
            training_params=train_params, 
            train_loader=train_data, 
            valid_loader=val_data
        )

    model.eval()

    import torch
    from tqdm import tqdm

    def evaluate_detection(model, dataloader, save_dir=None, num_samples=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        predictions = []

        with torch.no_grad():
            for images, targets in tqdm(dataloader):
                images = images.to(device)

                outputs = model(images)

                for output, target in zip(outputs, targets):
                    if isinstance(output, tuple):
                        output = output[0]  # Assuming the first element of the tuple is the predictions

                    pred_boxes = output[:, :4]  # Extract bounding boxes
                    pred_scores = output[:, 4]   # Extract confidence scores

                    # Convert predictions to numpy arrays
                    pred_boxes = pred_boxes.cpu().numpy()
                    pred_scores = pred_scores.cpu().numpy()

                    # Combine boxes and scores
                    pred_output = {"boxes": pred_boxes, "scores": pred_scores}
                    predictions.append(pred_output)

                    # Save sample output
                    if save_dir and len(predictions) <= num_samples:
                        save_prediction_image(images, pred_output, target, save_dir)

        return predictions

    import cv2

    def save_prediction_image(images, pred_output, target, save_dir):
        # Print the keys of the pred_output dictionary
        print("Keys of pred_output dictionary:", pred_output.keys())

        # Convert image tensor to numpy array
        images = images.cpu().numpy()

        # Iterate over each image in the batch
        for i, image in enumerate(images):
            # Check if the pred_output contains the expected keys
            if i in pred_output:
                pred_boxes = pred_output[i]["boxes"]
                pred_scores = pred_output[i]["scores"]
                pred_classes = pred_output[i]["labels"]

                # Iterate over each predicted box
                for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
                    xmin, ymin, xmax, ymax = box.astype(int)

                    # Draw bounding box on the image
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Add label and score to the bounding box
                    label = f"{CLASSES[cls]}: {score:.2f}"
                    cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the image
                image_name = os.path.join(save_dir, f"prediction_{i + 1}.jpg")
                cv2.imwrite(image_name, image)
            else:
                print(f"Prediction output not found for image {i}")

    # Usage example:
    predictions = evaluate_detection(model, val_data, save_dir="/home/uni/sample_output", num_samples=5)

    # Print some sample predictions for analysis
    for i, pred_output in enumerate(predictions[:5]):
        print(f"Sample Prediction {i + 1}:")
        print(pred_output)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model for object detection.")
    parser.add_argument("--dataset_folder", required=True, help="Folder where the dataset is located.")
    parser.add_argument("--save_weights", action="store_true", help="Flag to save weights.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    train_model(args.dataset_folder, args.save_weights)


