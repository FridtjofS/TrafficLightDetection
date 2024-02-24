import os
import shutil
import argparse
import random

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
RANDOM_SEED = 42

def split_dataset(dataset_folder, output_folder):
    # Create output folders if they don't exist
    for folder in ['labels', 'images']:
        os.makedirs(os.path.join(output_folder, folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, folder, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, folder, 'test'), exist_ok=True)

    # Initialize lists to store paths
    annotation_files = [f for f in os.listdir(os.path.join(dataset_folder, 'annotations')) if f.endswith('.txt')]
    random.seed(RANDOM_SEED)
    random.shuffle(annotation_files)

    total_annotations = len(annotation_files)
    train_size = int(total_annotations * TRAIN_SPLIT)
    val_size = int(total_annotations * VAL_SPLIT)
    test_size = total_annotations - train_size - val_size

    train_annotations = annotation_files[:train_size]
    val_annotations = annotation_files[train_size:train_size + val_size]
    test_annotations = annotation_files[train_size + val_size:]

    # Copy annotation files
    for annotation_file in train_annotations:
        shutil.copy(os.path.join(dataset_folder, 'annotations', annotation_file), os.path.join(output_folder, 'labels', 'train'))
    for annotation_file in val_annotations:
        shutil.copy(os.path.join(dataset_folder, 'annotations', annotation_file), os.path.join(output_folder, 'labels', 'val'))
    for annotation_file in test_annotations:
        shutil.copy(os.path.join(dataset_folder, 'annotations', annotation_file), os.path.join(output_folder, 'labels', 'test'))

    # Copy image files
    for annotation_file in train_annotations:
        image_file = os.path.splitext(annotation_file)[0] + '.jpg'
        shutil.copy(os.path.join(dataset_folder, 'images', image_file), os.path.join(output_folder, 'images', 'train'))
    for annotation_file in val_annotations:
        image_file = os.path.splitext(annotation_file)[0] + '.jpg'
        shutil.copy(os.path.join(dataset_folder, 'images', image_file), os.path.join(output_folder, 'images', 'val'))
    for annotation_file in test_annotations:
        image_file = os.path.splitext(annotation_file)[0] + '.jpg'
        shutil.copy(os.path.join(dataset_folder, 'images', image_file), os.path.join(output_folder, 'images', 'test'))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test folders.")
    parser.add_argument("--dataset_folder", required=True, help="Folder containing images and annotations.")
    parser.add_argument("--output_folder", required=True, help="Folder to save the split dataset.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    split_dataset(args.dataset_folder, args.output_folder)
    print("Dataset split completed.")

if __name__ == "__main__":
    main()
