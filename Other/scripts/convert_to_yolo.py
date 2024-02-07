import os
import json
import re
import argparse
import shutil
import cv2

def json_to_yolo(json_data, image_width, image_height, image_id, default_class=0):
    yolo_lines = []

    for key, value in json_data.items():
        x_center = (value["x1"] + value["x2"]) / (2.0 * image_width)
        y_center = (value["y1"] + value["y2"]) / (2.0 * image_height)
        width = (value["x2"] - value["x1"]) / image_width
        height = (value["y2"] - value["y1"]) / image_height

        yolo_line = f"{default_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    return yolo_lines

def save_to_file(lines, output_file):
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

def process_annotations(input_folder, output_folder):
    images_folder = os.path.join(output_folder, "images")
    annotations_folder = os.path.join(output_folder, "annotations")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_file_path = os.path.join(input_folder, filename)
            
            # Extract image ID from the filename using regex
            # match = re.search(r'_(\d+)', filename)
            # if match:
            #     image_id = match.group(1)
            # else:
            #     # If no match is found, use the entire filename (without extension) as the image ID
            #     image_id = os.path.splitext(filename)[0]
            image_id = os.path.splitext(filename)[0]
            image_filename = image_id + ".jpg"
            image_file_path = os.path.join(input_folder, image_filename)

            # Move images to the images folder
            shutil.copy(image_file_path, os.path.join(images_folder, image_filename))

            with open(json_file_path, 'r') as file:
                json_data = json.load(file)

            # Get image dimensions
            img = cv2.imread(image_file_path)
            
            # with open(image_file_path, 'rb') as img_file:
            img_width, img_height = get_image_dimensions(img)

            yolo_lines = json_to_yolo(json_data, img_width, img_height, image_id)

            output_file_path = os.path.join(annotations_folder, image_id + ".txt")
            save_to_file(yolo_lines, output_file_path)

def get_image_dimensions(img_file):
    # Implement image dimension extraction logic here
    # For simplicity, let's assume the image dimensions are known
    # In practice, you may want to use a library like PIL or OpenCV to obtain the dimensions
    img_height, img_width, _ = img_file.shape
    return img_width, img_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON annotations to YOLO format")
    parser.add_argument("--input", required=True, help="Path to the folder containing both JSON annotations and images")
    parser.add_argument("--output", required=True, help="Path to the folder for saving YOLO-formatted files")

    args = parser.parse_args()

    process_annotations(args.input, args.output)

