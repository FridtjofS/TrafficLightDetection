import os
import json
import cv2
from pathlib import Path
import sys

# fix relative imports
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, path)

def augment_data(od_path, sd_path, output_path):
    
  img_filenames = os.listdir(os.path.join(od_path, "images"))

  for img_filename in img_filenames:
    img = cv2.imread(os.path.join(od_path, "images", img_filename))
    img_height, img_width, _ = img.shape

    with open(os.path.join(od_path, "annotations", Path(img_filename).stem + ".txt"), "r") as f:
      lines = f.readlines()
      for i in range(len(lines)):
        line = lines[i].strip().split(" ")
        x, y, w, h = map(float, line[1:])


        # load state from sd_train_data folder
        with open(os.path.join(sd_path, Path(img_filename).stem + "_" + str(i) + ".json"), "r") as f:
          state = json.load(f)["state"]
        
        # double the bbox size
        x1 = int((x - w) * img_width)
        y1 = int((y - h) * img_height)
        x2 = int((x + w) * img_width)
        y2 = int((y + h) * img_height)

        # calculate padding
        top_bottom = int(h * img_width)
        left_right = int(w * img_height)

        # Pad the image with repeated edge pixels
        padded_img = cv2.copyMakeBorder(img, top_bottom, top_bottom, left_right, left_right, cv2.BORDER_REPLICATE)

        # Crop the padded image
        cropped_img = padded_img[y1 + top_bottom:y2 + top_bottom, x1 + left_right:x2 + left_right]

        # rescale image to 256x256
        #cropped_img = cv2.resize(cropped_img, (256, 256))
        cropped_img = cv2.resize(cropped_img, (128, 128))

        # save cropped image
        cropped_img_filename = Path(img_filename).stem + "_" + str(i) + ".jpg"
        cv2.imwrite(os.path.join(output_path, cropped_img_filename), cropped_img)

        # save state in json format
        state_dict = {"state": state}
        with open(os.path.join(output_path, Path(img_filename).stem + "_" + str(i) + ".json"), "w") as f:
          json.dump(state_dict, f)


  # load images from od_train_data folder
  # load their bboxes in yolo format
  # double the bbox size
  # load states from sd_train_data folder
  # save in output folder the cropped images and state as json



od_path = os.path.join("..", "ObjectDetection", "od_train_data", "raw_dataset") #"ObjectDetection\od_train_data\raw_dataset"
sd_path = os.path.join("..", "StateDetection", "sd_train_data") #"StateDetection\sd_train_data"
output_path = os.path.join("..", "StateDetection", "augmented_dataset") #"StateDetection\augmented_dataset"

def main():
  """
  Augment data, such that the images are cropped around twice the size of the bounding box
  and the state is saved in json format, such that it can be used for training the state detection model.
  """
  augment_data(od_path, sd_path, output_path)

if __name__ == "__main__":
  main()