import os
import cv2
import argparse
import numpy as np 
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory where raw data is stored", type=str, required=True)
    parser.add_argument("--save_dir", help="Directory where parsed data should be saved", type=str, required=True)
   
    return parser.parse_args()


def main(args):

    # Load data
    data_dir = args.data_dir
    save_dir = args.save_dir

    working_dir = os.getcwd()
    os.chdir(data_dir)

    # Convert and save images
    for file in os.listdir('.'):
        if file.endswith("k0.tiff"):
            raw_image = cv2.imread(os.path.join(data_dir, file), cv2.IMREAD_UNCHANGED) 
            image = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GB2BGR)
            image = np.right_shift(image, 4)
            image = image.astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir , Path(file).stem +'.jpg'), image)

    os.chdir(working_dir)


if __name__ == "__main__":
    main(parse_args())
