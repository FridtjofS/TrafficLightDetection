# Traffic Light Detection Dataset

For our project, we used the DTLD datset provided by driveU (Daimler Research Institute for Vehicle Environment Perception at Ulm University). The dataset along with an in depth documentation can be found at http://www.traffic-light-data.com/.

As we implemented out our Traffic Light Detection Pipeline, we did not make use of the labels provided by driveU. Instead, we only used the unlabeled RGB images included in the dataset. 

### Image Parsing

1. In order to get the RGB images, first download the data from http://www.traffic-light-data.com/. As we do not use the labels, it is not necessary to download the label files.

2. Clone the repository:
  ```bash
  git clone https://github.com/FridtjofS/TrafficLightDetection.git
  ```

3. Navigate to the project directory:
  ```bash
  cd TrafficLightDetection/Dataset
  ```

4. Run the Skript to parse the data:
  ```bash
  python parse_data.py --data_dir <data directory> --save_dir <save directory>
  ```

Please specify the directory where the data to be processed is stored and the directory where to save the processed data.

After running the script, you find the respective RGB images in the directory you specified. Note that you will have to parse each input folder seperately. In order to improve efficiency, it makes sense to move all raw_data into one folder and parse it all at once.

### DTLD Dataset

For more information about the dataset, see:

A. Fregin, J. Muller, U. Krebel and K. Dietmayer, "The DriveU Traffic Light Dataset: Introduction and Comparison with Existing Datasets," 2018 IEEE International Conference on Robotics and Automation (ICRA), Brisbane, QLD, Australia, 2018, pp. 3376-3383, doi: 10.1109/ICRA.2018.8460737.