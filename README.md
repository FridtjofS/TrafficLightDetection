# TrafficLightDetection
This is a project to detect traffic lights in images and videos as well as the state of the traffic light (red, red-yellow, yellow, green, off), to be used in driver-assistance systems and autonomous vehicles. In order to achieve this, we split our Pipeline into multiple separate steps, which are described in the following sections.

## Table of Contents
- [TrafficLightDetection](#trafficlightdetection)
  - [Table of Contents](#table-of-contents)
  - [1. Data Acquisition](#1-data-acquisition)
  - [2. Object Detection](#2-object-detection)
  - [3. State Detection](#3-state-detection)
  - [4. Visualization](#4-visualization)
  - [Installation Guide](#installation-guide)

## 1. Data Acquisition
We have used the DTLD datset provided by driveU, which is a large dataset of traffic scenarios in Germanys cities, from a point of view of a car. Further Information can be found in the [Dataset Readme](Dataset/Readme.md). To aquire the data from the images, we then created a Annotation Tool to label the traffic lights in the images. For further information, please visit the [Annotation Tool Readme](Annotation/GUI/README.md).

## 2. Object Detection
The first step in our pipeline is to detect where the traffic lights are in the images. We have used and finetuned the YOLO NAS Object Detection Model, which is a state-of-the-art model for object detection. For further information, please visit the [Object Detection Readme](ObjectDetection/README.md).

## 3. State Detection
The second step in our pipeline is to detect the state of the traffic lights. We have implemented our own ResNet 10 State Detection Model and trained it on our data, to detect the state of the traffic lights. For further information, please visit the [State Detection Readme](StateDetection/README.md).

## 4. Visualization
The final step in our pipeline is to visualize the detected traffic lights and their states in the images and videos. We have created another Graphical User Interface to input images, videos and live camera streams and visualize the detected traffic lights and their states. This takes in the output of the Object Detection and State Detection Models and visualizes the results. For further information, please visit the [Visualization Readme](Visualization/README.md).

## Installation Guide
To install and use this project, follow these steps:

1. Clone the repository:
  ```bash
  git clone https://github.com/your-username/TrafficLightDetection.git
  ```

2. Navigate to the project directory:
  ```bash
  cd TrafficLightDetection
  ```

3. Install the required dependencies (the most important packages can be found in the requirements_manual.txt file):
  ```bash
  pip install -r requirements.txt
  ```

4. Follow the installation and usage instructions in the individual README files for each component of the pipeline:
  - [Annotation Tool Readme](Annotation/GUI/README.md)
  - [Dataset Readme](Dataset/Readme.md)
  - [Object Detection Readme](ObjectDetection/README.md)
  - [State Detection Readme](StateDetection/README.md)
  - [Visualization Readme](Visualization/README.md)

5. Once all the dependencies are installed and the individual components are set up, you can run all the parts of the pipeline.
   