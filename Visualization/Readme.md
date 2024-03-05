# Traffic Light Detection Visualization Tool

This tool serves as an interface to use our Traffic Light Detection Pipeline. The user can eighter connect to a camera and process live images or input a video from file. The input is passed on to the pipeline frame by frame. The user can decide wheter each processed frame should be shown in real-time or saved to their computer. In the end, the user has the option to create a video from the processed frames and play it.

## Table of Contents
- [Installation Guide](#installation-guide)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Installation using Virtual Environment](#installation-using-virtual-environment)
- [Usage](#usage)

## Installation Guide
### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. Clone the repository:
  ```bash
  git clone https://github.com/FridtjofS/TrafficLightDetection.git
  ```

2. Download the weigths for the Object detection model and save them to the directory:
`../TrafficLightDetection/ObjectDetection/checkpoints/yolo_nas_l/ckpt_best.pth`

3. Navigate to Traffic Light Detection directory:
  ```bash
  cd TrafficLightDetection
  ```

4. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

5. Run the visualization tool:
  ```bash
  python Visualization/VizPipeline.py
  ```

### Installation using Virtual Environment

In case you face problems installing the Pipeline following the steps above, you might wnat to try using a virtual environment as there might be modules previously installed on your device that are conflicting with our requirements. 

1. Clone the repository:
  ```bash
  git clone https://github.com/FridtjofS/TrafficLightDetection.git
  ```

2. Download the weigths for the Object detection model and save them to the directory:
`../TrafficLightDetection/ObjectDetection/checkpoints/yolo_nas_l/ckpt_best.pth`

3. Navigate to Traffic Light Detection directory:
  ```bash
  cd TrafficLightDetection
  ```

4. Create virtual environment for Traffic Light Detection:
  ```bash
  python -m venv traffic_light_venv
  ```

5. Activate virtual environment:
  ```bash
  source traffic_light_venv/bin/activate
  ```

6. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

7. Run the visualization tool:
  ```bash
  python Visualization/VizPipeline.py
  ```


## Usage

1. Launch the annotation tool by running the `VizPipeline.py` script.

2. Upon starting the tool, the Input GUI will open. The user 
    - **Capture Live Video:** Choose to use camera input and specify the camera to be used. 
    - **Load Video from file:** Coose to use a video from file and select the respective file from your directory.
    - **Show annotated frames:** Check this box if you want to display each processed file in realtime. 
    - **Save annotated frames:** Check this box if you want to save each processed frame to our computer. After processing, a video of the processed frames will be created and saved to the same directory. The default saving directory is your input directory (unless you use camera input). You can change the saving directory by clicking on the button under the checkbox. 
    - **Process Video:** Once you chose an input to be processed and specified your preferences, click this button to start processing.

3. After processing, you will have the option to play the video in case you previously selected to save the frames. Also, you can process a new input or quit the Visualization tool. 

---