import os
import sys
import cv2
import argparse
import numpy as np

### SPECIFY PATH TO PARENT DIRECTORY 'TrafficLightDetection' HERE ###
sys.path.append('TrafficLightDetection')

from StateDetection.predict import TrafficLightStatePredictor
from ObjectDetection.predict import TrafficLightObjectDetector  
       

class TrafficLightClassifier:

    '''
    Using object detection model and state detection model, find traffic lights on given image and evaluate state of traffic lights.
    
    Args:
        img: single input image. The image is of the format ???
            
    Returns:
        predictedtrafficlights: list (or json???) of detected traffic lights in image with coordinates and states of traffic lights
    '''

    def __init__(self, objectdetector, statepredictor, device='cpu'):
        
        self.objectdetector = objectdetector
        self.statepredictor = statepredictor
        self.device = device

    def classify(self, image):







### Main

# Set Device -> Maybe specify in GUI as well??

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    try:
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())
    except:
        device = torch.device("cpu")


# Set working directory -> Maybe specify in GUI as well??

current_dir = os.getcwd()
print("Current working directory:", current_dir)
os.chdir("/Users/nadia/TrafficLightDetection")
working_dir = os.getcwd()
print("New current working directory:", working_dir)


# Parse input -> Will be done with GUI in the end, CHANGE

def input_parser():
    
    parser = argparse.ArgumentParser(description='Process input to VizTool')
    parser.add_argument("--input_type", help="Specify input type (0: live input; 1: input from file). ", choices=[0, 1], default=1, required=True)
    parser.add_argument("--input_path", help="Path to video or image that serves as input for VizTool. In case of live video capture, leave blank", type=str, required=False)

    return parser.parse_args()

input_type, input_path = input_parser()


# Capture input -> Will be done with GUI in the end, CHANGE

if input_type == 0:
    while True:
        try:
            cam_idx = int(input("Please specify integer index of input camera (Usually 0 for computers with only one camera): "))
            break
        except ValueError:
            print("The camera index you specified was not integer valued. Please try again.")
    
    cap = cv2.VideoCapture(cam_idx)
           
elif input_type == 1:

    cap = cv2.VideoCapture(input_path)

else:
    raise Exception('input_type has invalid value. Please choose value 0 or 1.')

# Process input video

if (cap.isOpened() == False):
    print('Video / camera stream could not be opened.')

while(cap.isOpened()):

    ret, frame = cap.read()
    if(ret == True):

        detector_path = os.path.join(dir, 'checkpoints/yolo_nas_s', 'RUN_20240223_132442_026334', 'ckpt_best.pth') # Could possibly be changed in GUI as well
        detector = TrafficLightObjectDetector(detector_path, device=device)

        predictor_path = os.path.join('StateDetection', 'models', 'model_51107', 'model.pth') # Could possibly be changed in GUI as well
        predictor = TrafficLightStatePredictor(predictor_path, device=device)

        classifier = TrafficLightClassifier(detector, predictor, device)
        output = classifier.classify(frame) 
        
        #TODO: Change output to what output will actually be in the end and show / save it....

        # Quit Camera / Video using 'q' key
        if cv.waitKey(1) == ord('q'):
            break
         
    else:
        break


cap.release()
cv2.destroyAllWindows()
print('End of video.')












