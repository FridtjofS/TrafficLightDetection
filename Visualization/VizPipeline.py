import os
import sys
import cv2
import argparse
import numpy as np

### SPECIFY PATH TO PARENT DIRECTORY 'TrafficLightDetection' HERE ###
sys.path.append('TrafficLightDetection')

from StateDetection.predict import TrafficLightStatePredictor
from ObjectDetection.detect import TrafficLightDetector   ### TODO: implement 
       



def get_predictions(self, img):
    '''
    Using object detection model and state detection model, find traffic lights on given image and evaluate state of traffic lights.
    
    Args:
        img: single input image. The image is of the format ???
            
    Returns:
        predictedtrafficlights: list (or json???) of detected traffic lights in image with coordinates and states of traffic lights
    '''

    trafficlight_pos = # Hier kommt die Liste von Koordinaten vom ObjectDetection Model raus

    tafficlight_imgs = # Die Liste aus trafficlight_pos muss genutzt werden um eine liste an images von den Ampeln zu generieren, die an TrafficLightStatePredictor weitergegeben wird
    
    trafficlight_states, trafficlight_probabilities, = # Hier kommen die Ergebnisse vom TrafficLightPredictor raus

    predictedtrafficlights = # Hier kommt eine Datei raus, die die ganzen Infos von oben enthält (Liste oder json)

    return predictedtrafficlights






### Main

# Parse input

def input_parser():
    
    parser = argparse.ArgumentParser(description='Process input to VizTool')
    parser.add_argument("--input_type", help="Specify input type (0: live input; 1: input from file). ", choices=[0, 1], default=1, required=True)
    parser.add_argument("--input_path", help="Path to video or image that serves as input for VizTool. In case of live video capture, leave blank", type=str, required=False)

    return parser.parse_args()

input_type, input_path = input_parser()

# Capture input

if input_type == 0:
    while True:
        try:
            cam_idx = int(input("Please specify integer index of input camera (Usually 0 for computers with only one camera): "))
            break
        except ValueError:
            print("The camera index you sepified was not integer valued. Please try again.")
    
    cap = cv2.VideoCapture(cam_idx)
           
elif input_type == 1:

else:
    raise Exception('input_type has invalid value. Please choose value 0 or 1.')

# Process input video

if (cap.isOpened() == False):
    print('Video / camera stream could not be opened.')

while(cap.isOpened()):

    ret, frame = cap.read()
    if(ret == True):

        # Here we process the frame using the above function. We output the processed frame
        # TODO

        # Quit Camera / Video using 'q' key
        if cv.waitKey(1) == ord('q'):
            break
         
    else:
        break


cap.release()
cv2.destroyAllWindows()
print('End of video.')












