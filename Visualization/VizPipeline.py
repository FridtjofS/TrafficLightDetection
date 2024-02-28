# Imports used for TrafficLightClassifier class

from PIL import Image



# TrafficLightClassifier

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

    def classify(self, img):

        image = Image.fromarray(img)

        # find traffic lights, return pixel valued bboxes [x_min, y_min, x_max, y_max] and confidences.
        bboxes_coordinates, bboxes_confidences = self.objectdetector.predict(image)
        
        # cut out traffic lights in order for them to be passed to state prediction
        num_tl = len(bboxes_coordinates)
        tl_imgs = []
        for i in range(num_tl):
            bboxes_coordinates_int = [int(x) for x in bboxes_coordinates[i]]
            img_crop = image.crop(bboxes_coordinates_int)
            #img_crop.show()
            tl_imgs.append(img_crop)

        # predict trafficlight states
        if num_tl != 0:
            tl_states, tl_probs, tl_idxs = self.statepredictor.predict(tl_imgs)
        else: 
            tl_states = None
            tl_probs = None
            tl_idxs = None

        classification = {
            'lights' : num_tl,
            'bboxes' : bboxes_coordinates,
            'bboxes_conf' : bboxes_confidences,
            'states' : tl_states,
            'states_conf' : tl_probs,
            'states_idx' : tl_idxs
            }

        return classification



### Main
    
def main():

    # Imports 
        
    import os
    import sys
    import cv2
    import torch

    # Add parent directory to path
    sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # Imports 

    from PIL import Image
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from StateDetection.predict import TrafficLightStatePredictor
    from ObjectDetection.predict import TrafficLightObjectDetector  
    from Visualization.ImageEditing import TrafficLightObject
    from Visualization.InputGUI import get_input
    from Visualization.EndGUI import todo_next
    from Visualization.VideoProcessing import TrafficLightVideo


    # Set Device 

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device(torch_directml.default_device())
            #device = torch.device("cpu")
        except:
            device = torch.device("cpu")


    # Get input using GUI
   
    input_type, file_path, cam_num, show_status, save_status, save_dir = get_input()

    print("The input has been specified as follows.")
    print("Input Type:", input_type)
    print("Input Path:", file_path)
    print("Camera Index:", cam_num)
    print("Show processed video:", show_status)
    print("Save porcessed video:", save_status)
    print("Save Video to:", save_dir)

    # Capture input 
    if input_type == 0:
        
        cap = cv2.VideoCapture(cam_num)  # TODO: Noch testen !!!
            
    elif input_type == 1:

        cap = cv2.VideoCapture(file_path)


    # Process input video

    frame_count = 0

    if (cap.isOpened() == False):
        print('Video / camera stream could not be opened.')

    cwd = os.getcwd()
    detector_path = os.path.join(cwd, 'ObjectDetection', 'checkpoints', 'ckpt_best.pth') 
    detector = TrafficLightObjectDetector(detector_path, device=device)
    
    predictor_path = os.path.join('StateDetection', 'models', '_model_best3', 'model.pth') 
    predictor = TrafficLightStatePredictor(predictor_path, device=device)

    classifier = TrafficLightClassifier(detector, predictor, device)

    while(cap.isOpened()):

        ret, frame = cap.read()
        frame_width = cap.get(3)
        frame_height = cap.get(4)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        if(ret == True):
            
            classification = classifier.classify(frame) 

            num_lights = classification['lights']

            for i in range(num_lights):

                color = (255, 255, 255)
                textcolor = (0, 0, 0)

                #print(classification['states'][i])

                if classification['states'][i] == 'off':
                    color = (146, 146, 146)
                    textcolor = (30, 30, 30)
                elif classification['states'][i] == 'red':
                    color = (220, 20, 60)  
                    textcolor = (139, 0, 0) 
                elif classification['states'][i] == 'red_yellow':
                    color = (255, 165, 0)  
                    textcolor = (255, 140, 0) 
                elif classification['states'][i] == 'yellow':
                    color = (255, 215, 0) 
                    textcolor = (255, 255, 0)  
                elif classification['states'][i] == 'green':
                    color = (0, 100, 0)   
                    textcolor = (0, 100, 0)

                c1 = classification['bboxes_conf'][i]
                c2 = max(classification['states_conf'][i])  
                confidence = c1 #* c2   

                if c2 <= 0.6: # State Detection Threshold
                    color = (255, 255, 255)

                class_dict = {
                    'frame' : frame,
                    'heigth' : frame_height,
                    'width' : frame_width,
                    'bbox' : classification['bboxes'][i],
                    'color' : color,
                    'textcolor' : textcolor,
                    'conf' : confidence
                    }

                '''
                print('The bounding box has the following coordinates.')
                print('xmin:', class_dict['bbox'][0])
                print('ymin:', class_dict['bbox'][1])
                print('xmax:', class_dict['bbox'][2])
                print('ymax:', class_dict['bbox'][3])
                '''
                
                object = TrafficLightObject(class_dict)
           
                frame = object.get_labeled_image()

            if show_status == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('TrafficLightDetection Visualized', frame)

                if cv2.waitKey(1) == ord('q'):  # Quit Visualization using 'q' key. This quits the entire Pipeline!
                    break
            
            if save_status == True:
                cv2.imwrite(os.path.join(save_dir , 'frame_'+str(frame_count)+'.jpg'), frame)
        
            frame_count += 1
            
        else:
            task = todo_next()

            if task == 'play':
                video = TrafficLightVideo(save_dir)
                video.make_video()
                video.play_video()

            elif task == 'new':

                input_type, file_path, cam_num, show_status, save_status, save_dir = get_input()
                frame_count = 0

                if input_type == 0:
                    cap = cv2.VideoCapture(cam_num)  # TODO: Noch testen !!!
                elif input_type == 1:
                    cap = cv2.VideoCapture(file_path)
                
            elif task == 'close':
                break
            else:
                print('Do not know what to do')



    cap.release()
    cv2.destroyAllWindows()
    print('The End.')


if __name__ == '__main__':

    main()











