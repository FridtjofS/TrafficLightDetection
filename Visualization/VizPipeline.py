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

        image = Image.open(img)

        # find traffic lights, return pixel valued bboxes [x_min, y_min, x_max, y_max] and confidences.
        bboxes_coordinates, bboxes_confidences = self.objectdetector.predict(image)
        
        # cut out traffic lights in order for them to be passed to state prediction
        num_tl = len(bboxes_coordinates)
        tl_imgs = []
        for i in range(num_tl):
            img_crop = image.crop(bboxes_coordinates[i])
            tl_imgs.append(img_crop)

        # predict trafficlight states
        tl_states, tl_probs, tl_idxs = self.statepredictor.predict(tl_imgs)

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


    # Set working directory 
    
    current_dir = os.getcwd()
    print("Current working directory:", current_dir)
    os.chdir("/Users/nadia/TrafficLightDetection")
    working_dir = os.getcwd()
    print("New current working directory:", working_dir)


    # Imports 

    from PIL import Image
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from StateDetection.predict import TrafficLightStatePredictor
    from ObjectDetection.predict import TrafficLightObjectDetector  
    from Visualization.ImageEditing import TrafficLightObject
    from Visualization.VizGUI import VizGUI


    # Set Device 

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device(torch_directml.default_device())
        except:
            device = torch.device("cpu")


    # Get input using GUI

    global input_type, input_path, save, save_path

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("TL_icon.png"))
    window = VizGUI()
    window.show()
    app.exec()

    print("The input has been specified as follows.")
    print("Input Type:", input_type)
    print("Input Path:", input_path)
    print("Save Video:", save)
    if save:
        print("Save Path:", save_path)


    # Capture input 
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
    

    # Specify path to parent directory 'TrafficLightDetection'
    sys.path.append('TrafficLightDetection')


    # Process input video

    if (cap.isOpened() == False):
        print('Video / camera stream could not be opened.')

    while(cap.isOpened()):

        ret, frame = cap.read()
        if(ret == True):

            detector_path = os.path.join(dir, 'checkpoints/yolo_nas_s', 'RUN_20240223_132442_026334', 'ckpt_best.pth') 
            detector = TrafficLightObjectDetector(detector_path, device=device)

            predictor_path = os.path.join('StateDetection', 'models', 'model_51107', 'model.pth') 
            predictor = TrafficLightStatePredictor(predictor_path, device=device)

            classifier = TrafficLightClassifier(detector, predictor, device)
            classification = classifier.classify(frame) 

            num_lights = classification['lights']

            for i in range(num_lights):

                if classification['states_idx'][i] == 0:
                    color = (20, 20, 86)
                elif classification['states_idx'][i] == 1:
                    color = (245, 86, 86)
                elif classification['states_idx'][i] == 2:
                    color = (245, 150, 86)
                elif classification['states_idx'][i] == 3:
                    color = (245, 245, 86)
                elif classification['states_idx'][i] == 4:
                    color = (100, 245, 86)

                confidence = classification['bboxes_conf'][i] * classification['states_conf'][i]

                class_dict = {
                    'frame' : frame,
                    'bbox' : classification['bboxes'][i],
                    'color' : color,
                    'conf' : confidence
                    }
                
                object = TrafficLightObject(class_dict)
           
                frame = object.get_labeled_image()

            cv2.imshow(frame)
            

            # Quit Camera / Video using 'q' key
            if cv2.waitKey(1) == ord('q'):
                break
            
        else:
            break


    cap.release()
    cv2.destroyAllWindows()
    print('End of video.')



if __name__ == '__main__':

    input_type = None
    input_path = None
    save = None
    save_path = None

    main()











