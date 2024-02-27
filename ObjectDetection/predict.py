import os
import torch
import random
from super_gradients.training import models
from PIL import Image


LABELS = ['traffic_light']


class TrafficLightObjectDetector:
    def __init__(self, model_path, device='cpu'):
        '''
        Args:
            model_path: path to the model file
            device: device to run the model
        '''
        self.model = models.get('yolo_nas_s',
                        num_classes=1,
                        checkpoint_path=model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, img):
        '''
        Predict bbox of traffic lights
        
        Args:
            img: 
                single PIL.Image object
        Returns:
            bboxes_xyxy: predicted bboxes
            confidences: probability of the predicted bboxes
        '''
        
        # Predict
        with torch.no_grad():
            output = self.model.predict(img)       
        
            confidences = output.prediction.confidence
            bboxes_xyxy = output.prediction.bboxes_xyxy
            
        return bboxes_xyxy, confidences
    
    def show(self, imgs):

        with torch.no_grad():
            outputs = self.model.predict(imgs)

        for output in outputs._images_prediction_lst:
            output.show()
            
def make_bboxes_relative(bboxes_xyxy, img):
    #TODO maybe
    pass
    
    
            

if __name__ == "__main__":
    # choosing 3 random images from "od_train_data" folder 
    dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(dir,'od_train_data/dataset/images/test')
    filtered_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    imgs_paths = random.sample(filtered_list, 3)

    # load images
    imgs = [Image.open(img) for img in imgs_paths]

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # predict
    checkpoint = os.path.join(dir, 'checkpoints/yolo_nas_s', 'RUN_20240223_132442_026334', 'ckpt_best.pth')
    predictor = TrafficLightObjectDetector(checkpoint, device=device)
    predictor.show(imgs)