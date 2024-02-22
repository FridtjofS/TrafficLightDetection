import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from super_gradients.training import models

from utils import set_seed, print2way

LABELS = ['traffic_light']


class TrafficLightObjectDetector:
    def __init__(self, model_path, device='cpu'):
        '''
        Args:
            model_path: path to the model file
            device: device to run the model
        '''
        print(device)
        self.model = models.get('yolo_nas_s',
                        num_classes=1,
                        checkpoint_path=model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, imgs):
        '''
        Predict bbox of traffic lights
        
        Args:
            imgs: 
                List of images 
                Each image is a PIL.Image object
        Returns:
            bbox: predicted bbox
            probs: probability of the predicted state
            names: name of the predicted state
        '''

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model's input size
            transforms.ToTensor(),           # Convert PIL Image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])
        
        imgs = [transform(img) for img in imgs]
        imgs = torch.stack(imgs).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(imgs).show()

            # probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            # names = [LABELS[i] for i in predicted]

        return outputs#names, probs, predicted
    
 

if __name__ == "__main__":

    ### Example usage

    import matplotlib.pyplot as plt
    from PIL import Image


    # choosing 3 random images from "od_train_data" folder 
    dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(dir,'od_train_data/dataset/images/train')
    filtered_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    imgs = random.sample(filtered_list, 1)

    # load images
    imgs = [Image.open(img) for img in imgs]

    
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ########################## THIS IS THE MAIN PART ###############################
    # predict
    checkpoint = os.path.join(dir, 'checkpoints/yolo_nas_s', 'RUN_20240221_000930_574341', 'ckpt_best.pth')
    predictor = TrafficLightObjectDetector(checkpoint, device=device)
    output = predictor.predict(imgs)
    print(output[0])
    print(output[1])
    ################################################################################
    
    # print(f'Predicted State: \n{pred_states}\n') # shape (3,)
    # print(f'Probability: \n{pred_probs}\n') # shape (3, 5)
    # print(f'Predicted index: \n{pred_idxs}\n') 

    # # plot images
    # fig, ax = plt.subplots(1, 3) 
    # # figsize 
    # fig.set_figheight(4)
    # fig.set_figwidth(10)

    # for i in range(3):
    #     ax[i].imshow(imgs[i])
    #     ax[i].set_title(f'Prediction:\n{pred_probs[i][pred_idxs[i]]*100:0.2f}% {pred_states[i]}')
    #     ax[i].axis('off')
    # plt.show()


# prints something like:
#     
# Predicted State: 
# ['red_yellow', 'red', 'green']
# 
# Probability:
# [[7.3754409e-04 4.3039691e-02 9.3485707e-01 2.0665282e-02 7.0042029e-04]
#  [1.2054642e-03 9.7605747e-01 1.8882029e-02 1.9762770e-03 1.8787937e-03]
#  [1.8353603e-03 1.5284615e-02 6.3486211e-04 3.1636301e-03 9.7908151e-01]]
# 
# Predicted index:
# [2 1 4]