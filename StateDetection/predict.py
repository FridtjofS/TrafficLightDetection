import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

sys.path.append('/Users/nadia/TrafficLightDetection')

from StateDetection.utils import set_seed
from StateDetection.utils import print2way
from StateDetection.ResNet_withBottleneck import ResNet

LABELS = ['off', 'red', 'red_yellow', 'yellow', 'green']

class TrafficLightStatePredictor:
    def __init__(self, model_path, device='cpu'):
        '''
        Args:
            model_path: path to the model file
            device: device to run the model
        '''
        self.input_size = (64, 64)
        self.model = ResNet(
            num_classes=5,
            input_size=self.input_size,
            channel_size=3,
            layers=[2, 2, 2, 2], #[1, 1, 1, 1],#[2, 2, 2, 2],
            out_channels=[64, 128, 256, 512],
            blocktype="simple",
            device=device,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, imgs):
        '''
        Predict the state of the detected object (traffic light)
        
        Args:
            imgs: 
                List of images of the detected object (traffic light)
                Each image is a PIL.Image object
        Returns:
            states: predicted state of the traffic light
            probs: probability of the predicted state
            names: name of the predicted state
        '''

        imgs = [img.convert('RGB') for img in imgs]

        # Preprocess image
        transform =transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        
        imgs = [transform(img) for img in imgs]
        imgs = torch.stack(imgs).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(imgs)
            state = torch.argmax(outputs, dim=1)
            predicted = state.cpu().numpy() 
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            names = [LABELS[i] for i in predicted]

        return names, probs, predicted
    
 

if __name__ == "__main__":

    ### Example usage

    import matplotlib.pyplot as plt
    from PIL import Image

    current_dir = os.getcwd()
    print("Current working directory:", current_dir)
    os.chdir("/Users/nadia/TrafficLightDetection")
    working_dir = os.getcwd()
    print("New current working directory:", working_dir)



    # choosing 3 random images from "sd_train_data" folder 
    filtered_list = [os.path.join('StateDetection', "sd_train_data", f) for f in os.listdir(os.path.join('StateDetection', "sd_train_data")) if f.endswith('.jpg')]
    imgs = random.sample(filtered_list, 3)
    
    
    img1 = r"StateDetection/examples/yellow.jpg"
    img2 = r"StateDetection/examples/green.jpg"
    img3 = r"StateDetection/examples/red_yellow.jpg"
    imgs = [img1, img2, img3]

    # load images
    imgs = [Image.open(img) for img in imgs]
    
    


    
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device(torch_directml.default_device())
        except:
            device = torch.device("cpu")

    ########################## THIS IS THE MAIN PART ###############################
    # predict
    predictor = TrafficLightStatePredictor(os.path.join('StateDetection', 'models', '_model_best3', 'model.pth'), device=device)
    pred_states, pred_probs, pred_idxs = predictor.predict(imgs)
    ################################################################################
    
    print(f'Predicted State: \n{pred_states}\n') # shape (3,)
    print(f'Probability: \n{pred_probs}\n') # shape (3, 5)
    print(f'Predicted index: \n{pred_idxs}\n') 

    # plot images
    fig, ax = plt.subplots(1, 3) 
    # figsize 
    fig.set_figheight(4)
    fig.set_figwidth(10)

    for i in range(3):
        ax[i].imshow(transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,)),
            ])(imgs[i]).transpose(0, 2).transpose(0, 1) )
        ax[i].set_title(f'Prediction:\n{pred_probs[i][pred_idxs[i]]*100:0.2f}% {pred_states[i]}')
        ax[i].axis('off')
    plt.show()


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