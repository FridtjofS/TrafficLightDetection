import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

from utils import set_seed, print2way
from ResNet_withBottleneck import ResNet

LABELS = ['off', 'red', 'red_yellow', 'yellow', 'green']

class TrafficLightStatePredictor:
    def __init__(self, model_path, device='cpu'):
        '''
        Args:
            model_path: path to the model file
            device: device to run the model
        '''
        self.model = ResNet(
            num_classes=5,
            input_size=(128, 128),
            channel_size=3,
            layers=[1, 1, 1, 1],
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

        # Preprocess image
        transform =transforms.Compose([
                transforms.Resize((128, 128)),
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


    # load example images
    img1 = "sd_train_data\Berlin_102_2.jpg"
    img2 = "sd_train_data\Berlin_220_1.jpg"
    img3 = "sd_train_data\Fulda_99_1.jpg"

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

    # predict
    predictor = TrafficLightStatePredictor('models\_baseline4\model.pt', device=device)
    names, probs, predicted = predictor.predict(imgs)
    
    print(f'Predicted: \n{names}\n')
    print(f'Probability: \n{probs}\n')
    print(f'Predicted index: \n{predicted}\n')

    # plot images
    fig, ax = plt.subplots(1, 3) 
    # figsize 
    fig.set_figheight(4)
    fig.set_figwidth(10)

    for i in range(3):
        ax[i].imshow(imgs[i])
        ax[i].set_title(f'Prediction:\n{probs[i][predicted[i]]*100:0.2f}% {names[i]}')
        ax[i].axis('off')
    plt.show()


