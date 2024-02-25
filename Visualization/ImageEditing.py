# Imports
import cv2

# TrafficLightObject

class TrafficLightObject:

    def __init__(self, class_dict):

        self.frame = class_dict['frame']
        self.xmin = class_dict['bbox'][0]
        self.ymin = class_dict['bbox'][1]
        self.xmax = class_dict['bbox'][2]
        self.ymax = class_dict['bbox'][3]
        self.color = class_dict['color']
        self.conf = class_dict['conf']

    def get_labeled_image(self):

        image = self.frame
        boxed_image = cv2.rectangle(image, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), self.color, 1)
        labeled_image = cv2.putText(boxed_image, str(self.conf), (int(self.xmin), int(self.ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)
       
        return labeled_image


    


    
