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
        self.textcolor = class_dict['textcolor']
        self.conf = round(class_dict['conf'], 4)

    def get_labeled_image(self):

        image = self.frame
        boxed_image = cv2.rectangle(image, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), self.color, 1)
        (w, h), _ = cv2.getTextSize(str(self.conf), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        boxed_image_copy = boxed_image.copy()
        boxed_imgae_labels = cv2.rectangle(boxed_image_copy, (int(self.xmin), int(self.ymin) - (h+4)), (int(self.xmin) + w, int(self.ymin) - 2), self.color, -1)
        labelspace_image = cv2.addWeighted(boxed_imgae_labels, 0.7, boxed_image, 0.3, 0)
        labeled_image = cv2.putText(labelspace_image, str(self.conf), (int(self.xmin), int(self.ymin) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
       
        return labeled_image


    


    
