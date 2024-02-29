# Imports
import cv2

# TrafficLightObject

class TrafficLightObject:

    def __init__(self, class_dict):

        self.frame = class_dict['frame']
        self.height = class_dict['heigth']
        self.width = class_dict['width']
        self.xmin = class_dict['bbox'][0]
        self.ymin = class_dict['bbox'][1]
        self.xmax = class_dict['bbox'][2]
        self.ymax = class_dict['bbox'][3]
        self.color = class_dict['color']
        self.textcolor = class_dict['textcolor']
        self.conf = round(class_dict['conf'], 4)

    def get_labeled_image(self):

        image = self.frame
        boxed_image = cv2.rectangle(image, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), self.color, 2)
        (w, h), _ = cv2.getTextSize(str(self.conf), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        if self.color != (255, 255, 255):
            boxed_image_copy = boxed_image.copy()
            text_color = (0, 0, 0) if self.color != (20, 20, 86) else (255, 255, 255)
            if self.ymin > h + 4:
                boxed_image_labels = cv2.rectangle(boxed_image_copy, (int(self.xmin), int(self.ymin) - (h+4)), (int(self.xmin) + w, int(self.ymin) - 2), self.color, -1)
                labelspace_image = cv2.addWeighted(boxed_image_labels, 1, boxed_image, 0, 0)
                labeled_image = cv2.putText(labelspace_image, str(self.conf), (int(self.xmin), int(self.ymin) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1) 
            else:
                boxed_image_labels = cv2.rectangle(boxed_image_copy, (int(self.xmin), int(self.ymax) + 2), (int(self.xmin) + w, int(self.ymax) + (h+4)), self.color, -1)
                labelspace_image = cv2.addWeighted(boxed_image_labels, 1, boxed_image, 0, 0)
                labeled_image = cv2.putText(labelspace_image, str(self.conf), (int(self.xmin), int(self.ymax) + (h+4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        else:
            labeled_image = boxed_image        
       
        return labeled_image


    


    
