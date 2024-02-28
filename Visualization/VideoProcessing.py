import os 
import sys
import cv2 
from PIL import Image 

class TrafficLightVideo:

    def __init__(self, path):

        self.path = path


    def make_video(self):
		
        path = self.path

        mean_width = 0
        mean_height = 0
		
        working_dir = os.getcwd()
        os.chdir(path)

        num_of_images = len(os.listdir('.')) 
			
        for file in os.listdir('.'):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"): 
                im = Image.open(os.path.join(path, file)) 
                width, height = im.size 
                mean_width += width 
                mean_height += height 

        # Finding the mean height and width of all images.  
                
        mean_width = int(mean_width / num_of_images) 
        mean_height = int(mean_height / num_of_images) 

        # Resizing 

        print('Resizing frames...')
        for file in os.listdir('.'): 
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"): 
                im = Image.open(os.path.join(path, file)) 
                width, height = im.size 
                imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
                imResize.save( file, 'JPEG', quality = 95)
            

        # Video Generation
                
        print('Generating Video...')
        image_folder = '.' 
        video_name = 'ProcessedVideo.avi'
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")] 
        images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        frame = cv2.imread(os.path.join(image_folder, images[0])) 
        
        height, width, layers = frame.shape 
        fps = 1
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)) 
        
        for image in images: 
              video.write(cv2.imread(os.path.join(image_folder, image))) 
              
        cv2.destroyAllWindows() 
        video.release() 
        print('Done Processing.')

        os.chdir(working_dir)

    def play_video(self):

        vid_reader = cv2.VideoCapture(os.path.join(self.path, 'ProcessedVideo.avi') ) 
        while True:
            ret, frame = vid_reader.read()
            if not ret:
                # Restart the video when it reaches the end
                vid_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow("Video", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Exit when 'q' key is pressed
                break

        vid_reader.release()
        cv2.destroyAllWindows()