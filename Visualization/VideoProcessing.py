import os 
import sys
import cv2 
from PIL import Image 

class TrafficLightVideo:

    def __init__(self, save_path, temp_path=None, fps=1):

        self.save_path = save_path
        self.temp_path = temp_path
        self.fps = fps


    def make_video(self):      

        # Video Generation
                        
        print('Generating Video...')
        image_folder = self.temp_path
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")] 
        images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.video_name = os.path.basename(images[0])[:-12] + '.avi' # remove the frame number from the first image name and add .avi
        
        frame = cv2.imread(os.path.join(image_folder, images[0])) 
        
        height, width, layers = frame.shape
        video = cv2.VideoWriter(os.path.join(self.save_path, self.video_name), cv2.VideoWriter_fourcc(*"DIVX"), self.fps, (width, height)) 
        
        for image in images: 
              video.write(cv2.imread(os.path.join(image_folder, image)))

        # delete temp_path
        for file in os.listdir(self.temp_path):
            print('Deleting:', os.path.join(self.temp_path, file))
            os.remove(os.path.join(self.temp_path, file))
        os.rmdir(self.temp_path)

              
        cv2.destroyAllWindows() 
        video.release() 
        print('Done Processing.')


    def play_video(self):

        vid_reader = cv2.VideoCapture(os.path.join(self.save_path, self.video_name))
        while True:
            ret, frame = vid_reader.read()
            if not ret:
                # Restart the video when it reaches the end
                vid_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow("Video", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Exit when 'q' key is pressed
                break
            if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:  # Exit when close button is pressed
                break

        vid_reader.release()
        cv2.destroyAllWindows()