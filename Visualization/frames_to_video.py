# importing libraries 
import os 
import cv2 
from PIL import Image 

# Checking the current directory path 
print(os.getcwd()) 

# Folder which contains all the images 
# from which video is to be generated 
os.chdir("/Users/nadia/Documents/Uni/WS23/ProgrammierPraktikum/Daten/Koeln/Fahrt1") 
path = "/Users/nadia/Documents/Uni/WS23/ProgrammierPraktikum/Daten/Koeln/Fahrt1"

mean_height = 0
mean_width = 0

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

# Resizing of the images to give 

print('Resizing frames...')
for file in os.listdir('.'): 
	if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"): 
		# opening image using PIL Image 
		im = Image.open(os.path.join(path, file)) 

		# im.size includes the height and width of image 
		width, height = im.size 
		#print(width, height) 

		# resizing 
		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
		imResize.save( file, 'JPEG', quality = 95) # setting quality 
		# printing each resized image name 
		#print(im.filename.split('\\')[-1], " is resized") 


# Video Generating function 
def generate_video(): 
	print('Generating Video...')
	image_folder = '.' # make sure to use your folder 
	video_name = 'Fahrt1.avi'
	os.chdir("/Users/nadia/Documents/Uni/WS23/ProgrammierPraktikum/Daten/Koeln/Fahrt1") 
	
	images = [img for img in os.listdir(image_folder) 
			if img.endswith(".jpg") or
				img.endswith(".jpeg") or
				img.endswith("png")] 
	images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	
	# Array images should only consider 
	# the image files ignoring others if any 
	#print(images) 
	frame = cv2.imread(os.path.join(image_folder, images[0])) 

	# setting the frame width, height width 
	# the width, height of first image 
	height, width, layers = frame.shape 
	fps = 1
	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)) 

	# Appending the images to the video one by one 

	for image in images: 
		video.write(cv2.imread(os.path.join(image_folder, image))) 
	
	# Deallocating memories taken for window creation 
	cv2.destroyAllWindows() 
	video.release() # releasing the video generated 
	print('Done Processing.')


# Calling the generate_video function 
generate_video() 

print('Play Video:')

vid_reader = cv2.VideoCapture('/Users/nadia/Documents/Uni/WS23/ProgrammierPraktikum/Daten/Koeln/Fahrt1/Fahrt1.avi') 
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
