import cv2

cap = cv2.VideoCapture('/Users/nadia/Documents/Uni/WS23/ProgrammierPraktikum/VizExamples/Koeln_Fahrt1/frame_9.jpg')

frame_width = cap.get(3)
frame_height = cap.get(4)

print(frame_width, frame_height)