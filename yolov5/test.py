import cv2
try:
    vid = cv2.VideoCapture("stream3.mp4")
except cv2.error as e:
    print(e)
except:
    print('error')