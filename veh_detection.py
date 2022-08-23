from yolov5 import detect
cars =[2, 7]
img_url = 'rtsp://admin:1234@158.193.237.25/stream3'

detect.run(source=img_url, weights="yolov5/yolov5s.pt", conf_thres=0.25, imgsz=(640, 448), classes= cars, view_img=True, line_thickness=1, device="cpu")