from yolov5 import detect

cars =[2, 7]

img_url = 'rtsp://admin:1234@158.193.237.25/stream3'
img_url1 = 'rtsp://admin:1234@192.168.0.103:8080/h264_ulaw.sdp'
img_url3 = 'rtsp://admin:UNIZA2020@158.193.237.17/1'
img_url2 = 'traffic.mp4'
img_url5 = '0'


detect.run(source=img_url1, weights="yolov5s.pt", conf_thres=0.25, imgsz=(640, 640), classes= cars, view_img=True, line_thickness=1, nosave=True, device = 0) 