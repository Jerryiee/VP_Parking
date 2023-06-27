# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import csv
import time
import sqlite3
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.tracker import EuclideanDistTracker

#our definitions for counting_vehicles
cars = [2,7]
im_centroids = []
detections = []
detection_lines = None
detection_ref_lines = None
data_in = []
data_out = []
prev_positions = {}
tracker = EuclideanDistTracker()
drawing= False
drawing_ref= False


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        global im_centroids, detections
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        x1 = int(xyxy[0] + (int(xyxy[2]-xyxy[0])/2))
                        y1 = int(xyxy[1] + (int(xyxy[3]-xyxy[1])/2))
                        centroid = (x1 , y1)
                        cv2.circle(im0, centroid, radius=3, color=(0,0,255), thickness=-1) 
                        im_centroids.append(centroid)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                global prev_positions
                #read database file
                conn = sqlite3.connect("kiosk/parking.db")
                c = conn.cursor()
                c.execute("SELECT capacity, occupancy FROM parking_spaces ORDER BY id DESC LIMIT 1")
                result = c.fetchone()
                capacity, occupancy = result
                nums = [capacity, occupancy]
                conn.close()
                #Open file in read mode and read coordinates
                with open("coordinates.txt", "r") as f:
                    # Read the first line, which contains the first set of coordinates
                    lines = f.readlines()
                    coordinate1 = lines[0].split()
                    sx1, sy1, ex1, ey1 = map(int, coordinate1)
                    # Read the second line, which contains the second set of coordinates
                    coordinate2 = lines[1].split()
                    sxr1, syr1, exr1, eyr1 = map(int, coordinate2)
                    detection_lines = [sx1, sy1, ex1, ey1]
                    detection_ref_lines = [sxr1, syr1, exr1, eyr1]





                if(detection_lines is not None and detection_ref_lines is not None):


                    start_point = (sx1, sy1)
                    start_point_ref = (sxr1, syr1)
                    end_point = (ex1, ey1)
                    end_point_ref = (exr1, eyr1)

                    line_first = (start_point, end_point)
                    line_second = (start_point_ref, end_point_ref)
                    
                    detection = tracker.update(im_centroids)

                    #show all ids on frame
                    for i in range(len(detection)):
                        (cx, cy ,id) = detection[i]
                        cv2.putText(im0, str(id), (int(cx), int(cy)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0,0,255), thickness=1)
                        if id in prev_positions:
                            prev_cx, prev_cy = prev_positions[id]
                             #dist error
                            cv2.line(im0, (cx, cy), (prev_cx, prev_cy), (0, 0, 255), thickness=1)
                            #print(prev_positions)

                            m = (ey1-sy1)/(ex1-sx1)
                            m1 = (eyr1-syr1)/(exr1-sxr1)
                            line_centroid= m * (cx-sx1) + sy1
                            line_centroid_ref= m1 * (cx-sxr1) + syr1
                            line_points = ((prev_cx, prev_cy), (cx, cy))
 

                            #chcek if lines are intersecting
                            if(intersects(line_first, line_points)):

                                    #chcek if from what side cross the line and if is id on data
                                if(line_centroid < cy and id not in data_in):
                                    data_in.append(id)
                                        #print(id)
                                elif(line_centroid > cy and id in data_out):
                                    data_out.remove(id)
                                    conn = sqlite3.connect("kiosk/parking.db")
                                    c = conn.cursor()
                                    c.execute("SELECT capacity, occupancy FROM parking_spaces ORDER BY id DESC LIMIT 1")
                                    result = c.fetchone()
                                    capacity, occupancy = result
                                    nums = [capacity, occupancy]

                                    nums[1] -= 1

                                    c.execute("UPDATE parking_spaces SET occupancy=? WHERE id=1", (nums[1],))
                                    conn.commit()
                                    write_to_csv(time.time(), "-1", nums[1])  # log the departure of the car
                                    conn.close()
                                    
    
                                        #print(id)
                            if(intersects(line_second, line_points)):
                                #chcek if from what side cross the line and if is id on data
                                if(line_centroid_ref > cy and id not in data_out):
                                    data_out.append(id)

                                elif(line_centroid_ref < cy and id in data_in):
                                    data_in.remove(id)
                                    conn = sqlite3.connect("kiosk/parking.db")
                                    c = conn.cursor()
                                    c.execute("SELECT capacity, occupancy FROM parking_spaces ORDER BY id DESC LIMIT 1")
                                    result = c.fetchone()
                                    capacity, occupancy = result
                                    nums = [capacity, occupancy]

                                    nums[1] += 1

                                    c.execute("UPDATE parking_spaces SET occupancy=? WHERE id=1", (nums[1],))
                                    conn.commit()
                                    write_to_csv(time.time(), "+1", nums[1])  # log the departure of the car

                                    conn.close()
    



                    #reset free spaces if there is more than capacity
                    if(nums[1] < 0):
                        nums[1] = 0
                        conn = sqlite3.connect("kiosk/parking.db")
                        c = conn.cursor()
                        c.execute("UPDATE parking_spaces SET occupancy=? WHERE id=1", (nums[1],))
                        conn.commit() 
                        conn.close()


                

                        
                    #cv2.putText(im0, text1.format(nums[1]), (1100, 785), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 2, cv2.LINE_AA)# show count
                    #cv2.putText(im0, text2.format(nums[2]), (1100, 850), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 2, cv2.LINE_AA)# show count
                    cv2.line(im0,(start_point),(end_point),(0,0,255),2) #intercepting line
                    cv2.line(im0,(start_point_ref),(end_point_ref),(255,0,0),2) #intercepting line

                        # line counting numbers
                    #cv2.putText(im0, str(nums[1]), (sx1-5, sy1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)# show count
                    #cv2.putText(im0, str(nums[2]), (sxr1-5, syr1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)# show count

                prev_positions = {}
                for i in range(len(detection)):
                    (cx, cy ,id) = detection[i]
                    prev_positions[id] = (cx, cy)
                detections = []
                im_centroids = []


                # define font and colors for text
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Define video capture
                kiosk = cv2.imread("background_texture.jpg")
                kiosk_height, kiosk_width, _ = kiosk.shape

                free_spaces = nums[0] - nums[1]

                # Show the image
                # Draw the number of vacant and occupied seats in the kiosk
                if(free_spaces > 0):
                    cv2.putText(kiosk, str(free_spaces), (kiosk_width-105, kiosk_height-150), font, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(kiosk, str(free_spaces), (kiosk_width-110, kiosk_height-150), font, 1, (0, 0, 255), 2)



                # Display frame
                cv2.namedWindow("Stream")
                cv2.imshow('Stream', im0)
                cv2.setMouseCallback("Stream", drawLine)
                cv2.imshow('Kiosk', kiosk)

                if cv2.waitKey(1) == ord('q'):
                    break
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def distanceCalculate(p1, p2): 
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def drawLine(event, x, y, flags, param):
    # Mouse event handlers for drawing lines
    global x1, y1, drawing, drawing_ref, detection_lines, detection_ref_lines

    with open("coordinates.txt", "r") as f:
    # Read the first line, which contains the first set of coordinates
        lines = f.readlines()

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:  # Start drawing a line
            x1, y1 = x, y
            drawing = True
        else:  # Stop drawing a line
            x2, y2 = x, y
            detection_lines = [x1, y1, x2, y2]
            drawing = False
            
            print(detection_lines)
            # Write the second list to the second line of the output file
            with open('coordinates.txt', 'w') as f:
                str_cor1 = ' '.join(map(str, detection_lines))
                lines[0] = str_cor1 + "\n"
                f.writelines(lines)
                

    if event == cv2.EVENT_RBUTTONDOWN:
        if drawing_ref == False:  # Start drawing a line
            x1, y1 = x, y
            drawing_ref = True
        else:  # Stop drawing a line
            x2, y2 = x, y
            detection_ref_lines = [x1, y1, x2, y2]
            drawing_ref = False
            print(detection_ref_lines)
             # Write the second list to the second line of the output file
            with open('coordinates.txt', 'w') as f:
                str_cor2 = ' '.join(map(str, detection_ref_lines))
                lines[1] = str_cor2
                f.writelines(lines)
                

def intersects(line1, line2):
    """Test if two lines intersect.
    Each line should be a tuple containing the coordinates of its endpoints.
    Returns True if the lines intersect, False otherwise."""
    
    # Extract the coordinates of the endpoints of each line
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    # Compute the slopes and y-intercepts of each line
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    b1 = y1 - m1 * x1 if x2 != x1 else x1
    
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
    b2 = y3 - m2 * x3 if x4 != x3 else x3
    
    # Test if the lines are parallel
    if m1 == m2:
        if x1 == x2:
            return True
        
        elif x3 == x4:
            return True
        else:
            return False
    
    # Compute the intersection point
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    
    # Test if the intersection point lies on both lines
    if (x < min(x1, x2) or x > max(x1, x2) or
        x < min(x3, x4) or x > max(x3, x4) or
        y < min(y1, y2) or y > max(y1, y2) or
        y < min(y3, y4) or y > max(y3, y4)):
        return False
    
    return True


# function to write data to CSV file
def write_to_csv(timestamp, status, num):
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    with open('parking_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Date and Time", "Status", "Free Spaces"])
        writer.writerow([timestamp_str, status, num])  # write timestamp, status, and free spaces count to CSV



run(source="rtsp://admin:UNIZA2020@158.193.237.17/2", weights="yolov5n.engine", conf_thres=0.30, imgsz=[640, 640], classes= cars, view_img=True, line_thickness=1, nosave=True, device = 0)
"""
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
"""
