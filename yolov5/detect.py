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
cars_in = 0
cars_out = 0
cars_total = 0
im_centroids = []
detections = []
detection_lines = None
detection_ref_lines = None
data_in = []
data_out = []
last_centroids =()
tracker = EuclideanDistTracker()
drawing= False
drawing_ref= False
free_spaces = 50

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
                        cv2.circle(im0, centroid, radius=3, color=colors(c, True), thickness=-1) 
                        im_centroids.append(centroid)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                global last_centroids, cars_in, cars_out, data_in, data_out, start_point, end_point, detection_lines, detection_ref_lines, detections
                
                if(len(last_centroids) == 0):
                    tracker.id_count = 0
                
                detection = tracker.update(im_centroids)

                if(detection_lines is not None and detection_ref_lines is not None):
                    sx1, sy1, ex1, ey1 = detection_lines
                    sxr1, syr1, exr1, eyr1 = detection_ref_lines

                    start_point = (sx1, sy1)
                    start_point_ref = (sxr1, syr1)
                    end_point = (ex1, ey1)
                    end_point_ref = (exr1, eyr1)

                    line_first = (start_point, end_point)
                    line_second = (start_point_ref, end_point_ref)

                    if(len(last_centroids) != 0):

                    #draw centroid from last positions
                    #for cntx, cnty in last_centroids: 
                     #cv2.circle(im0, (cntx, cnty), radius=5, color=(255,0,255), thickness=-1)

                        #find nearest last centroid to every current centorid 
                        for i in range(len(im_centroids)):
                            (cx, cy ,id) = detection[i]
                            nearest = min(last_centroids, key=lambda x: distanceCalculate(x, im_centroids[i]))
                            dis = distanceCalculate(nearest, im_centroids[i])

                                #cv2.circle(im0, nearest, radius=3, color=(255,0,255), thickness=-1) #draw centroid from last positions, only nearest

                            if(dis<50):
                                (ix, iy) = im_centroids[i]
                                m = (ey1-sy1)/(ex1-sx1)
                                m1 = (eyr1-syr1)/(exr1-sxr1)
                                line_centroid= m * (ix-sx1) + sy1
                                line_centroid_ref= m1 * (ix-sxr1) + syr1
                                line_points = (nearest, im_centroids[i])
                                #cv2.line(im0, nearest, im_centroids[i] ,(0,0,255),1) line betwen centroid and last near centroid
                                #chcek if lines are intersecting
                                if(intersects(line_first, line_points)):
                                    #chcek if from what side cross the line and if is id on data
                                    if(line_centroid < iy and id not in data_in):
                                        data_in.append(id)
                                        #print(id)
                                    elif(line_centroid > iy and id in data_out):
                                        cars_out += 1
                                        data_out.remove(id)
                                        with open('spaces.txt','r') as f:
                                            free_spaces = int(f.read())
                                            free_spaces += 1
                                            used_spaces = 25 - free_spaces

                                        with open('spaces.txt','w') as f2:
                                            f2.truncate() # clear previous content
                                            f2.write(f'{str(free_spaces)}')
                                        with open('usedspaces.txt','w') as f2:
                                            f2.truncate() # clear previous content
                                            f2.write(f'{str(used_spaces)}')
                                        #print(id)
                                if(intersects(line_second, line_points)):
                                    #chcek if from what side cross the line and if is id on data
                                    if(line_centroid_ref > iy and id not in data_out):
                                        data_out.append(id)

                                    elif(line_centroid_ref < iy and id in data_in):
                                        cars_in += 1
                                        data_in.remove(id)
                                        with open('spaces.txt','r') as f:
                                            free_spaces = int(f.read())
                                            free_spaces -= 1
                                            used_spaces = 25 - free_spaces

                                        with open('spaces.txt','w') as f2:
                                            f2.truncate() # clear previous content
                                            f2.write(f'{str(free_spaces)}')
                                        with open('usedspaces.txt','w') as f2:
                                            f2.truncate() # clear previous content
                                            f2.write(f'{str(used_spaces)}')



                        
                    cv2.putText(im0, str(cars_in), (sx1, sy1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)# show count
                    cv2.putText(im0, str(cars_out), (sx1-25, sy1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)# show count
                    cv2.line(im0,(start_point),(end_point),(0,0,255),1) #intercepting line
                    cv2.line(im0,(start_point_ref),(end_point_ref),(255,0,0),1) #intercepting line
                        # line numbers
                        #cv2.putText(im0, str(b), (sx1-5, sy1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)# show count
                        #cv2.putText(im0, str(b), (sxr1-5, syr1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)# show count
                #show all ids on frame
                for i in range(len(im_centroids)):
                    (cx, cy ,id) = detection[i]
                    cv2.putText(im0, str(id),(cx,cy-15),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=colors(c, True), thickness=1)

                last_centroids = im_centroids
                im_centroids = []
                detections = []

                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.waitKey(1)
                cv2.namedWindow("video")
                cv2.imshow("video", im0)
                cv2.setMouseCallback("video", drawLine)
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
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

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
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:  # Start drawing a line
            x1, y1 = x, y
            drawing = True
        else:  # Stop drawing a line
            x2, y2 = x, y
            detection_lines = [x1, y1, x2, y2]
            drawing = False
            
            print(detection_lines)

    if event == cv2.EVENT_RBUTTONDOWN:
        if drawing_ref == False:  # Start drawing a line
            x1, y1 = x, y
            drawing_ref = True
        else:  # Stop drawing a line
            x2, y2 = x, y
            detection_ref_lines = [x1, y1, x2, y2]
            drawing_ref = False
            print(detection_ref_lines)

def on_segment(p, q, r):
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r):
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 : return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
#check if seg1 and seg2 intersect

    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
#find all orientations

    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
#check general case

        return True

    if o1 == 0 and on_segment(p1, q1, p2) : return True
#heck special cases

    if o2 == 0 and on_segment(p1, q1, q2) : return True
    if o3 == 0 and on_segment(p2, q2, p1) : return True
    if o4 == 0 and on_segment(p2, q2, q1) : return True

    return False

run(source="0", weights="yolov5n6.pt", conf_thres=0.25, imgsz=(640, 640), classes= cars, view_img=True, line_thickness=1, nosave=True, device = 0) 
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
