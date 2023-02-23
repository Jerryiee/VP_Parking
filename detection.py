import cv2
import numpy as np
from tracker import MahalanobisDistTracker
import torch


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
tracker = MahalanobisDistTracker()
drawing= False
drawing_ref= False


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


# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set device to run inference on (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, or replace with video file path

# Loop over frames from video capture
while True:
    frame, im0 = cap.read()

    if not frame:
        break

    # Convert BGR image to RGB and resize
    img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))

    # Perform object detection on image
    results = model(img, size=640, augment=False).pandas().xyxy[0]

    # Loop over detections and draw bounding boxes around cars and trucks
    for index, row in results.iterrows():
        xyxy = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        label = row['class']
        confidence = row['confidence']
        if label == 2:
            x1 = int(xyxy[0] + (int(xyxy[2]-xyxy[0])/2))
            y1 = int(xyxy[1] + (int(xyxy[3]-xyxy[1])/2))
            centroid = (x1 , y1)
            im_centroids.append(centroid)
            cv2.circle(im0, centroid, radius=3, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 1)
            label_text = f"car: {confidence:.2f}"
            # draw label background rectangle
            label_x, label_y = int(xyxy[0]), int(xyxy[1]) - 5
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_width, label_height = text_width + 2, text_height + 2
            label_rect = np.full((label_height, label_width, 3), (0, 255, 0), dtype=np.uint8)
            im0[label_y:label_y+label_height, label_x:label_x+label_width] = label_rect
            # draw label text
            cv2.putText(im0, label_text, (label_x+1, label_y+label_height-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif label == 7:
            x1 = int(xyxy[0] + (int(xyxy[2]-xyxy[0])/2))
            y1 = int(xyxy[1] + (int(xyxy[3]-xyxy[1])/2))
            centroid = (x1 , y1)
            im_centroids.append(centroid)
            cv2.circle(im0, centroid, radius=3, color=(0, 50, 255), thickness=-1)
            cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 50, 255), 1)
            label_text = f"truck: {confidence:.2f}"
            # draw label background rectangle
            label_x, label_y = int(xyxy[0]), int(xyxy[1]) - 5
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_width, label_height = text_width + 2, text_height + 2
            label_rect = np.full((label_height, label_width, 3), (0, 50, 255), dtype=np.uint8)
            im0[label_y:label_y+label_height, label_x:label_x+label_width] = label_rect
            # draw label text
            cv2.putText(im0, label_text, (label_x+1, label_y+label_height-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    if(len(last_centroids) == 0):
        tracker.id_count = 0
                
    detection = tracker.update(im_centroids)



                    #Open file in read mode and read second line
    with open("kiosk/spaces.txt", "r") as f:
        lines = f.readlines()
        nums = lines[1].split()

    #Convert strings to integers num[1] kapacita, num[2] volne, num[3] obsadene
    nums = [int(n) for n in nums]


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
                            nums[1] += 1
                            nums[2] -= 1
    
                                        #print(id)
                    if(intersects(line_second, line_points)):
                        #chcek if from what side cross the line and if is id on data
                        if(line_centroid_ref > iy and id not in data_out):
                            data_out.append(id)

                        elif(line_centroid_ref < iy and id in data_in):
                            cars_in += 1
                            data_in.remove(id)
                            nums[1] -= 1
                            nums[2] += 1




                    
        # Modify the result string
        nums_str = "        ".join([str(n) for n in nums])
        #Replace the original line with the modified one
        lines[1] = nums_str + "\n"
        with open("kiosk/spaces.txt", "w") as f:
            f.writelines(lines)
        #Open file in read mode and read second line
        with open("kiosk/spaces.txt", "r") as f:
            lines = f.readlines()
            nums = lines[1].split()

        #Convert strings to integers
        nums = [int(n) for n in nums]

                        
        #text1="volne {}"
        #text2="obsadene {}"
        #cv2.rectangle(im0, (1050, 700), (1650, 900), (255,255,255), thickness =-1)    
        #cv2.putText(im0, text1.format(nums[1]), (1100, 785), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 2, cv2.LINE_AA)# show count
        #cv2.putText(im0, text2.format(nums[2]), (1100, 850), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 2, cv2.LINE_AA)# show count
        cv2.line(im0,(start_point),(end_point),(0,0,255),2) #intercepting line
        cv2.line(im0,(start_point_ref),(end_point_ref),(255,0,0),2) #intercepting line
            # line numbers
        cv2.putText(im0, str(nums[1]), (sx1-5, sy1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)# show count
        cv2.putText(im0, str(nums[2]), (sxr1-5, syr1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1, cv2.LINE_AA)# show count
    #show all ids on frame
    for i in range(len(im_centroids)):
        (cx, cy ,id) = detection[i]
        cv2.putText(im0, str(id),(cx,cy-15),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0,0,255), thickness=1)

    last_centroids = im_centroids
    im_centroids = []
    detections = []


    # Display frame
    cv2.namedWindow("Original")
    cv2.imshow('Original', im0)
    cv2.setMouseCallback("Original", drawLine)

    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()


