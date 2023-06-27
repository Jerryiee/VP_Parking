import math

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
        # Store the number of frames an object has been lost for
        self.lost_frames = {}

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            cx, cy = rect
            #cx = (x + x + w) // 2
            #cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 45:
                    # If object is within radius, update its center point and reset its lost_frames counter
                    self.center_points[id] = (cx, cy)
                    self.lost_frames[id] = 0
                    #print(self.center_points)
                    objects_bbs_ids.append([cx, cy, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the next available ID to that object
            if same_object_detected is False:
                new_id = self.id_count % 51  # Assign IDs from 0 to 20 and then start from 0 again
                self.id_count += 1

                self.center_points[new_id] = (cx, cy)
                self.lost_frames[new_id] = 0
                objects_bbs_ids.append([cx, cy, new_id])

        # Increment the lost_frames counter for objects not detected in this frame
        for id in self.center_points.keys() - set(obj[2] for obj in objects_bbs_ids):
            if self.lost_frames.get(id, 0) >= 15:
                del self.center_points[id]
                del self.lost_frames[id]
            else:
                self.lost_frames[id] += 1
                objects_bbs_ids.append([self.center_points[id][0], self.center_points[id][1], id])

        return objects_bbs_ids