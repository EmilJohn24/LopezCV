import cv2
import numpy as np
import operator
"""
Huge portions of the code originate/inspired from:
https://github.com/amarlearning/Finger-Detection-and-Tracking
"""
hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)


    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)
    max_points = 20
    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        total_far_point = (0,0)
        for point in traverse_point:
            if point is not None:
                total_far_point = tuple(map(operator.add, point, total_far_point))
        average_far_point = tuple(int(val/max_points) for val in total_far_point)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if far_point is not None:
            if len(traverse_point) < max_points:
                traverse_point.append(far_point)
            else:
                traverse_point.pop(0)
                traverse_point.append(far_point)
            

        draw_circles(frame, traverse_point)
        print(str(average_far_point))
        return average_far_point, cnt_centroid

class Canvas:
    def __init__(self, name, canvas_img):
        self.canvas_img = canvas_img
        self.original_img = canvas_img.copy()
        self.paint_color = (0, 0, 0)
        self.prev_coord = None
        self.name = name
        self.radius = 10
        
    def set_color(self, color: tuple):
        self.paint_color = color
        
    def set_radius(self, radius):
        self.radius = radius
    
    def show(self):
        cv2.imshow(self.name, self.canvas_img)
    def reset(self):
        self.canvas_img = self.original_img.copy()
        self.show()
        
    def draw(self, coord):
        
        """
        #Original Painting method:
        #Size paint mask the same as canvas size
        if self.prev_coord is None:
            self.prev_coord = coord
        cv2.line(self.canvas_img, self.prev_coord, 
                 coord, self.paint_color, self.radius)
        self.prev_coord = coord
        self.show()
        """
        
        cv2.circle(self.canvas_img, coord, self.radius, self.paint_color, -1)
        self.show()
        

def main(): 
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(1)
    #Canvas setup
    _, canvas_sampler = capture.read()
    canvas_img = np.zeros([canvas_sampler.shape[0],
                           canvas_sampler.shape[1]], dtype=np.uint8)
    
    canvas_img.fill(255) #Defaults to white    
    canvas = Canvas("Drawing", canvas_img)
    finger_brush = None

    

    while capture.isOpened():
        #Hand Detection Phase
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            results = manage_image_opr(frame, hand_hist)
            if results is not None:
                finger_brush, _ = results
            cv2.imshow("Contours", hist_masking(frame, hand_hist))

        else:
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27 or pressed_key == ord('q'):
            break
        
        
        #Canvas Drawing Phase
        if pressed_key == ord(' '):
            if finger_brush is not None:
                canvas.draw(finger_brush)
                finger_brush = None
        if pressed_key == ord('r'):
            canvas.reset()
            results = None
                
    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
