"""
Hand gesture recognition
==========================
This is main function for the project.
Source Code:
https://github.com/RobinCPC/CE264-Computer_Vision
Usage:
------
    gesture_hci.py [<video source>] (default: 0)
Keys:
-----
    ESC     - exit
    c       - toggle mouse control (default: False)
    t       - toggle hand tracking (default: False)
    s       - toggle skin calibration (need debug)
"""

import cv2
import numpy as np
import math
import time

# for controlling mouse and keyboard
import pyautogui

# Fail-safe mode (prevent from ou of control)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1       # pause each pyautogui function 0.1 sec


# Build fixed-length Queue
class FixedQueue:
    """
    A container with First-In-First-Out (FIFO) queuing policy
    But can only storage maximum number of items in container
    """
    def __init__(self):
        self.list = []
        self.max_item = 5
        self.n_maj = 4

    def __str__(self):
        """Return list as string for printing"""
        return str(self.list)

    def push(self, item):
        """Enqueue the item into the queue, but check length before add in"""
        if self.list.__len__() == self.max_item:
            self.list.pop()
        self.list.insert(0, item)

    def pop(self):
        """Dequeue the earliest enqueued item still in the queue."""
        return self.list.pop()

    def major(self):
        """Return the number that shows often in list"""
        maj = 0
        count = 0
        for i in range(5):
            cur_cnt = self.list.count(i)
            if cur_cnt > count:
                maj = i
                count = cur_cnt
        return maj

    def count(self, value):
        """Return how many times that value show up in the queue"""
        return self.list.count(value)

    def isEmpty(self):
        """Return true if the queue is empty"""
        return len(self.list) == 0


# Dummy callback for trackbar
def nothing(x):
    pass

# uncomment if want to do on-line skin calibration
cv2.namedWindow('YRB_calib')
cv2.createTrackbar('Ymin', 'YRB_calib', 54, 255, nothing)
cv2.createTrackbar('Ymax', 'YRB_calib', 143, 255, nothing)
cv2.createTrackbar('CRmin', 'YRB_calib', 131, 255, nothing)
cv2.createTrackbar('CRmax', 'YRB_calib', 157, 255, nothing)
cv2.createTrackbar('CBmin', 'YRB_calib', 110, 255, nothing)
cv2.createTrackbar('CBmax', 'YRB_calib', 155, 255, nothing)

# Main part of gesture_hci
class App(object):
    def __init__(self, video_src):
        self.cam = cv2.VideoCapture(video_src)
        ret, self.frame = self.cam.read()
        cv2.namedWindow('gesture_hci')
        
        # set channel range of skin detection 
        self.mask_lower_yrb = np.array([44, 131, 80])       # [54, 131, 110]
        self.mask_upper_yrb = np.array([163, 157, 155])     # [163, 157, 135]
        # create trackbar for skin calibration
        self.calib_switch = False

        # create background subtractor 
        self.fgbg = cv2.BackgroundSubtractorMOG2(history=120, varThreshold=50, bShadowDetection=True)

        # define dynamic ROI area
        self.ROIx, self.ROIy = 200, 200
        self.track_switch = False
        # record previous positions of the centroid of ROI
        self.preCX = None
        self.preCY = None

        # A queue to record last couple gesture command
        self.last_cmds = FixedQueue()
        
        # prepare some data for detecting single-finger gesture  
        self.fin1 = cv2.imread('./test_data/index1.jpg')
        self.fin2 = cv2.imread('./test_data/index2.jpg')
        self.fin3 = cv2.imread('./test_data/index3.jpg')

        # switch to turn on mouse input control
        self.cmd_switch = False
        
        # count loop (frame), for debugging
        self.n_frame = 0


# On-line Calibration for skin detection (bug, not stable)
    def skin_calib(self, raw_yrb):
        mask_skin = cv2.inRange(raw_yrb, self.mask_lower_yrb, self.mask_upper_yrb)
        cal_skin = cv2.bitwise_and(raw_yrb, raw_yrb, mask=mask_skin)
        cv2.imshow('YRB_calib', cal_skin)
        k = cv2.waitKey(5) & 0xFF
        if k == ord('s'):
            self.calib_switch = False
            cv2.destroyWindow('YRB_calib')

        ymin = cv2.getTrackbarPos('Ymin', 'YRB_calib')
        ymax = cv2.getTrackbarPos('Ymax', 'YRB_calib')
        rmin = cv2.getTrackbarPos('CRmin', 'YRB_calib')
        rmax = cv2.getTrackbarPos('CRmax', 'YRB_calib')
        bmin = cv2.getTrackbarPos('CBmin', 'YRB_calib')
        bmax = cv2.getTrackbarPos('CBmax', 'YRB_calib')
        self.mask_lower_yrb = np.array([ymin, rmin, bmin])
        self.mask_upper_yrb = np.array([ymax, rmax, bmax])


# Do skin detection with some filtering
    def skin_detect(self, raw_yrb, img_src):
        # use median blurring to remove signal noise in YCRCB domain
        raw_yrb = cv2.medianBlur(raw_yrb, 5)
        mask_skin = cv2.inRange(raw_yrb, self.mask_lower_yrb, self.mask_upper_yrb)

        # morphological transform to remove unwanted part
        kernel = np.ones((5, 5), np.uint8)
        #mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        mask_skin = cv2.dilate(mask_skin, kernel, iterations=2)

        res_skin = cv2.bitwise_and(img_src, img_src, mask=mask_skin)
        #res_skin_dn = cv2.fastNlMeansDenoisingColored(res_skin, None, 10, 10, 7,21)

        return res_skin


# Do background subtraction with some filtering
    def background_subtract(self, img_src):
        fgmask = self.fgbg.apply(cv2.GaussianBlur(img_src, (25, 25), 0))
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        #fgmask = self.fgbg.apply(cv2.medianBlur(img_src, 11))
        org_fg = cv2.bitwise_and(img_src, img_src, mask=fgmask)
        return org_fg

# Update Position of ROI
    def update_ROI(self, img_src):
        # setting flexible ROI range
        Rxmin,Rymin,Rxmax,Rymax = (0,)*4
        if self.ROIx - 100 < 0:
            Rxmin = 0
        else:
            Rxmin = self.ROIx - 100
        
        if self.ROIx + 100 > img_src.shape[0]:
            Rxmax = img_src.shape[0]
        else:
            Rxmax = self.ROIx + 100
        
        if self.ROIy - 100 < 0:
            Rymin = 0
        else:
            Rymin = self.ROIy - 100
        
        if self.ROIy + 100 > img_src.shape[1]:
            Rymax = img_src.shape[1]
        else:
            Rymax = self.ROIy + 100
        
        return Rxmin, Rymin, Rxmax, Rymax


# Find contour and track hand inside ROI
    def find_contour(self, img_src, Rxmin, Rymin, Rxmax, Rymax):
        cv2.rectangle(img_src, (Rxmax, Rymax), (Rxmin, Rymin), (0, 255, 0), 0)
        crop_res = img_src[Rymin: Rymax, Rxmin:Rxmax]
        grey = cv2.cvtColor(crop_res, cv2.COLOR_BGR2GRAY)

        _, thresh1 = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('Thresh', thresh1)
        contours, hierchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # draw contour on threshold image
        if len(contours) > 0:
            cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
            
        return contours, crop_res


# Check ConvexHull  and Convexity Defects
    def get_defects(self, cnt, drawing):
        defects = None        
        hull = cv2.convexHull(cnt)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
        hull = cv2.convexHull(cnt, returnPoints=False)       # For finding defects
        if hull.size > 2:
            defects = cv2.convexityDefects(cnt, hull)        
        
        return defects


# Gesture Recognition
    def gesture_recognize(self, cnt, defects, count_defects, crop_res):
        # use angle between start, end, defect to recognize # of finger show up
        if type(defects) is not None and cv2.contourArea(cnt) >= 5000:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 180/math.pi
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_res, far, 5, [0, 0, 255], -1)
                cv2.line(crop_res, start, end, [0, 255, 0], 2)
                
        ## single fingertip check
        if count_defects == 0 and cv2.contourArea(cnt) >= 5000:
            count_defects = self.single_finger_check(cnt)
        
        # return the result of gesture recognition
        return count_defects


# Check if single-finger show up (OpenCV API using matchShape)
    def single_finger_check(self, cnt):
        # use single finger image to check current fame has single finger
        grey_fin1 = cv2.cvtColor(self.fin1, cv2.COLOR_BGR2GRAY)
        _, thresh_fin1 = cv2.threshold(grey_fin1, 127, 255, 0)
        contour_fin1, hierarchy = cv2.findContours(thresh_fin1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt1 = contour_fin1[0]
        ret1 = cv2.matchShapes(cnt, cnt1, 1, 0)
        
        grey_fin2 = cv2.cvtColor(self.fin2, cv2.COLOR_BGR2GRAY)
        _, thresh_fin2 = cv2.threshold(grey_fin2, 127, 255, 0)
        contour_fin2, hierarchy = cv2.findContours(thresh_fin2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt2 = contour_fin2[0]
        ret2 = cv2.matchShapes(cnt, cnt2, 1, 0)
        
        grey_fin3 = cv2.cvtColor(self.fin3, cv2.COLOR_BGR2GRAY)
        _, thresh_fin3 = cv2.threshold(grey_fin3, 127, 255, 0)
        contour_fin3, hierarchy = cv2.findContours(thresh_fin3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt3 = contour_fin3[0]
        ret3 = cv2.matchShapes(cnt, cnt3, 1, 0)
        reta = (ret1 + ret2 + ret3)/3
        if reta <= 0.3:
            return 5        # set as one-finger module
        else:
            return 0        # not detect, still 0


# Use PyAutoGUI to control mouse event
    def input_control(self, count_defects, img_src):
        # update position difference with previous frame (for move mouse)
        d_x, d_y = 0, 0
        if self.preCX is not None:
            d_x = self.ROIx - self.preCX
            d_y = self.ROIy - self.preCY
        
        # checking current command, and filter out unstable hand gesture
        cur_cmd = 0
        if self.cmd_switch:
            if self.last_cmds.count(count_defects) >= self.last_cmds.n_maj:
                cur_cmd = count_defects
                #print 'major command is ', cur_cmd
            else:
                cur_cmd = 0     # self.last_cmds.major()
        else:
            cur_cmd = count_defects
        
        # send mouse input event depend on hand gesture
        if cur_cmd == 1:
            str1 = '2, move mouse dx,dy = ' + str(d_x*3) + ', ' + str(d_y*3)
            cv2.putText(img_src, str1, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            if self.cmd_switch:
                pyautogui.moveRel(d_x*3, d_y*3)
                self.last_cmds.push(count_defects)
                #pyautogui.mouseDown(button='left')
                #pyautogui.moveRel(d_x, d_y)
            #else:
            #    pyautogui.mouseUp(button='left')
        elif cur_cmd == 2:
            cv2.putText(img_src, '3 Left (rotate)', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            if self.cmd_switch:
                pyautogui.dragRel(d_x, d_y, button='left')
                self.last_cmds.push(count_defects)
                #pyautogui.scroll(d_y,pause=0.2) 
        elif cur_cmd == 3:
            cv2.putText(img_src, '4 middle (zoom)', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            if self.cmd_switch:
                pyautogui.dragRel(d_x, d_y, button='middle')
                self.last_cmds.push(count_defects)
        elif cur_cmd == 4:
            cv2.putText(img_src, '5 right (pan)', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            if self.cmd_switch:
                pyautogui.dragRel(d_x, d_y, button='right')
                self.last_cmds.push(count_defects)
        elif cur_cmd == 5:
            cv2.putText(img_src, '1 fingertip show up', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            if self.cmd_switch:
                self.last_cmds.push(count_defects)
        else:
            cv2.putText(img_src, 'No finger detect!', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            if self.cmd_switch:
                self.last_cmds.push(count_defects)  # no finger detect or wrong gesture


# testing pyautogui
    def test_auto_gui(self):
        if self.cmd_switch:
            # Drag mouse to control some object on screen (such as google map at webpage)
            distance = 100.
            while distance > 0:
                pyautogui.dragRel(distance, 0, duration=2, button='left')    # move right
                distance -= 25
                pyautogui.dragRel(0, distance, duration=2, button='left')    # move down
                distance -= 25
                pyautogui.dragRel(-distance, 0, duration=2, button='left')    # move right
                distance -= 25
                pyautogui.dragRel(0, -distance, duration=2, button='left')    # move down
                distance -= 25

            # scroll mouse wheel (zoom in and zoom out google map)
            pyautogui.scroll(10, pause=1.)
            pyautogui.scroll(-10, pause=1)

            pyautogui.scroll(10, pause=1.)
            pyautogui.scroll(-10, pause=1)

            # message box
            pyautogui.alert(text='pyautogui testing over, click ok to end', title='Alert', button='OK')
            self.cmd_switch = not self.cmd_switch   # turn off

# main function of the project (run all processes)
    def run(self):
        while self.cam.isOpened():
            if self.n_frame == 0:
                ini_time = time.time()
            ret, self.frame = self.cam.read()
            org_vis = self.frame.copy()
            #org_vis = cv2.fastNlMeansDenoisingColored(self.frame, None, 10,10,7,21) # try to denoise but time comsuming

            ### Skin detect filter
            yrb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCR_CB)
            res_skin = self.skin_detect(yrb, org_vis)

            ## check if want to do skin calibration
            if self.calib_switch:
                self.skin_calib(yrb)
            
            ### Background Subtraction
            org_fg = self.background_subtract(org_vis)

            ### Find Contours and track hand inside ROI
            Rxmin, Rymin, Rxmax, Rymax = self.update_ROI(org_fg)
            contours, crop_res = self.find_contour(org_fg, Rxmin, Rymin, Rxmax, Rymax)

            ### Get Convexity Defects if Contour in ROI is bigger enough 
            drawing = np.zeros(crop_res.shape, np.uint8)
            max_area = -1
            ci = 0
            if len(contours) > 0:
                for i in range(len(contours)):
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if area > max_area:
                        max_area = area
                        ci = i
                cnt = contours[ci]

                # use minimum rectangle to crop conimport cv2
import numpy as np

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

    # thresh = cv2.dilate(thresh, None, iterations=5)

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

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)
        return far_point, cnt_centroid

def draw_on_canvas(canvas_img, coords, paint_color, 
                   paint_radius):
    #Size paint mask the same as canvas size
    paint_mask = np.zeros(canvas_img.shape[:2], np.uint8)
    x = coords[0]
    y = coords[1]
    paint_mask[x][y] = paint_color #Mark this as spot to be painted
    return  cv2.inpaint(canvas_img, paint_mask, 
                        paint_radius, cv2.INPAINT_TELEA)
    


def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(1)
    #Canvas setup
    _, canvas_sampler = capture.read()
    canvas_background = 1
    canvas_img = np.zeros([canvas_sampler.shape[0],
                           canvas_sampler.shape[1], canvas_background], dtype=np.uint8)
    canvas_img.fill(255) #Defaults to white
    cv2.imshow("Canvas", canvas_img)
    paint_color = 0x00FF00 #Blue
    paint_radius = 3
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
                canvas_img = draw_on_canvas(canvas_img, finger_brush, 
                                            paint_color, paint_radius)
                finger_brush = None
                cv2.imshow("Canvas", canvas_img)
    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
