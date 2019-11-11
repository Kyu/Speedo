import numpy as np
import cv2
from datetime import datetime

movement_detected = False
start = None
stop = None
trial = 0
diff_threshold = 100
dist = 1


def calc_speed(diff):
    global dist
    # Using a magic formula

    spd = (0.0251*dist + 0.89)

    time_in_camera = timed_movement(diff)
    if time_in_camera:
        speed = time_in_camera.total_seconds() * 10
        return speed
    return None


def timed_movement(diff):
    global movement_detected, start, stop, trial, diff_threshold
    diff_max = diff.max()

    if movement_detected:
        if diff_max >= diff_threshold:
            pass
        elif diff_max < diff_threshold:
            movement_detected = False
            stop = datetime.now()

            time_taken = stop-start
            trial += 1
            # print("Trial: ", trial)
            # print(time_taken)
            start = None
            stop = None
            return time_taken
    if not movement_detected:
        if diff_max >= diff_threshold:
            start = datetime.now()
            movement_detected = True
        elif diff_max < diff_threshold:
            pass


cv2.namedWindow('MotionDetect', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MotionDetect', 800, 600)

still_img = '/home/precious/dynamics/still-img.webm'
sec_cam = '/home/precious/dynamics/security_cam.mkv'
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

ret,frame1 = cap.read()
ret,frame2 = cap.read()


while ret:
    ret, frame = cap.read()

    diff = cv2.absdiff(frame1, frame2)
    speed = calc_speed(diff)
    if speed:

        print("Speed: {0} mph".format(speed))

    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
    c, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, c, -1, (255/2, 0, 255/2), 2)
    cv2.imshow("MotionDetect", frame1)

    if cv2.waitKey(1) == 27:
        break

    frame1 = frame2
    ret, frame2 = cap.read()

cap.release()
cv2.destroyAllWindows()

# +30 cpu, + 4 ram w/o display

