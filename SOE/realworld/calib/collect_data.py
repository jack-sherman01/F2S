import os
import cv2
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from device.robot import FlexivRobot
from device.camera import CameraD400
from device.keyboard_v2 import Keyboard

keymap = {
    'd': 'done'
}
cam_serial = '242322072982'
output_dir = 'calib/out/test'

def checkChessboard(win_name, color_image, shape=(11,8)):
    color_image = color_image.copy()
    flag, corners = cv2.findChessboardCorners(color_image, shape, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if flag:
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        img = cv2.drawChessboardCorners(color_image, shape, corners2, flag)
    else:
        img = color_image
    cv2.imshow(win_name, img)
    return flag

if __name__ == '__main__':
    cam = CameraD400(cam_serial, low_res=True)
    os.makedirs(output_dir, exist_ok=True)
    k = Keyboard(keymap=keymap)
    for i in range(30):
        # time.sleep(0.1)
        while True:
            k.state.done = False
            data = None
            while not k.state.done:
                curr_time = int(time.time() * 1000)
                color_image, depth_image = cam.get_data()
                flag = checkChessboard('color', color_image)
                if flag:
                    data = curr_time, color_image, depth_image
                cv2.waitKey(100)
            if data is not None:
                break
        print(i, 'saved')
        curr_time, color_image, depth_image = data
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        Image.fromarray(color_image).save(os.path.join(output_dir, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(os.path.join(output_dir, f'{curr_time}d.png'))
