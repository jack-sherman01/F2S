import os
import cv2
import sys
import time
import numpy as np
from PIL import Image

from device.keyboard_v2 import Keyboard
from device.robot import FlexivRobot
from device.camera import CameraD400

from calib.utils import checkChessboard

keymap = {
    'f': 'finish',
}
hand_cam_serial = '242322072982'
output_dir = 'calib/out/cali_hand'

# def checkChessboard(win_name, color_image, shape=(11,8)):
#     flag, corners = cv2.findChessboardCorners(color_image, shape, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
#     if flag:
#         gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
#         img = cv2.drawChessboardCorners(color_image, shape, corners2, flag)
#     else:
#         img = color_image
#     cv2.imshow(win_name, img)
#     return flag

if __name__ == '__main__':
    robot = FlexivRobot()
    robot.robot.setMode(robot.mode.NRT_PLAN_EXECUTION)
    robot.robot.executePlan("PLAN-FreeDriveAuto")

    cam = CameraD400(hand_cam_serial, low_res=True)
    os.makedirs(output_dir)
    k = Keyboard(keymap=keymap)
    for i in range(30):
        # time.sleep(0.1)
        while True:
            k.state.finish = False
            data = None
            while not k.state.finish:
                curr_time = int(time.time() * 1000)
                color_image, depth_image = cam.get_data()
                tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
                flag, result_img = checkChessboard(color_image)
                cv2.imshow('color', result_img)
                cv2.waitKey(100)
                if flag:
                    data = curr_time, tcpPose, jointPose, color_image, depth_image
            if data is not None:
                break
        curr_time, tcpPose, jointPose, color_image, depth_image = data
        print(i, 'saved')
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        Image.fromarray(color_image).save(os.path.join(output_dir, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(os.path.join(output_dir, f'{curr_time}d.png'))
        np.savetxt(os.path.join(output_dir, f'{curr_time}t.txt'), tcpPose)
        np.savetxt(os.path.join(output_dir, f'{curr_time}j.txt'), jointPose)
