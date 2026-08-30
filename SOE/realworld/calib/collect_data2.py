import os
import sys
import cv2
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
side_cam_serial = '135122079702'
output_dir = 'calib/out/cali_side'

if __name__ == '__main__':
    robot = FlexivRobot(default_pose=[0.6,0,0.2,0,0,1,0])
    robot.robot.setMode(robot.mode.NRT_PLAN_EXECUTION)
    robot.robot.executePlan("PLAN-FreeDriveAuto")

    cam_hand = CameraD400(hand_cam_serial, low_res=True)
    cam_side = CameraD400(side_cam_serial, low_res=True)
    os.makedirs(output_dir)
    dir_hand = os.path.join(output_dir, 'hand')
    dir_side = os.path.join(output_dir, 'side')
    os.mkdir(dir_hand)
    os.mkdir(dir_side)
    k = Keyboard(keymap=keymap)
    for i in range(30):
        # time.sleep(0.1)
        while True:
            k.state.finish = False
            data = None
            while not k.state.finish:
                curr_time = int(time.time() * 1000)
                color_image_hand, depth_image_hand = cam_hand.get_data()
                color_image, depth_image = cam_side.get_data()
                tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
                flag1, result_img = checkChessboard(color_image)
                flag2, result_img_hand = checkChessboard(color_image_hand)
                cv2.imshow('side', result_img)
                cv2.imshow('hand', result_img_hand)
                cv2.waitKey(100)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                result_img_hand = cv2.cvtColor(result_img_hand, cv2.COLOR_BGR2RGB)

                if flag1 and flag2:
                    data = curr_time, color_image_hand, depth_image_hand, color_image, depth_image, result_img, result_img_hand, tcpPose, jointPose
            if data is not None:
                break
            else:
                print('no chessboard found')
        
        curr_time, color_image_hand, depth_image_hand, color_image, depth_image, result_img, result_img_hand, tcpPose, jointPose = data
        print(i, 'saved')
        Image.fromarray(result_img_hand).save(os.path.join(dir_hand, f'{curr_time}r.png'))
        Image.fromarray(color_image_hand).save(os.path.join(dir_hand, f'{curr_time}c.png'))
        Image.fromarray(depth_image_hand).save(os.path.join(dir_hand, f'{curr_time}d.png'))
        
        Image.fromarray(result_img).save(os.path.join(dir_side, f'{curr_time}r.png'))
        Image.fromarray(color_image).save(os.path.join(dir_side, f'{curr_time}c.png'))
        Image.fromarray(depth_image).save(os.path.join(dir_side, f'{curr_time}d.png'))
        np.savetxt(os.path.join(output_dir, f'{curr_time}.txt'), tcpPose)

       

