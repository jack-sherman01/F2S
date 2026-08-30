import os
import sys
import cv2
import glob
import time
import numpy as np
from PIL import Image

from device.robot import FlexivRobot
from device.camera import CameraD400

from calib.utils import checkChessboard

hand_cam_serial = '242322072982'
ref_dir = 'calib/out/old/old/old/cali_hand_old'
output_dir = 'calib/out/cali_hand'

def interp(joints, k):
    n = len(joints)
    ret = [joints[0]]
    for i in range(n-1):
        for j in range(k):
            ret.append( (k-j-1)/k * joints[i] + (j+1)/k *joints[i+1])
    return ret

if __name__ == '__main__':
    robot = FlexivRobot()
    cam = CameraD400(hand_cam_serial, low_res=True)
    jointList = glob.glob(os.path.join(ref_dir, '*j.txt'))
    jointList.sort()
    os.makedirs(output_dir)
    joints = [np.loadtxt(filename) for filename in jointList]
    # joints = interp(joints, 5)

    for i, joint in enumerate(joints):
        print(joint)
        robot.send_joint_pose(joint)
        while True:
            time.sleep(0.5)
            tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
            diff = np.linalg.norm(np.array(jointPose) - np.array(joint))
            if (diff < 0.01):
                break
        time.sleep(0.5)
        curr_time = int(time.time() * 1000)
        color_image, depth_image = cam.get_data()
        print(color_image.shape)
        tcpPose, jointPose, tcpVel, jointVel = robot.get_robot_state()
        flag, result_img = checkChessboard(color_image)
        cv2.imshow('color', result_img)
        cv2.waitKey(100)
        if flag:
            print(i, 'saved')
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(result_img).save(os.path.join(output_dir, f'{curr_time}r.png'))
            Image.fromarray(color_image).save(os.path.join(output_dir, f'{curr_time}c.png'))
            Image.fromarray(depth_image).save(os.path.join(output_dir, f'{curr_time}d.png'))
            np.savetxt(os.path.join(output_dir, f'{curr_time}t.txt'), tcpPose)
            np.savetxt(os.path.join(output_dir, f'{curr_time}j.txt'), jointPose)
        else:
            print('chessboard not found')
