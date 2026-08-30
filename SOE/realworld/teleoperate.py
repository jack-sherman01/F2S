import os
import cv2
import time
import json
import shutil
import numpy as np
import multiprocessing

from PIL import Image
from typing import List
from scipy.spatial.transform import Rotation as R

from device.camera import CameraD400
from device.robot import FlexivRobot, FlexivGripper
from device.sigma import Sigma7
from device.keyboard_v2 import Keyboard

camera_serial = ["135122079702", "242322072982"]
path_prefix = '/mnt/sdc2/jinyang/exploration/mug_hang_7_29'
key_map = {
    's': 'start',
    'f': 'finish',
    'd': 'discard',
    'q': 'quit',
    'b': 'begin',
}
default_pose = [0.6,0,0.2,0,0,1,0]

def save_data(
    color_image, 
    depth_image, 
    color_dir,
    depth_dir,
):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_image).save(color_dir)
    Image.fromarray(depth_image).save(depth_dir)

def record(robot:FlexivRobot, gripper:FlexivGripper, cameras:List[CameraD400], sigma:Sigma7, keyboard: Keyboard, pool: multiprocessing.Pool):
    start_time = int(time.time() * 1000)
    
    demo_path = os.path.join(path_prefix, f'{start_time}')
    os.makedirs(demo_path)
    cam_path = [os.path.join(demo_path, "cam_{}".format(s)) for s in camera_serial]
    color_dir = [os.path.join(path, 'color') for path in cam_path]
    depth_dir = [os.path.join(path, 'depth') for path in cam_path]
    for path in cam_path:
        os.mkdir(path)
    for path in color_dir:
        os.mkdir(path)
    for path in depth_dir:
        os.mkdir(path)

    tcp_dir = os.path.join(demo_path, 'tcp')
    joint_dir = os.path.join(demo_path, 'joint')
    action_dir = os.path.join(demo_path, 'action')
    gripper_dir = os.path.join(demo_path, 'gripper_command')
    
    os.mkdir(tcp_dir)
    os.mkdir(joint_dir)
    os.mkdir(gripper_dir)
    os.mkdir(action_dir)

    with open(os.path.join(demo_path, "timestamp.txt"), "w") as f:
        f.write('2')

    keyboard.state.start = False
    keyboard.state.discard = False
    keyboard.state.finish = False
    cnt = 0
    start_time = None
    started = False
    robot.move_to_default_pose()
    sigma.set_base()
    while not keyboard.state.quit and not keyboard.state.discard and not keyboard.state.finish:
        # time.sleep(0.1)
        curr_time = int(time.time() * 1000)
        cam_data = []
        for camera in cameras:
            color_image, depth_image = camera.get_data()
            cv2.imshow(camera.serial, color_image)
            cv2.waitKey(1)
            cam_data.append((color_image, depth_image))
        tcpPose, jointPose, _, _ = robot.get_robot_state()
        
        diff_p, diff_r, width = sigma.get_control()
        diff_p = diff_p + robot.init_pose[:3]
        # robot.init_pose is w,x,y,z 
        # but scipy assume x,y,z,w
        diff_r = R.from_quat([*robot.init_pose[4:7], robot.init_pose[3]]) * diff_r
        # Send command.
        action = np.concatenate((diff_p, diff_r.as_quat()[[3,0,1,2]]), 0)
        robot.send_tcp_pose(action)
        gripper.move_from_sigma(width)
        if not keyboard.state.start and not started:
            continue
        if not started:
            started = True
            print("start recording...")
            start_time = time.time()
        cnt += 1
        for (color_image, depth_image), color_path, depth_path in zip(cam_data, color_dir, depth_dir):
            pool.apply_async(save_data, args=(
                color_image.copy(), 
                depth_image.copy(), 
                os.path.join(color_path, f'{curr_time}.png'), 
                os.path.join(depth_path, f'{curr_time}.png'))
            )
        np.save(os.path.join(tcp_dir, f'{curr_time}.npy'), tcpPose)
        np.save(os.path.join(joint_dir, f'{curr_time}.npy'), jointPose)
        np.save(os.path.join(gripper_dir, f'{curr_time}.npy'), [width])
        np.save(os.path.join(action_dir, f'{curr_time}.npy'), action)
    if not keyboard.state.start or keyboard.state.quit or keyboard.state.discard:
        print('WARNING: discard the demo!')
        time.sleep(5)
        shutil.rmtree(demo_path)
        return
    print('saved:', demo_path, 'fps:', cnt / (time.time()-start_time + 1e-6))
    meta = {'finish_time': int(time.time() * 1000)}
    with open(os.path.join(demo_path, "metadata.json"), "w") as f:
        json.dump(meta, f)

def main():
    robot = FlexivRobot(default_pose=default_pose)
    gripper = FlexivGripper(robot)
    camera = [CameraD400(s, low_res=True) for s in camera_serial]
    for cam in camera:
        color_image, depth_image = cam.get_data()
        cv2.imshow(cam.serial, color_image)
        cv2.waitKey(1)
    sigma = Sigma7()
    keyboard = Keyboard(keymap=key_map)
    pool = multiprocessing.Pool(16)
    while not keyboard.state.quit:
        keyboard.state.begin = False
        print("============")
        print("Press 'b' to begin teleoperation")
        print("Press 'q' to quit")
        print("============")
        while not keyboard.state.begin and not keyboard.state.quit:
            for cam in camera:
                color_image, depth_image = cam.get_data()
                cv2.imshow(cam.serial, color_image)
                cv2.waitKey(1)
        if keyboard.state.quit:
            print("Quit command received, exiting...")
            break
        print("begin...")
        record(robot, gripper, camera, sigma, keyboard, pool)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()