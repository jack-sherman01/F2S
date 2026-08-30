import os
import cv2
import glob
import time
import json
import torch
import shutil
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import multiprocessing

from PIL import Image
from easydict import EasyDict as edict

from eval_agent import Agent
from device.keyboard_v2 import Keyboard

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from utils.constants import *
from utils.training import set_seed
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform, xyz_rot_transform, xyz_rot_to_mat, mat_to_xyz_rot
from utils.open3d_utlis import create_point_cloud, cat_cloud, transform_cloud
from dataset.realworld import RealWorldDataset

key_map = {
    's': 'start',
    'd': 'discard',
    'q': 'quit',
    'j': 'success',
    'k': 'fail',
    'g': 'good',
    'b': 'bad',
    'h': 'drop',
    'c': 'ctn',
    'x': 'switch',
    'm': 'next_ref',
    'n': 'prev_ref',
    'i': 'next_sample',
    'o': 'prev_sample',
    'z': 'next_dim',
    'v': 'prev_dim',
    'r': 'refresh',
}
inference_blocking = True
gripper_blocking = False
default_pose = [0.6,0,0.2,0,0,1,0]
sleep_before_acting = False
sleep_after_acting = True
assert sleep_before_acting or sleep_after_acting
assert not (sleep_before_acting and sleep_after_acting)
exploring = False
vis_2d = True
vis_3d = False
vis_action_map = False
batch_size = 10
enable_FPS = False
FPS_samples = 10
enable_hil = False  # if enabled, the first sample in the batch will have no noise, for FPS purposes
# SNR = [0.027, 0.004, 269.222, 914.734, 0.004, 0.003, 0.004, 0.004, 58.328, 0.005, 0.012, 0.003, 0.003, 124.257, 1137.813, 13.943]
# SNR = [24.005, 0.003, 174.758, 0.002, 11.196, 0.004, 106.902, 0.003, 0.003, 0.003, 0.005, 0.003, 0.002, 14.674, 135.498, 21.576]
# SNR = [ 15.213  0.007  385.254  0.015  0.008  0.004  151.165  0.005  0.006 0.005  0.006  0.003  0.003  1.115  237.128  247.554]
# SNR = [15.213, 0.007, 385.254, 0.015, 0.008, 0.004, 151.165, 0.005, 0.006, 0.005, 0.006, 0.003, 0.003, 1.115, 237.128, 247.554]
# SNR = [0.021, 0.005, 706.182, 0.036, 0.009, 0.005, 105.891, 0.004, 0.005, 0.024, 0.004, 590.793, 0.004, 9.011, 303.848, 81.762]
# SNR = [143.160, 0.004, 169.481, 0.006, 0.006, 64.131, 0.004, 0.004, 0.008, 0.004, 7.174, 31.584, 0.004, 1.560, 127.580, 4.744]
# SNR = [1240.286, 12.546, 213.110, 50.729, 40.655, 1879.527, 633.812, 70.216, 442.691, 49.341, 100.147, 644.765, 11.114, 6965.954, 113.890, 267.119]
SNR = [344.864, 44.426, 132.394, 0.003, 66.363, 1.345, 0.003, 0.003, 0.004, 0.002, 0.008, 0.004, 0.003, 13.587, 31.259, 0.003]
SNR_threshold = 0.05
# SNR_threshold = 0
# dim_mask = [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1]  # mask for each dimension in the action space
# initial_dim_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# initial_dim_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# initial_dim_mask = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
initial_dim_mask = None
uniform_exploration = False
draw_normals = False

def farthest_point_sampling(action, num_samples):
    """
    Perform farthest point sampling on the action points.
    Args:
        action (np.ndarray): The action points of shape (num_points, action_step, action_dim).
        num_samples (int): The number of samples to select.
    """
    if len(action) <= num_samples:
        return action
    sampled_indices = []
    sampled_indices.append(0)  # always include the first point
    for _ in range(num_samples - 1):
        distances = -1
        sampled_index = None
        for i in range(len(action)):
            if i in sampled_indices:
                continue
            # calculate the minimum distance to the already selected points
            min_distance = np.min([np.linalg.norm(action[i,:,:3] - action[j,:,:3]) for j in sampled_indices])
            # select the point with the maximum minimum distance
            if min_distance > distances:
                distances = min_distance
                sampled_index = i
        print(f"Sampled index: {sampled_index}, distance: {distances}")
        sampled_indices.append(sampled_index)
    return action[sampled_indices]

def tcp_to_cam(tcp, intrinsic, camT):
    tcp = tcp[:3]
    tcp_vector = np.array([tcp[0], tcp[1], tcp[2], 1.0])
    tcp_vector = np.dot(np.linalg.inv(camT), tcp_vector)
    point = np.dot(intrinsic, tcp_vector[:3])
    point = point[:3] / point[2]                                        
    u = int(point[0])
    v = int(point[1])
    return u, v

def fix_tcp_to_gripper(tcp, rotation_rep):
    # compensate for gripper length
    # print("action_tcp[i, :] shape:", action_tcp[i, :].shape)
    # print("tcp_pose shape:", tcp_pose.shape)
    pose_mat = xyz_rot_to_mat(
        tcp, 
        rotation_rep=rotation_rep,
    )
    return tcp[..., :3] + np.dot(pose_mat[..., :3, :3], np.array([0, 0, 0.1]))

def unnormalize_action(action, cfg):
    if cfg.policy.name == "DP" or cfg.policy.name == "DPExt":
        action[..., :3] = (action[..., :3] + 1) / 2.0 * (WORLD_TRANS_MAX - WORLD_TRANS_MIN) + WORLD_TRANS_MIN
    elif cfg.policy.name == "RISE":
        action[..., :3] = (action[..., :3] + 1) / 2.0 * (CAM_TRANS_MAX - CAM_TRANS_MIN) + CAM_TRANS_MIN
    else:
        raise NotImplementedError
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps

def save_data(
        color_image, 
        depth_image, 
        color_dir,
        depth_dir,):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(color_image).save(color_dir)
    Image.fromarray(depth_image).save(depth_dir)
    # print("saving data to", color_dir)
                    
class Recorder:
    def __init__(
        self,      
        agent: Agent, 
        pool: multiprocessing.Pool, 
        cam_ids, 
        record_path
    ):
        self.agent = agent
        self.pool = pool
        self.cam_ids = cam_ids
        self.record_path = record_path

    def new(self):
        self.start_time = int(time.time() * 1000)
        self.demo_path = os.path.join(self.record_path, f'{self.start_time}')
        os.makedirs(self.demo_path)

        self.cam_path = [os.path.join(self.demo_path, "cam_{}".format(s)) for s in self.cam_ids]
        self.color_dir = [os.path.join(path, 'color') for path in self.cam_path]
        self.depth_dir = [os.path.join(path, 'depth') for path in self.cam_path]
        for path in self.cam_path:
            os.mkdir(path)
        for path in self.color_dir:
            os.mkdir(path)
        for path in self.depth_dir:
            os.mkdir(path)
        
        self.tcp_dir = os.path.join(self.demo_path, 'tcp')
        self.joint_dir = os.path.join(self.demo_path, 'joint')
        self.action_dir = os.path.join(self.demo_path, 'action')
        self.gripper_dir = os.path.join(self.demo_path, 'gripper_command')
        os.mkdir(self.tcp_dir)
        os.mkdir(self.joint_dir)
        os.mkdir(self.gripper_dir)
        os.mkdir(self.action_dir)
        
        with open(os.path.join(self.demo_path, "timestamp.txt"), "w") as f:
            f.write('1')

    def save(self):
        curr_time = int(time.time() * 1000)
        cam_data = []
        for camera in self.agent.cameras:
            color_image, depth_image = camera.get_data()
            cam_data.append((color_image, depth_image))
        for (color_image, depth_image), color_path, depth_path in zip(cam_data, self.color_dir, self.depth_dir):
            self.pool.apply_async(save_data, args=(
                color_image.copy(), 
                depth_image.copy(), 
                os.path.join(color_path, f'{curr_time}.png'), 
                os.path.join(depth_path, f'{curr_time}.png'))
            )
        tcpPose, jointPose, _, _ = self.agent.robot.get_robot_state()
        width = self.agent.gripper.get_gripper_state() * 1000 / self.agent.gripper.max_width
        np.save(os.path.join(self.tcp_dir, f'{curr_time}.npy'), tcpPose)
        np.save(os.path.join(self.joint_dir, f'{curr_time}.npy'), jointPose)
        np.save(os.path.join(self.gripper_dir, f'{curr_time}.npy'), [width])

    def good(self):
        with open(os.path.join(self.demo_path, "label.txt"), "w") as f:
            f.write('good')
    
    def bad(self):
        with open(os.path.join(self.demo_path, "label.txt"), "w") as f:
            f.write('bad')
    
    def success(self):
        with open(os.path.join(self.demo_path, "preference.txt"), "w") as f:
            f.write('success')
    
    def fail(self):
        with open(os.path.join(self.demo_path, "preference.txt"), "w") as f:
            f.write('fail')

    def discard(self):
        print('WARNING: discard the demo!')
        time.sleep(5)
        shutil.rmtree(self.demo_path)
    

def evaluate_single(
        cfg,
        args, 
        agent: Agent, 
        policy: nn.Module, 
        keyboard: Keyboard, 
        pool: multiprocessing.Pool, 
        device: torch.device, 
    ):
    keyboard.state.discard = False
    # keyboard.state.finish = False
    keyboard.state.quit = False
    keyboard.state.success = False    # j
    keyboard.state.fail = False       # k
    keyboard.state.good = False       # g
    keyboard.state.bad = False        # b
    keyboard.state.drop = False       # h

    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)

    if args.record:
        traj_recorder = Recorder(agent, pool, agent.cam_ids, args.record_path)
        traj_recorder.new()
        if args.step_record:
            step_recorder_path_suffix = "_steps"
            step_recorder_path = args.record_path + step_recorder_path_suffix
            step_recorder = Recorder(agent, pool, agent.cam_ids, step_recorder_path)
            step_recorder.new()

    global vis_2d, vis_3d, vis_action_map, initial_dim_mask

    if args.vis and vis_3d:
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # pcd = o3d.geometry.PointCloud()
        # vis.add_geometry(pcd)
        pass
        
    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)
    with torch.inference_mode():
        policy.eval()
        prev_width = None
        for t in range(args.max_steps):
            start_time = time.time()

            if keyboard.state.quit or keyboard.state.discard or keyboard.state.success or keyboard.state.fail:
                break

            if args.record:
                traj_recorder.save()
                if args.step_record:
                    step_recorder.save()

            if t % args.num_action == 0:
                if args.record:
                    while True and t != 0 and args.step_record:
                        cv2.waitKey(0)
                        time.sleep(0.1)
                        if keyboard.state.quit or keyboard.state.discard or keyboard.state.success or keyboard.state.fail:
                            break
                        if keyboard.state.good:
                            step_recorder.good()
                            step_recorder.new()
                            break
                        if keyboard.state.bad:
                            step_recorder.bad()
                            step_recorder.new()
                            break
                        if keyboard.state.drop:
                            step_recorder.discard()
                            step_recorder.new()
                            break
                    keyboard.state.good = False       # g
                    keyboard.state.bad = False        # b
                    keyboard.state.drop = False       # h
                    if keyboard.state.quit or keyboard.state.discard or keyboard.state.success or keyboard.state.fail:
                        break

                keyboard.state.ctn = True
                start_from_current = False
                selected_id = 0
                keyboard.state.next_sample = False
                keyboard.state.prev_sample = False
                keyboard.state.next_dim = False
                keyboard.state.prev_dim = False
                keyboard.state.refresh = False
                dim_mask = initial_dim_mask.copy() if initial_dim_mask is not None else [1] * policy.style_dim
                obs_data = None
                while keyboard.state.ctn or keyboard.state.switch or \
                    keyboard.state.next_dim or keyboard.state.prev_dim or \
                    keyboard.state.next_sample or keyboard.state.prev_sample or \
                    keyboard.state.refresh:
                    # time.sleep(0.1)

                    keyboard.state.ctn = False
                    if keyboard.state.switch:
                        # start_from_current = not start_from_current
                        global exploring
                        if not exploring:
                            enable_exploration_as_args(policy, cfg, args)
                            print("Exploration enabled")
                        else:
                            disable_exploration(policy, cfg, args)
                            print("Exploration disabled")
                        exploring = not exploring
                        keyboard.state.switch = False
                    if keyboard.state.next_dim:
                        dim_mask = dim_mask[-1:] + dim_mask[:-1]
                    if keyboard.state.prev_dim:
                        dim_mask = dim_mask[1:] + dim_mask[:1]
                    activated_dim_SNR = [SNR[i] for i in range(len(SNR)) if dim_mask[i]]
                    while SNR_threshold is not None and np.max(activated_dim_SNR) < SNR_threshold:
                        # automatically find a dimension to activate
                        print("SNR below threshold, activating next dimension")
                        if keyboard.state.prev_dim:
                            dim_mask = dim_mask[1:] + dim_mask[:1]
                        else:
                            dim_mask = dim_mask[-1:] + dim_mask[:-1]
                        activated_dim_SNR = [SNR[i] for i in range(len(SNR)) if dim_mask[i]]
                    print("dim_mask:", dim_mask)
                    print("activated dim SNR:", activated_dim_SNR)
                    keyboard.state.next_dim = False
                    keyboard.state.prev_dim = False

                    if keyboard.state.refresh:
                        print("Refreshing observation data")
                        obs_data = None
                        keyboard.state.refresh = False

                    # tcp
                    tcp_pose = agent.robot.get_tcp_pose()
                    gripper_width = agent.gripper.get_gripper_state() * 0.095 / agent.gripper.max_width
                    obs_tcps = np.stack([tcp_pose])
                    obs_grippers = np.stack([gripper_width])
                    obs_tcps = xyz_rot_transform(obs_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
                    obs_low_dim = np.concatenate((obs_tcps, obs_grippers[..., np.newaxis]), axis = -1)
                    obs_low_dim_normalized = RealWorldDataset._normalize_tcp(obs_low_dim.copy())
                    obs_low_dim = torch.from_numpy(obs_low_dim).float().to(device)
                    obs_low_dim_normalized = torch.from_numpy(obs_low_dim_normalized).float().to(device)
                    print("obs_low_dim_normalized:", obs_low_dim_normalized)

                    # rgbd
                    colors_dict, depths_dict = agent.get_observation()
                    
                    if cfg.policy.name == "DP" or cfg.policy.name == "DPExt":
                        from dataset.realworld import pre_process_inputs
                    else:
                        raise NotImplementedError
                    if obs_data is None:
                        obs_data = pre_process_inputs(colors_dict, depths_dict, device, cfg, batch_size=batch_size, obs_low_dim_normalized=obs_low_dim_normalized)

                    # predict
                    inference_start_time = time.time()
                    pred_raw_action = policy(
                        obs_data, 
                        actions = None, 
                        debug = True,
                        std_mask = [np.zeros_like(dim_mask, dtype = np.float32)] + [dim_mask] * (batch_size - 1) if enable_hil else [dim_mask],
                        uniform_exploration = uniform_exploration,
                    ).cpu().numpy()
                    print("inference time:", time.time() - inference_start_time)
                    
                    if start_from_current:
                        pred_raw_action[...,:2] = pred_raw_action[...,:2] - pred_raw_action[..., 0, :2] + obs_low_dim_normalized[0, :2].cpu().numpy()
                    
                    if "predict_delta_pos" in cfg.policy and cfg.policy.predict_delta_pos:
                        assert cfg.policy.name == "DP" or cfg.policy.name == "DPExt", "predict_delta_pos is only supported in DP and DPExt"
                        pred_raw_action[...,:3] = pred_raw_action[...,:3] + obs_low_dim_normalized[0, :3].cpu().numpy()
                    # unnormalize predicted actions
                    action = unnormalize_action(pred_raw_action, cfg)
                    
                    
                    action_tcp = action[..., :-1]
                    action_width = action[..., -1]
                    # safety insurance
                    if np.any(action_tcp[..., :3] < SAFE_WORKSPACE_MIN + SAFE_EPS) or np.any(action_tcp[..., :3] > SAFE_WORKSPACE_MAX - SAFE_EPS):
                        print("!!! Action out of safe workspace, clipping to safe workspace !!!")
                        action_tcp[..., :3] = np.clip(action_tcp[..., :3], SAFE_WORKSPACE_MIN + SAFE_EPS, SAFE_WORKSPACE_MAX - SAFE_EPS)
                    # full actions
                    action = np.concatenate([action_tcp, action_width[..., np.newaxis]], axis = -1)
                    if enable_FPS:
                        action = farthest_point_sampling(action, num_samples=FPS_samples)
                    action_tcp = action[..., :-1]
                    action_width = action[..., -1]

                    # print("start tcp:", tcp_pose[:3])
                    # print("action:", action)
                    # diff = tcp_pose[:3] - action[0,:3]
                    # print("diff:", diff)
                    
                    # visualization
                    keyboard.state.next_sample = False
                    keyboard.state.prev_sample = False
                    while True:
                        # selection
                        if args.vis:
                            if vis_2d:
                                # open a window to visualize the color image
                                for cam_id in colors_dict:
                                    color_img = cv2.cvtColor(colors_dict[cam_id], cv2.COLOR_RGB2BGR)

                                    if cam_id in [side_cam_serial, hand_cam_serial]:
                                        intrinsic = camera_intrinsics[cam_id]
                                        camT = camera_camT[cam_id] if cam_id == side_cam_serial else xyz_rot_to_mat(
                                            tcp_pose, 
                                            rotation_rep="quaternion",
                                        ) @ camera_camT[cam_id]
                                        # plot tcp
                                        u, v = tcp_to_cam(fix_tcp_to_gripper(tcp_pose, "quaternion"), intrinsic, camT)
                                        cv2.circle(color_img, (u, v), 3, (0, 0, 255), -1)
                                        # plot action points
                                        begin_color = np.array([0, 255, 255]) # BGR
                                        begin_color_selected = np.array([0, 255, 0]) # BGR
                                        begin_color_centered = np.array([0, 0, 255]) # BGR
                                        end_color = np.array([255, 0, 255])
                                        end_color_selected = np.array([255, 255, 0])
                                        end_color_centered = np.array([0, 0, 255])
                                        for b in range(len(action)):
                                            if b in [0, selected_id]:
                                                continue
                                            u_last, v_last = u, v
                                            for i in range(len(action[b, ...])):
                                                u_new, v_new = tcp_to_cam(fix_tcp_to_gripper(action_tcp[b, i, :], "rotation_6d"), intrinsic, camT)
                                                tmp_color = begin_color * (1 - i/len(action[b, ...])) + end_color * i/len(action[b, ...])
                                                cv2.line(color_img, (u_last, v_last), (u_new, v_new), (int(tmp_color[0]), int(tmp_color[1]), int(tmp_color[2])), 2)
                                                u_last, v_last = u_new, v_new
                                        u_last, v_last = u, v
                                        for i in range(len(action[0, ...])):
                                            u_new, v_new = tcp_to_cam(fix_tcp_to_gripper(action_tcp[0, i, :], "rotation_6d"), intrinsic, camT)
                                            tmp_color = begin_color_centered * (1 - i/len(action[0, ...])) + end_color_centered * i/len(action[0, ...])
                                            cv2.line(color_img, (u_last, v_last), (u_new, v_new), (int(tmp_color[0]), int(tmp_color[1]), int(tmp_color[2])), 2)
                                            u_last, v_last = u_new, v_new
                                        u_last, v_last = u, v
                                        for i in range(len(action[selected_id, ...])):
                                            u_new, v_new = tcp_to_cam(fix_tcp_to_gripper(action_tcp[selected_id, i, :], "rotation_6d"), intrinsic, camT)
                                            tmp_color = begin_color_selected * (1 - i/len(action[selected_id, ...])) + end_color_selected * i/len(action[selected_id, ...])
                                            cv2.line(color_img, (u_last, v_last), (u_new, v_new), (int(tmp_color[0]), int(tmp_color[1]), int(tmp_color[2])), 2)
                                            if draw_normals and cam_id == side_cam_serial:
                                                u_ee_new, v_ee_new = tcp_to_cam(action_tcp[selected_id, i, :], intrinsic, camT)
                                                cv2.arrowedLine(color_img, (u_new, v_new), (u_ee_new, v_ee_new), (int(tmp_color[0]), int(tmp_color[1]), int(tmp_color[2])), 1, tipLength=0.05)
                                            u_last, v_last = u_new, v_new
                                    cv2.imshow(cam_id, color_img)

                            if vis_action_map:
                                # plot tcp and action on a top-down 2D map
                                action_map_width = int((SAFE_WORKSPACE_MAX[0] - SAFE_WORKSPACE_MIN[0])*1000)
                                action_map_height = int((SAFE_WORKSPACE_MAX[1] - SAFE_WORKSPACE_MIN[1])*1000)
                                action_map = np.zeros((action_map_height, action_map_width, 3), dtype = np.uint8)
                                
                                begin_color = np.array([0, 255, 255]) # BGR
                                end_color = np.array([255, 0, 255])
                                # plot tcp
                                tcp_x = (tcp_pose[0] - SAFE_WORKSPACE_MIN[0])*1000
                                tcp_y = (tcp_pose[1] - SAFE_WORKSPACE_MIN[1])*1000
                                cv2.circle(action_map, (int(tcp_x), int(tcp_y)), 3, 
                                        (0, 0, 255), -1)
                                for b in range(len(action)):
                                    # plot action
                                    action_x = (action[b, :, 0] - SAFE_WORKSPACE_MIN[0])*1000
                                    action_y = (action[b, :, 1] - SAFE_WORKSPACE_MIN[1])*1000
                                    for i in range(len(action[b, ...])):
                                        tmp_color = begin_color * (1 - i/len(action[b, ...])) + end_color * i/len(action[b, ...])
                                        cv2.circle(action_map, (int(action_x[i]), int(action_y[i])), 1, 
                                                (int(tmp_color[0]), int(tmp_color[1]), int(tmp_color[2])), -1)
                                cv2.imshow('action_map', action_map[:,:,:].transpose(1,0,2))

                            if vis_3d:
                                # colors1, depths1 = colors_dict[agent.cam_ids[0]], depths_dict[agent.cam_ids[0]]
                                # colors2, depths2 = colors_dict[agent.cam_ids[1]], depths_dict[agent.cam_ids[1]]
                                # cloud1 = create_point_cloud(colors1, depths1, camera_intrinsics[agent.cam_ids[0]], voxel_size=0.0005)
                                # cloud2 = create_point_cloud(colors2, depths2, camera_intrinsics[agent.cam_ids[1]], voxel_size=0.0005)
                                # cloud = cat_cloud(cloud1, cloud2, camera_camT[agent.cam_ids[0]], camera_camT[agent.cam_ids[1]])

                                colors, depths = colors_dict[side_cam_serial], depths_dict[side_cam_serial]
                                cloud = create_point_cloud(colors, depths, camera_intrinsics[side_cam_serial], voxel_size=0.005)
                                cloud = transform_cloud(cloud, camera_camT[side_cam_serial])

                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                                pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                                tcp_vis_list = []
                                for b in range(len(action)):
                                    for raw_tcp in action[b, ...]:
                                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.005).translate(raw_tcp[:3])
                                        tcp_vis_list.append(tcp_vis)
                                o3d.visualization.draw_geometries([pcd, *tcp_vis_list])

                                # vis.clear_geometries()
                                # vis.add_geometry(pcd)
                                # for tcp_vis in tcp_vis_list:
                                #     vis.add_geometry(tcp_vis)

                                # vis.poll_events()
                                # vis.update_renderer()

                        global inference_blocking
                        if inference_blocking:
                            cv2.waitKey(0)
                            time.sleep(0.1)
                        else:
                            cv2.waitKey(1)

                        # NOTE: len(action) might not be equal to batch_size if FPS is enabled
                        if keyboard.state.next_sample:
                            selected_id += 1
                            selected_id = selected_id % len(action)
                            keyboard.state.next_sample = False
                            keyboard.state.prev_sample = False
                            print("current sample:", selected_id)
                        elif keyboard.state.prev_sample:
                            selected_id -= 1
                            selected_id = (len(action) + selected_id) % len(action)
                            keyboard.state.next_sample = False
                            keyboard.state.prev_sample = False
                            print("current sample:", selected_id)
                        else: 
                            break
                    
                    # time.sleep(0.1)

                # add to ensemble buffer
                ensemble_buffer.add_action(action[selected_id], t)

            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            if step_action is None:   # no action in the buffer => no movement.
                print("step_action is None")
                continue

            step_tcp = step_action[:-1]
            step_width = step_action[-1]
            # print("step width:", step_width)


            # blocking
            if sleep_before_acting:
                if time.time() - start_time < 0.1:
                    time.sleep(0.1 - (time.time() - start_time))
            # send tcp pose to robot
            if args.discretize_rotation:
                rot_steps = discretize_rotation(last_rot, step_tcp[3:], np.pi / 16)
                last_rot = step_tcp[3:]
                for rot in rot_steps:
                    step_tcp[3:] = rot
                    agent.set_tcp_pose(
                        step_tcp, 
                        rotation_rep = "rotation_6d",
                        # blocking = True
                        blocking = False
                    )
            else:
                agent.set_tcp_pose(
                    step_tcp,
                    rotation_rep = "rotation_6d",
                    # blocking = True
                    blocking = False
                )
            # blocking
            if sleep_after_acting:
                if time.time() - start_time < 0.1:
                    time.sleep(0.1 - (time.time() - start_time))


            # send gripper width to gripper (thresholding to avoid repeating sending signals to gripper)
            if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                agent.set_gripper_width(
                    step_width, 
                    blocking = gripper_blocking and (
                        prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD * 2 
                        # blocking if the difference is large
                    )
                )
                # prev_width = step_width
                prev_width = agent.gripper.get_gripper_state()
                # print("set_gripper_width:", prev_width)

            print("step time:", time.time() - start_time)

        if args.vis and vis_3d:
            # vis.destroy_window()
            pass

        if args.record:
            if keyboard.state.success:
                traj_recorder.success()
                if args.step_record:
                    step_recorder.success()
                    step_recorder.good()
                return True
            elif keyboard.state.discard or keyboard.state.quit:
                traj_recorder.discard()
                if args.step_record:
                    step_recorder.discard()
                return None
            else:
                traj_recorder.fail()
                if args.step_record:
                    step_recorder.fail()
                    step_recorder.bad()
                return False
        return None

def enable_exploration_as_args(policy, cfg, args):
    if args.enable_exploration:
        if cfg.policy.name == "DP" or cfg.policy.name == "DPExt" and args.sime:
            print("Enabling SIME-style exploration")
            policy.action_decoder.enable_exploration = True
            if args.tau1 is not None:
                policy.action_decoder.tau1 = args.tau1
            if args.tau2 is not None:
                policy.action_decoder.tau2 = args.tau2
            if args.noise_scale is not None:
                policy.action_decoder.noise_scale = args.noise_scale
            use_different_noise_scale_at_lowdim = False
            if use_different_noise_scale_at_lowdim:
                policy.action_decoder.noise_scale_for_lowdim = 0.01
                lowdim_size = 0
                for k, v in cfg.policy.params.obs_shape_meta.items():
                    if v.type == "low_dim":
                        assert len(v.shape) == 1
                        lowdim_size += v.shape[0]
                print("lowdim_size: ", lowdim_size)
                policy.action_decoder.lowdim_size = lowdim_size
        elif cfg.policy.name == "DPExt":
            print("Enabling DPExt-style exploration")
            policy.enable_exploration_extension = True
            if args.noise_scale is not None:
                policy.noise_scale = args.noise_scale

def disable_exploration(policy, cfg, args):
    if cfg.policy.name == "DP" or cfg.policy.name == "DPExt" and args.sime:
        policy.action_decoder.enable_exploration = False
    elif cfg.policy.name == "DPExt":
        policy.enable_exploration_extension = False

def evaluate(cfg, args):
    pool = multiprocessing.Pool(16)
    keyboard = Keyboard(keymap=key_map)

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # policy
    print("Loading policy ...")
    params = cfg.policy.params.copy()
    if args.num_inference_steps is not None:
        params["num_inference_steps"] = args.num_inference_steps
    if cfg.policy.name == "DP":
        from policy.dp import DP
        policy = DP(**params).to(device)
    elif cfg.policy.name == "DPExt":
        from policy.dp_ext import DPExt
        policy = DPExt(**params).to(device)
    else:
        raise NotImplementedError
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # evaluation
    agent = Agent(cam_ids=cfg.dataset.params.cam_ids, cam_low_res=True, default_pose=default_pose)

    # reference images
    if args.ref_dir is not None:
        print("Loading reference images from {}".format(args.ref_dir))
        ref_list = glob.glob(os.path.join(args.ref_dir, "*"))
        ref_list.sort()

    cnt = 0
    ref_cnt = -1
    ref_sr_list = []
    if args.ref_dir is not None:
        for ref in ref_list:
            if os.path.exists(os.path.join(ref, "preference.txt")):
                with open(os.path.join(ref, "preference.txt"), "r") as f:
                    pref = f.read().strip()
                    if pref == 'success':
                        ref_sr_list.append(2)
                    elif pref == 'fail':
                        ref_sr_list.append(0)
                    else:
                        ref_sr_list.append(-1)
            else:
                ref_sr_list.append(-1)
    same_init_repeat_cnt = 0
    while not keyboard.state.quit:
        keyboard.state.start = False
        keyboard.state.next_ref = False
        keyboard.state.prev_ref = False
        print("============")
        print("Evaluation", cnt)
        print("Press 's' to start")
        print("Press 'q' to quit")
        print("Press 'm' to show the next reference image")
        print("Press 'n' to show the previous reference image")
        print("============")
        agent.reset()
        while not keyboard.state.start and not keyboard.state.quit:
            if args.ref_dir is not None:
                if keyboard.state.next_ref:
                    ref_cnt += 1
                    if ref_cnt >= len(ref_list):
                        ref_cnt = len(ref_list) - 1
                        print("No more reference images.")
                    else:
                        print("Showing reference image {}/{} {}".format(
                            ref_cnt, len(ref_list), ref_list[ref_cnt]
                        ))
                if keyboard.state.prev_ref:
                    ref_cnt -= 1
                    if ref_cnt < 0:
                        ref_cnt = -1
                        print("Stopped showing reference images.")
                    else:
                        print("Showing reference image {}/{} {}".format(
                            ref_cnt, len(ref_list), ref_list[ref_cnt]
                        ))
                if keyboard.state.next_ref or keyboard.state.prev_ref:
                    if os.path.exists(os.path.join(ref_list[ref_cnt], "preference.txt")):
                        with open(os.path.join(ref_list[ref_cnt], "preference.txt"), "r") as f:
                            pref = f.read().strip()
                            if pref == 'success':
                                print("Reference image preference: success")
                            elif pref == 'fail':
                                print("Reference image preference: fail")
                    keyboard.state.next_ref = False
                    keyboard.state.prev_ref = False
                    same_init_repeat_cnt = 0
                
            for cam in agent.cameras:
                color_image, depth_image = cam.get_data()
                if args.ref_dir is not None and ref_cnt >= 0:
                    ref_image_paths = glob.glob(os.path.join(ref_list[ref_cnt], "cam_{}".format(cam.serial), "color", "*"))
                    ref_image_paths.sort()
                    ref_image_path = ref_image_paths[0]
                    ref_image = cv2.imread(ref_image_path)
                    if ref_image is not None:
                        color_image = cv2.addWeighted(color_image, 0.5, ref_image, 0.5, 0)
                cv2.imshow(cam.serial, color_image)
                cv2.waitKey(1)
            # time.sleep(0.1)
        if keyboard.state.quit:
            break
        is_success = evaluate_single(cfg, args, agent, policy, keyboard, pool, device)
        if not keyboard.state.discard:
            cnt += 1
            if args.ref_dir is not None:
                if is_success is None:
                    pass
                elif is_success:
                    if ref_cnt >= 0 and ref_sr_list[ref_cnt] != 2:
                        ref_sr_list[ref_cnt] = 1
                s = "".join([str(i) for i in ref_sr_list])
                print(s)

        # count success rate
        if args.record:
            demos = os.listdir(args.record_path)
            success_cnt = 0
            fail_cnt = 0
            for demo in demos:
                if os.path.exists(os.path.join(args.record_path, demo, "preference.txt")):
                    with open(os.path.join(args.record_path, demo, "preference.txt"), "r") as f:
                        pref = f.read().strip()
                        if pref == 'success':
                            success_cnt += 1
                        elif pref == 'fail':
                            fail_cnt += 1

            print("Success: {}, Fail: {}, Total: {}".format(success_cnt, fail_cnt, success_cnt + fail_cnt))
            if success_cnt + fail_cnt > 0:
                print("Success rate: {:.2f}%".format(success_cnt / (success_cnt + fail_cnt) * 100))

        same_init_repeat_cnt += 1
        print("Times cnt:", same_init_repeat_cnt)

    agent.stop()

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action = 'store', type = str, help = 'config path', required = True)

    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--num_inference_steps', action = 'store', type = int, help = 'number of diffusion inference steps', required = False, default = None)
    parser.add_argument('--max_steps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')
    parser.add_argument('--record', action = 'store_true')
    parser.add_argument('--record_path', action = 'store', type = str, required = False)
    parser.add_argument('--step_record', action = 'store_true')
    
    parser.add_argument(
        "--enable_exploration",
        action='store_true',
        help="(optional) enable modal-level exploration",
    )
    parser.add_argument(
        "--tau1",
        type=float,
        default=0.0,
        help="(optional) exploration tau1",
    )
    parser.add_argument(
        "--tau2",
        type=float,
        default=1.0,
        help="(optional) exploration tau2",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.5,
        help="(optional) exploration noise scale",
    )
    parser.add_argument(
        "--sime",
        action='store_true',
        help="(optional) use SIME-style exploration for DPExt trained policies",
    )

    parser.add_argument(
        "--ref_dir",
        type=str,
        default=None,
        help="(optional) reference image directory",
    )

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    for key in cfg.policy.params:
        if key in vars(args):
            cfg.policy.params[key] = vars(args)[key]
    print(cfg)
    evaluate(cfg, args)