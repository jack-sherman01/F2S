import glob
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def calc_smoothness(dataset, cam_ids):
    demos = glob.glob(f'{dataset}/*')
    print(demos)
    smoothness_list = []
    for demo in demos:
        # print(demo)
        for cam in cam_ids:
            colors = os.path.join(demo, f'cam_{cam}','color')
            depths = os.path.join(demo, f'cam_{cam}','depth')
            frame_ids = [
                int(os.path.splitext(x)[0]) 
                for x in sorted(os.listdir(colors))
            ]
            frame_ids = sorted(frame_ids)
            xyz_list = []
            for i, id in enumerate(frame_ids):
                tcp_path = os.path.join(demo, 'tcp', f'{id}.npy')
                width_path = os.path.join(demo, 'gripper_command', f'{id}.npy')

                curr_tcp = np.load(tcp_path)
                curr_width = np.load(width_path)
                curr_xyz = curr_tcp[:3]
                curr_quat = R.from_quat([*curr_tcp[4:], curr_tcp[3]])
                xyz_list.append(curr_xyz)    

            xyz = np.array(xyz_list)
            sub_smoothness_list = []
            for i in range(1, len(xyz), 20):
                print("i:", i)
                sub_xyz = xyz[i:i+20]
                sub_jerk = np.diff(sub_xyz, 3, axis=0)
                sub_smoothness = np.mean(np.linalg.norm(sub_jerk, axis=1))
                sub_smoothness_list.append(sub_smoothness)
            smoothness = np.mean(sub_smoothness_list)
            print(f'Smoothness for {demo}: {smoothness:.4f}')
            smoothness_list.append(smoothness)
    print("Jerk mean:", np.mean(smoothness_list)*1000, "m/s^3")
    print('Done!')

if __name__ == '__main__':
    cam_ids = ['135122079702', '242322072982']
    parser = argparse.ArgumentParser(description='Calculate Smoothness')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    for dataset in glob.glob(args.dataset):
        calc_smoothness(dataset, cam_ids)