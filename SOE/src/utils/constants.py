import os
import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
CAM_TRANS_MIN, CAM_TRANS_MAX = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7]) 
WORLD_TRANS_MIN, WORLD_TRANS_MAX = np.array([0.2, -0.4, 0.0]), np.array([0.8, 0.4, 0.4])
MAX_GRIPPER_WIDTH = 0.11 # meter

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.5])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.01 # meter

hand_cam_serial = '242322072982'
side_cam_serial = '135122079702'
camera_intrinsics = {
    '242322072982': np.load(os.path.join(
        os.path.dirname(__file__),
        '../../realworld',
        'calib/out/cali_hand/intrinsic.npy'
    )),
    '135122079702': np.load(os.path.join(
        os.path.dirname(__file__),
        '../../realworld',
        'calib/out/cali_side/intrinsic.npy'
    )),
}
camera_camT = {
    '242322072982': np.load(os.path.join(
        os.path.dirname(__file__),
        '../../realworld',
        'calib/out/cali_hand/camT.npy'
    )),
    '135122079702': np.load(os.path.join(
        os.path.dirname(__file__),
        '../../realworld',
        'calib/out/cali_side/camT.npy'
    )),
}