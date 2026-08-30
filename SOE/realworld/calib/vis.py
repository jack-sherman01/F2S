import os
import sys
import cv2
import glob 
import numpy as np
import transforms3d
import open3d as o3d
from PIL import Image

from calib.utils import getXYZRGB

WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

# np read from file
intrinsic = np.load('calib/out/cali_hand/intrinsic.npy')
camT = np.load('calib/out/cali_hand/camT.npy')
camsideT = np.load('calib/out/cali_side/camT.npy')
intrinsicside = np.load('calib/out/cali_side/intrinsic.npy')

# Hand = True
Hand = False

g_list= []
if Hand:
    dirName = 'cali_hand'
    flangeposList = glob.glob(f"calib/out/{dirName}/*t.txt")
    flangeposList.sort()
    imgList = glob.glob(f"calib/out/{dirName}/*c.png")
    imgList.sort()
    depthList = glob.glob(f"calib/out/{dirName}/*d.png")
    depthList.sort()
else:
    dirName = 'cali_side'
    flangeposList = glob.glob(f"calib/out/{dirName}/*.txt")
    flangeposList.sort()
    imgList = glob.glob(f"calib/out/{dirName}/hand/*c.png")
    imgList.sort()
    depthList = glob.glob(f"calib/out/{dirName}/hand/*d.png")
    depthList.sort()
    imgList2 = glob.glob(f"calib/out/{dirName}/side/*c.png")
    imgList2.sort()
    depthList2 = glob.glob(f"calib/out/{dirName}/side/*d.png")
    depthList2.sort()

for i, filename in enumerate(flangeposList):
    if len(flangeposList) > 30 and np.random.rand()< 0.9:
        continue
    ee_pose_tq = np.loadtxt(filename).tolist()
    print(ee_pose_tq)
    ee_pose_t = ee_pose_tq[:3]
    ee_pose_q = ee_pose_tq[3:]
    ee_pose_r = transforms3d.quaternions.quat2mat(ee_pose_q)


    current_pose = np.eye(4)
    current_pose[:3,:3] = ee_pose_r
    current_pose[:3,3] = ee_pose_t
    g2w = np.linalg.inv(current_pose)

    color = cv2.imread(imgList[i])
    depth = np.array(Image.open(depthList[i]), dtype = np.float32)
    
    xyzrgb = getXYZRGB(color, depth, current_pose, camT, intrinsic, )
    points = xyzrgb[:,:3]
    colors = xyzrgb[:,3:]


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    g_list.extend([pcd, o])

    if not Hand:
        color2 = cv2.imread(imgList2[i])
        depth2 = np.array(Image.open(depthList2[i]), dtype = np.float32)
        xyzrgb2 = getXYZRGB(color2, depth2, np.eye(4), camsideT, intrinsicside, )
        points2 = xyzrgb2[:,:3]
        colors2 = xyzrgb2[:,3:]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)
        g_list.extend([pcd2])

    # cam = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(camO)
    # c = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05).transform(ppp)
    # g_list.extend([c])

    # 添加桌面框
    points = [
        [0.3, -0.4, 0],
        [0.9, -0.4, 0],
        [0.3, 0.4, 0],
        [0.9, 0.4, 0],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    g_list.extend([line_set])

    if not Hand:
        o3d.visualization.draw_geometries(g_list)
        g_list = []

if Hand:
    o3d.visualization.draw_geometries(g_list)
    g_list = []