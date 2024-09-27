# -*- coding:utf-8 -*-
"""
@Time: 2024/9/28 1:22
@author: RefineM
@file: 3dgs_data_process.py
"""

import torch

def getProjectionMatrix_crop(znear, zfar, fx, fy, cx, cy, H, W):
    """
    Args:
        znear: 近平面距离
        zfar: 远平面距离
        fx: 相机水平方向焦距
        fy: 相机垂直方向焦距
        cx: 影像裁剪后x轴方向的像主点偏移
        cy: 影像裁剪后y轴方向的像主点偏移
        H: 图像高度
        W: 图像宽度
    Returns:
        P: 投影矩阵
    """
    # OPENGL：右手系RUB -> NDC: 左手系RUF
    # cx, cy是在colmap/opencv坐标系（右手系RDF）下算的
    # 转为opengl坐标系时，x轴不变，y轴取反
    
    # 计算视锥焦平面上的顶点坐标
    left = - cx
    right = W - cx 
    bottom = cy - H 
    top = cy
    
    # 由焦平面缩放到近平面
    left *= znear / fx
    right *= znear / fx
    bottom *= znear / fy
    top *= znear / fy

    P = torch.zeros(4, 4)
    # 左手系，看向Z正半轴
    z_sign = 1.0

    # 3dgs推导和opengl不同，见issue
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    
    # 分子为 zFar 表示深度映射范围是 [0, 1] 并且 zFar=>1
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

