from saliency_models import gbvs, ittikochneibur
import cv2
import time
from matplotlib import pyplot as plt
import numpy as np
if __name__ == '__main__':
    image = "E:\code\ResShift-unet\S10110_origin_0.jpg"
    # image = r"E:\code\ResShift-unet\temp_img\gbvs-master\images\1.jpg"

    # 读取图像
    image = cv2.imread(image)
    print(image.shape)
    # 将图像转换为 NumPy 数组
    image_np = np.array(image, dtype=np.float32)

    # 将 NaN 值替换为 0
    image_np[np.isnan(image_np)] = 1
    nan_mask = np.isnan(image_np)
    if np.any(nan_mask):
        print("Image contains NaN values.")
        # 输出包含 NaN 值的像素的索引
        nan_indices = np.argwhere(nan_mask)
        print("Indices of NaN values:")
        print(nan_indices)
    else:
        print("Image does not contain NaN values.")

    # 将 NumPy 数组转换回 uint8 类型，以便保存或显示
    image_fixed = image_np.astype(np.uint8)

    saliency_map_gbvs = gbvs.compute_saliency(image_fixed)
    _saliency_map_gbvs = np.repeat(saliency_map_gbvs[:, :, np.newaxis], 3, axis=2)
    saliency_map_gbvs1 = np.repeat(saliency_map_gbvs[:, :, np.newaxis], 3, axis=2)


    _candy = cv2.Canny(image,0,15)
    r,g,b = cv2.split(image)
    add = (r+g+b)/3
    saliency_map_gbvs2 = saliency_map_gbvs + add
    saliency_map_gbvs2 = np.repeat(saliency_map_gbvs2[:, :, np.newaxis], 3, axis=2)

    saliency_map_gbvs1[:, :, 0] += _candy  # Red通道
    saliency_map_gbvs1[:, :, 1] += _candy  # Green通道
    saliency_map_gbvs1[:, :, 2] += _candy  # Blue通道
    # saliency_map_gbvs3 = np.stack(saliency_map_gbvs, _candy)


    print(_saliency_map_gbvs.shape,saliency_map_gbvs2.shape)

    # saliency_map_ikn = ittikochneibur.compute_saliency(image_fixed)

    oname = "gbvs.jpg"
    cv2.imwrite(oname, _saliency_map_gbvs)
    cv2.imwrite("gbvs2.jpg", saliency_map_gbvs2)
    cv2.imwrite("gbvs1.jpg", saliency_map_gbvs1)

    # cv2.imwrite("gbvs3.jpg", saliency_map_gbvs3)
