import numpy as np
import random
import cv2

sigma_inp = 7
n = sigma_inp * 6 + 1
g_inp = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        g_inp[i, j] = np.exp(-((i - n / 2) ** 2 + (j - n / 2) ** 2) / (2. * sigma_inp * sigma_inp))


# https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/utils/img.py
def Gaussian(sigma):
    if sigma == 7:
        return np.array([0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529,
                         0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
                         0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
                         0.2301, 0.5205, 0.8494, 1.0000, 0.8494, 0.5205, 0.2301,
                         0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
                         0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
                         0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529]).reshape(7, 7)
    elif sigma == n:
        return g_inp
    else:
        raise Exception('Gaussian {} Not Implement'.format(sigma))


# https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/utils/img.py
def DrawGaussian(img, pt, sigma):
    tmpSize = int(np.math.ceil(3 * sigma))
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = Gaussian(size)

    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def flip(img, ignore_mask, keypoints, flip_indices):
    width = img.shape[1]
    img = cv2.flip(img, 1)
    ignore_mask = cv2.flip(ignore_mask, 1)
    keypoints[:,:,0] = width - 1 - keypoints[:,:,0]
    for flip_id in flip_indices:
        temp =  keypoints[:, flip_id[0], :].copy()
        keypoints[:, flip_id[0], :] = keypoints[:, flip_id[1], :]
        keypoints[:, flip_id[1], :] = temp
    return img, ignore_mask, keypoints


def resize(img, ignore_mask, keypoints, imgSize):
    width, height = img.shape[0], img.shape[1]
    img = cv2.resize(img, (imgSize, imgSize))
    ignore_mask = cv2.resize( ignore_mask, (imgSize, imgSize))
    keypoints[:, :, 0] = keypoints[:, :, 0] * imgSize / height
    keypoints[:, :, 1] = keypoints[:, :, 1] * imgSize / width
    return img, ignore_mask, keypoints


def resize_hm(heatmap, paf, ignore_mask, hm_size):
    ignore_mask = cv2.resize(ignore_mask, (hm_size, hm_size))
    heatmap = cv2.resize(heatmap.transpose(1, 2, 0), (hm_size, hm_size))
    heatmap = heatmap.transpose(2, 0, 1)
    paf = paf.transpose(2,3,0,1)
    paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
    paf = cv2.resize(paf, (hm_size, hm_size))
    paf = paf.transpose(2, 0, 1)
    return heatmap, paf, ignore_mask


def color_augment(img, ignore_mask, keypoints, color_aug):
    for channel in range(img.shape[2]):
        img[:, :, channel] = np.clip(img[:, :, channel] * (np.random.random()*color_aug*2 + 1 - color_aug) , 0, 1)
    return img, ignore_mask, keypoints
