import numpy as np
import cv2


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def apply_heatmap(img, heatmap):
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return img


def visualize_masks(img, ignore_mask):
    if ignore_mask.sum() > 0:
        cv2.imshow('masked_img', apply_mask(img.copy(), ignore_mask, color=(0, 0, 1)))
        cv2.waitKey()


def visualize_heatmap(img, heat_maps):
    colored = cv2.applyColorMap(heat_maps.max(axis=0), cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    cv2.imshow('heatmaps', img)
    cv2.waitKey()


def visualize_keypoints(img, keypoints, body_part_map):
    img = img.copy()
    keypoints = keypoints.astype('int32')
    for person in range(keypoints.shape[0]):
        for i in range(keypoints.shape[0]):
            x = keypoints[person, i, 0]
            y = keypoints[person, i, 1]
            if keypoints[person, i, 2] > 0:
                cv2.circle(img, (x, y), 3, (0, 1, 0), -1)
        for part in body_part_map:
            keypoint_1 = keypoints[person, part[0]]
            x_1 = keypoint_1[0]
            y_1 = keypoint_1[1]
            keypoint_2 = keypoints[person, part[1]]
            x_2 = keypoint_2[0]
            y_2 = keypoint_2[1]
            if keypoint_1[2] > 0 and keypoint_2[2] > 0:
                cv2.line(img, (x_1, y_1), (x_2, y_2), (1, 0, 0), 2)
    cv2.imshow('keypoints', img)
    cv2.waitKey()


def visualize_paf(img, pafs):
    img = img.copy()
    paf = (pafs[:,0,:,:] > 1e-8).astype('bool') | (pafs[:,0,:,:] < -1e-8).astype('bool')
    paf = (paf.max(axis=0)*255).astype('uint8')
    #paf = (pafs[:,:,:].sum(axis=0)*255).astype('uint8')
    colored = cv2.applyColorMap(paf, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    cv2.imshow('pafs', img)
    cv2.waitKey()