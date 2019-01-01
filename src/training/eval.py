import cv2


def eval(test_loader, outputs):
    keypoints_ar = []
    indices_ar = []
    for i in range(len(outputs)):
        outputs_batch = outputs[i]
        for j in range(outputs_batch.shape[0]):
            output = outputs_batch[j]
            heatmaps = output[0]
            pafs = output[1]
            index = output[2]
            keypoints = extract_keypoints(heatmaps, pafs)
            keypoints_ar.append(keypoints)
            indices_ar.append((index))
    eval_COCO(keypoints_ar, indices_ar)


def eval_COCO(keypoints_ar, indices_ar):
    pass

def extract_keypoints(heatmaps, pafs):
    pass

