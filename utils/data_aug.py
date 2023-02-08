import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance


def data_augmentation(image, boxes, use_bb=False):

    # NumPy to PIL
    image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
    w, h = image.size
    some_box = len(boxes) > 0
    # if some_box:
    #     boxes = boxes[~np.all(boxes == 0, axis=1)]

    # Resize
    if rand() > 0.5:
        jitter = 0.1 if rand() > 0.5 else 0
        min_scale_factor = 0.90 if rand() > 0.5 else 1
        max_scale_factor = 1.10 if rand() > 0.5 else 1
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(min_scale_factor, max_scale_factor)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (0, 0, 0))
        new_image.paste(image, (dx, dy))
        image = new_image
        if some_box:
            if use_bb:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / w + dx
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / h + dy
            else:
                boxes[:, 0] = boxes[:, 0] * nw / w + dx
                boxes[:, 1] = boxes[:, 1] * nh / h + dy

    # Horizontal Flip
    if rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if some_box:
            if use_bb:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            else:
                boxes[:, 0] = w - boxes[:, 0]

    # Vertical Flip
    if rand() > 1:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if some_box:
            if use_bb:
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            else:
                boxes[:, 1] = h - boxes[:, 1]

    # Rotate
    if rand() > 0.5 and not use_bb:
        angle = rand(0, 30)
        image = image.rotate(angle, Image.NEAREST, expand=False)
        if some_box:
            boxes = rotate((w // 2, h // 2), boxes, math.radians(-angle))

    # Brightness change
    if rand() > 1:
        factor = rand(0.5, 1.5)
        image = ImageEnhance.Brightness(image).enhance(factor)

    # Contrast change
    if rand() > 1:
        factor = rand(0.5, 1.5)
        image = ImageEnhance.Contrast(image).enhance(factor)

    # PIL to NumPy
    image = np.array(image) / 255

    # Horizontal translation
    if rand() > 0.5:
        shift_x = 0.2
        shift_x = rand(-shift_x, shift_x) * w
        m = np.array([[1, 0, shift_x], [0, 1, 0]])
        image = cv2.warpAffine(image, m, (w, h))
        if some_box:
            boxes[:, 0] = boxes[:, 0] + shift_x

    # Vertical translation
    if rand() > 0.5:
        shift_y = 0.2
        shift_y = rand(-shift_y, shift_y) * h
        m = np.array([[1, 0, 0], [0, 1, shift_y]])
        image = cv2.warpAffine(image, m, (w, h))
        if some_box:
            boxes[:, 1] = boxes[:, 1] + shift_y

    # Shear
    if rand() > 1:  # Deactivated
        shear_factor = 0.2
        shear_factor = rand(-shear_factor, shear_factor)
        m = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
        nw = w + abs(shear_factor * h)
        image = cv2.warpAffine(image, m, (int(nw), h))
        image = cv2.resize(image, (w, h))
        scale_factor_x = nw / w
        if some_box:
            if use_bb:
                boxes[:, [0, 2]] += ((boxes[:, [1, 3]]) * abs(shear_factor)).astype(int)
                boxes[:, :4] = boxes[:, :4] / [scale_factor_x, 1, scale_factor_x, 1]
            else:
                boxes[:, 0] = (boxes[:, 0] + abs(shear_factor) * boxes[:, 1]) / scale_factor_x

    # Amend boxes
    if some_box:
        if use_bb:
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0  # Fit boxes into the image
            boxes[:, 2][boxes[:, 2] > w] = w
            boxes[:, 3][boxes[:, 3] > h] = h
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]  # Discard invalid box
            min_box_area = 200
            boxes = boxes[bbox_area(boxes) > min_box_area]
        else:
            boxes = boxes[0 < boxes[:, 0]]  # Remove boxes out of image
            boxes = boxes[boxes[:, 0] < w]
            boxes = boxes[0 < boxes[:, 1]]
            boxes = boxes[boxes[:, 1] < h]
    return image, boxes


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def rotate(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    q = np.zeros(points.shape)
    for i, point in enumerate(points):
        ox, oy = origin
        if len(point) == 2:
            px, py = point
        else:
            px, py, label = point
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        if len(point) == 2:
            q[i] = [qx, qy]
        else:
            q[i] = [qx, qy, label]
    return q


def rand(a=0., b=1.):
    return np.random.rand()*(b-a) + a
