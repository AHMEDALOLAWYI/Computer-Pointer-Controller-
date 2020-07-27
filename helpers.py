import numpy as np
import cv2


def handle_image(input_image, width=60, height=60):
    
    p_image = cv2.resize(input_image, (width, height))
    p_image = p_image.transpose((2,0,1))
    p_image = p_image.reshape(1, 3, height, width)

    return p_image


def get_eyes_crops(face_crop, r_eye, l_eye, relative_eye_size=0.20):

    w = face_crop.shape[1]
    h = face_crop.shape[0]

    x_r_eye = r_eye[0]*w
    y_r_eye = r_eye[1]*h
    x_l_eye = l_eye[0]*w
    y_l_eye = l_eye[1]*h

    relative_eye_size_x = w*relative_eye_size
    relative_eye_size_y = h*relative_eye_size

    r_eye_dimensions = [int(y_r_eye-relative_eye_size_y/2), int(y_r_eye+relative_eye_size_y/2),
    int(x_r_eye-relative_eye_size_x/2), int(x_r_eye+relative_eye_size_x/2)]

    l_eye_dimensions = [int(y_l_eye-relative_eye_size_y/2), int(y_l_eye+relative_eye_size_y/2),
    int(x_l_eye-relative_eye_size_x/2), int(x_l_eye+relative_eye_size_x/2)]

    r_eye_crop = face_crop[r_eye_dimensions[0]:r_eye_dimensions[1], 
                                r_eye_dimensions[2]:r_eye_dimensions[3]]

    l_eye_crop = face_crop[l_eye_dimensions[0]:l_eye_dimensions[1],
                                l_eye_dimensions[2]:l_eye_dimensions[3]]

    return r_eye_crop, l_eye_crop, r_eye_dimensions, l_eye_dimensions
