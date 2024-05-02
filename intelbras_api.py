# Image imports
import cv2

# Basic imports
import time
import numpy as np
from os import listdir,rename,remove,mkdir, getcwd
from os.path import isfile, join, split

# YOLOv7 imports 
from utils.torch_utils import select_device, time_synchronized

from ip_camera import VideoBuffer, IP_cam
from data_handler import reshape_numpy,img2torch,imgs2torch,set_folders,buffer2source,opencv_buffer
from detector import initialize_models,run_plate_detector

def infinite(buffer_seconds = 10,buffer_fps = 5, SHOW_TIME = False, DEBUG = False, save = True):
    # *********************************** LOADING MODELS ***********************************
    weights =  [r"C:\Users\jedua\OneDrive\Documents\Python Scripts\yolov7\rede2\best_rede2_tiny.pt", 
                r"C:\Users\jedua\OneDrive\Documents\Python Scripts\yolov7\rede3\rede3_hard_best_me.pt"]
    img_sizes = [640, 160]
    if save:
        save_dir = set_folders()
    else:
        save_dir = None
    device = select_device('0')
    half = device.type != 'cpu'

    t0 = time.time()
    redes, new_img_sizes, strides = initialize_models(weights, img_sizes, device)
    t1 = time_synchronized()
    if SHOW_TIME:
        print(f"Loading models: {(t1-t0)}\n")

    # *********************************** LOADING CAMERA ***********************************
    cam = IP_cam('jeduapf','j81720013','192.168.0.3','1313')

    # Initialize video buffer
    video = VideoBuffer(cam, buffer_seconds, buffer_fps)

    # Original image shape
    img0_shape = video.img_shape

    # *********************************** MODELS PARAMETERS ***********************************
    params = {
    "conf_thres": [0.4, 0.5],
    "iou_thres": [0.45, 0.45],
    "agnostic_nms": [False, False],
    "original_img_shape": img0_shape,
    "torch_img_shape": img_shape,
    "strides": strides,
    "new_img_sizes": new_img_sizes,
    "half": half,
    "device": device
    }

    text_params = {
                "font": cv2.FONT_HERSHEY_SIMPLEX, 
                # fontScale 
                "fontScale": 1,
                # Blue color in BGR 
                "color": (0, 0, 0), 
                # Line thickness of 2 px 
                "thickness":  2,
                }

    flag = True
    while flag:
        # Reset buffer
        buffer.clear() # Remove all elements for a new buffer
        buffer = video.get()

        # Reshape images for network
        t0 = time.time()
        torch_imgs, img_shape = buffer2source(buffer, new_img_sizes, strides, half, device)
        t1 = time_synchronized()
        if SHOW_TIME:
            print(f"Converting buffer imgs to torch: {(t1-t0)}\n\n")

        # Run model
        source = torch_imgs    
        results = run_plate_detector(buffer, redes, source, params, DEBUG, SAVE = save_dir)
        t2 = time_synchronized()
        if SHOW_TIME:
            print(f"Running model: {(t2-t1)}\n\n")

        # Show detections
        opencv_buffer(results, buffer, buffer_fps, text_params, save_dir)

    # *********************************** ANALYSING RESULTS ***********************************

    # result =[   [i, None ], ============================================================> Nothing detected
    #             [i, [x1,y1,x2,y2,car_or_moto,old_or_new]], ===============================================> Only plate detected
    #             [i, [x1,y1,x2,y2,car_or_moto,old_or_new], [[x1,y1,x2,y2,p,'a'],...,[x1,y1,x2,y2,p,'c']]] => Plate and chars detected
    #         ]

def dataset(dataset, buffer_fps = 5, SHOW_TIME = False, DEBUG = False, save = True):
    # *********************************** LOADING MODELS ***********************************
    weights =  [r"C:\Users\jedua\OneDrive\Documents\Python Scripts\DataOrganizer\rede2\best_rede2_tiny.pt", 
                r"C:\Users\jedua\OneDrive\Documents\Python Scripts\DataOrganizer\rede3\teste.pt"]
    img_sizes = [640, 160]
    if save:
        save_dir = set_folders()
    else:
        save_dir = None
    device = select_device('0')
    half = device.type != 'cpu'

    t0 = time.time()
    redes, new_img_sizes, strides = initialize_models(weights, img_sizes, device)
    t1 = time_synchronized()

    if SHOW_TIME:
        print(f"Loading models: {(t1-t0)}\n")

    # *********************************** ANALYSING RESULTS ***********************************

    # result =[   [i, None], ============================================================> Nothing detected
    #             [i, [x1,y1,x2,y2,p,0]], ===============================================> Only plate detected
    #             [i, [x1,y1,x2,y2,p,0], [[x1,y1,x2,y2,p,'a'],...,[x1,y1,x2,y2,p,'c']]] => Plate and chars detected
    #         ]

    text_params = {
                "font": cv2.FONT_HERSHEY_SIMPLEX, 
                # fontScale 
                "fontScale": 1,
                # Blue color in BGR 
                "color": (0, 0, 0), 
                # Line thickness of 2 px 
                "thickness":  2,
                }

    for filename in listdir(dataset):
 
        # check if the image ends with png or jpg or jpeg
        if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            img = cv2.imread(join(dataset,filename))
            buffer = [[filename,img]]

            # Reshape images for network
            t0 = time.time()
            torch_imgs, img_shape = buffer2source(buffer, new_img_sizes, strides, half, device)
            t1 = time_synchronized()

            params = {
            "conf_thres": [0.3, 0.3],
            "iou_thres": [0.35, 0.35],
            "agnostic_nms": [False, False],
            "original_img_shape": img.shape,
            "torch_img_shape": img_shape,
            "strides": strides,
            "new_img_sizes": new_img_sizes,
            "half": half,
            "device": device
            }

            if SHOW_TIME:
                print(f"Converting buffer imgs to torch: {(t1-t0)}\n\n")

            # Run model
            source = torch_imgs    
            results = run_plate_detector(buffer, redes, source, params, DEBUG, SAVE = save_dir)
            t2 = time_synchronized()

            if SHOW_TIME:
                print(f"Running model: {(t2-t1)}\n\n")

            # Show detections
            opencv_buffer(results, buffer, buffer_fps, text_params, save_dir, dataset = True)

def main():
    # infinite(buffer_seconds = 1,buffer_fps = 3, SHOW_TIME = False, DEBUG = False, save = True)
    dataset(r"D:\experimental\casa", buffer_fps = 1, SHOW_TIME = True, DEBUG = False, save = False)

if __name__ == "__main__":
    main()
