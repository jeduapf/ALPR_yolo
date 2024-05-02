# Torch 
import torch
import torch.backends.cudnn as cudnn

# Image imports
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go  
from skimage.transform import hough_line, hough_line_peaks    

# Basic imports
import threading  
import requests 
import time
import numpy as np
import json
from os import listdir,rename,remove,mkdir, getcwd
from os.path import isfile, join, split

# YOLOv7 imports 
from utils.datasets import letterbox

def reshape_numpy(img, imgsz, s = 32, inter = cv2.INTER_LINEAR):
    reshaped, ratio, (dw, dh) = letterbox(img,  new_shape=imgsz, 
                                            color=(114, 114, 114), 
                                            auto=True, 
                                            scaleFill=False, 
                                            scaleup=True, 
                                            stride=s, 
                                            interpolation=inter)

    # Convert
    img = reshaped[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img_shape = img.shape

    return img, img_shape

def img2torch(img, device, half):

    # Torch convert
    torch_img = torch.from_numpy(img).to(device)
    torch_img = torch_img.half() if half else torch_img.float()  # uint8 to fp16/32

    torch_img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if torch_img.ndimension() == 3:
        torch_img = torch_img.unsqueeze(0)

    return torch_img

def imgs2torch(img, imgsz, device, half, stride = 32, inter = cv2.INTER_LINEAR):

    assert img.dtype == 'uint8', "Images must be uint8 dtype elements !"

    img, img_shape = reshape_numpy(img, imgsz, s =stride, inter = cv2.INTER_LINEAR)
    torch_img = img2torch(img, device, half)

    return torch_img, img_shape


def preprocessing(img, imgsz, device, half, s = 32, inter = cv2.INTER_LINEAR, save = None):

    # Reshape to match Height = Weight
    img_reshaped, img_shape = reshape_numpy(img, imgsz, s, inter)

    # Preprocessing stage
    gray_image = cv2.cvtColor(img_reshaped.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)

    # Input images of the networ MUST be in (3,Height,Width), Height = Width
    # TODO: check if there is a better way of converting to grayscale
    in_img = np.stack((gray_image, gray_image,gray_image))
    
    if save is not None:
        cv2.imwrite(join(save, str(int(1000*time.time())) )+'.png',gray_image)
        # print(f"IMAGE {str(int(1000*time.time()))+'.png'} \n\tSHAPE: {in_img.shape}")
   
    torch_img = img2torch(in_img, device, half)

    return torch_img, img_shape

def single_img(imgsz, device,half,stride, path):
    img = cv2.imread(path)
    img0_shape = np.shape(img)
    torch_img,img_shape = imgs2torch(img, imgsz, device, half, stride = stride, inter = cv2.INTER_LINEAR)
    return img, torch_img, img0_shape, img_shape

def buffer2source(buffer, new_img_sizes, strides, half, device):

    # Reshape images to YOLO standars
    torch_imgs = []
    for timestamp,frame in buffer:
        torch_img, img_shape = imgs2torch(frame,    imgsz = new_img_sizes[0],
                                                    half = half, 
                                                    stride = strides[0],
                                                    device = device, 
                                                    inter = cv2.INTER_LINEAR)
        torch_imgs.append([timestamp,torch_img])   

    return torch_imgs, img_shape

def plotly_plot(results, buffer, frame):

    # Add plates string over the bounding box in the image
    for detec in results:
        if detec[0] == frame:
            fig = go.Figure()
        
            # Nothing detect in the frame
            if detec[-1] is None:
                # Add original image with some transparency
                fig.add_trace(go.Image(z=buffer[frame][:,:,::-1], opacity=1))
                fig.add_annotation(dict(font=dict(color='black',size=15),
                                                    x=0,
                                                    y=20,
                                                    showarrow=False,
                                                    text='<b>Nothing detected</b>',
                                                    textangle=0,
                                                    xanchor='left'))
                fig.add_annotation(dict(font=dict(color='white',size=15),
                                                    x=0,
                                                    y=buffer[frame].shape[0]-20,
                                                    showarrow=False,
                                                    text='<b>Nothing detected</b>',
                                                    textangle=0,
                                                    xanchor='left'))
                fig.show()
                return None

            # Just detected the plate not the number 
            elif len(detec) == 2:
                # Add original image with some transparency
                fig.add_trace(go.Image(z=buffer[frame][:,:,::-1], opacity=0.5))
                fig.add_annotation(dict(font=dict(color='black',size=15),
                                                    x=0,
                                                    y=buffer[frame].shape[0]-20,
                                                    showarrow=False,
                                                    text='<b>Impossible to read plate</b>',
                                                    textangle=0,
                                                    xanchor='left'))
                fig.add_annotation(dict(font=dict(color='white',size=15),
                                                    x=0,
                                                    y=20,
                                                    showarrow=False,
                                                    text='<b>Impossible to read plate</b>',
                                                    textangle=0,
                                                    xanchor='left'))

                fig.add_shape(  type="rect",
                    x0=detec[1][0], y0=detec[1][1], x1=detec[1][2], y1=detec[1][3],
                    line=dict(color="RoyalBlue"),
                )

            # Detected all  
            elif len(detec) == 3: 
                prob = round(10*(( 3*detec[1][4] ) + 7*np.mean(np.array([prob[-2] for prob in detec[-1]]))), 2) # Add prob of plate times mean of letters weighted
                placa = [ letter[-1] for letter in detec[-1]]
                placa = ''.join(placa)

                print(f'Buffer[{detec[0]}] '+placa+f' ({prob}%)')

                fig.add_shape(  type="rect",
                                x0=detec[1][0], y0=detec[1][1], x1=detec[1][2], y1=detec[1][3],
                                line=dict(color="RoyalBlue"),
                            )
                fig.add_annotation(dict(font=dict(color='black',size=12),
                                                    x=detec[1][0],
                                                    y=detec[1][1]-12,
                                                    showarrow=False,
                                                    text='<b>'+placa+f' ({prob}%)</b>',
                                                    textangle=0,
                                                    xanchor='left'))

            # Add original image with some transparency
            fig.add_trace(go.Image(z=buffer[frame][:,:,::-1], opacity=0.3))

        fig.show()
        
    else:
        print("\tNo such frame !") 

def opencv_buffer(results, buffer, buffer_fps, text_params, save_dir = None, dataset = False):

    # TODO: Lembrar que o mesmo frame pode aparecer varias vezes no results se tiver mais de uma placa detectada...
    for (result,frame) in zip(results,buffer):
        # frame[-1][:,:,::-1]
        image = frame[-1]
        window_name = "frame"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
        
        # No detections # TODO: before was result[-1] is it wrong now?
        if result is None:
            pass

        elif len(result) == 2:
            # TODO: nao parece unificado... o result nos results....
            if result[-1] is not None:
                x1,y1,x2,y2 = result[1][0],result[1][1],result[1][2],result[1][3]
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 4)

                # Using cv2.putText() method 
                image = cv2.putText(image, 'Plate', (x1, y1), text_params["font"],  
                                   text_params["fontScale"], text_params["color"], text_params["thickness"], cv2.LINE_AA)
            else:
                pass
                
        elif len(result) == 3 :
            # TODO: nao parece unificado... o result nos results....
            if result[-1] is not None:
                x1,y1,x2,y2 = result[1][0],result[1][1],result[1][2],result[1][3]
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 4)

                prob = round(10*(( 3*result[1][4] ) + 7*np.mean(np.array([prob[-2] for prob in result[-1]]))), 2) # Add prob of plate times mean of letters weighted
                placa = [ letter[-1] for letter in result[-1]]
                placa = ''.join(placa)

                if not dataset:
                    print(f'Buffer[{time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(result[0]))}]: '+placa+f' ({prob}%)')
                else:
                    print(f'Buffer[{result[0]}]: '+placa+f' ({prob}%)')
                # Using cv2.putText() method 
                image = cv2.putText(image, placa+f' ({prob}%)', (x1, y1-2), text_params["font"],  
                                   text_params["fontScale"], text_params["color"], text_params["thickness"], cv2.LINE_AA)

        else: 
            raise ValueError(f"Something went wrong with result in results list...\n\n\t{result}")

        # Displaying the image 
        cv2.resizeWindow(window_name, 1920,1080)

        # Save image and JSON
        if save_dir is not None and result is not None:# TODO: before was result[-1] is it wrong now?
            if result[-1] is not None:
                # result = [frame = buffer(i), [x1,y1,x2,y2,p,c], [[box,prob,letter][box,prob,letter],...,[box,prob,letter]]] => ONE detection in the frame
                if not dataset:
                    img_name = str(int(1000*frame[0])) + '.png'
                    img_name = join(save_dir,img_name)
                    save_detection_json(result2data(result, img_name), save_dir)
                else:
                    img_name = str(frame[0]) + '.png'
                    img_name = join(save_dir,img_name)
                    save_detection_json(result2data(result, img_name, True), save_dir)
                cv2.imwrite(img_name, image)

        cv2.imshow(window_name, image) 
        cv2.waitKey(int(1000/buffer_fps)) 

def result2data(result, img_name, dataset = False):
    #                                       nothing = -1 / car = 0 / moto = 1
    #                                       nothing = -1 / old = 0 / new = 1
    # result = [frame = buffer(i), [x1,y1,x2,y2,p,'car or moto','new or old'],     => ONE detection in the frame
    #               [[box,prob,letter][box,prob,letter],...,[box,prob,letter]]     
    #           ] 
    data = {}
    if len(result) == 2:
        data["timestamp"] = result[0]
        if not dataset:
            data["detected_time"] = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(result[0]))
        else:
            data["detected_time"] = result[0]
        data["img_path"] = img_name
        data["plate_coords"] = (result[1][0],result[1][1],result[1][2],result[1][3])
        data["plate_prob"] = round(result[1][4],4)
        if result[1][5] == 0:
            data["veicule"] = 'Car'
        elif result[1][5] == 1:
            data["veicule"] = 'Moto'
        elif result[1][5] == -1:
            data["veicule"] = 'Unknown'
        else: 
            raise ValueError(f"Value for veicule is not in the list... ({result[1][5]})")
        if result[1][6] == 0:
            data["plate_version"] = 'Old'
        elif result[1][6] == 1:
            data["plate_version"] = 'New'
        elif result[1][6] == -1:
            data["plate_version"] = 'Unknown'
        else: 
            raise ValueError(f"Value for plate type is not in the list... ({result[1][6]})")

        data["letters_prob"] = (0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        data["plate"] = ''

    elif len(result) == 3: 
        data["timestamp"] = result[0]
        if not dataset:
            data["detected_time"] = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(result[0]))
        else:
            data["detected_time"] = result[0]
        data["img_path"] = img_name
        data["plate_coords"] = (result[1][0],result[1][1],result[1][2],result[1][3])
        data["plate_prob"] = round(result[1][4],4)
        if result[1][5] == 0:
            data["veicule"] = 'Car'
        elif result[1][5] == 1:
            data["veicule"] = 'Moto'
        elif result[1][5] == -1:
            data["veicule"] = 'Unknown'
        else: 
            raise ValueError(f"Value for veicule is not in the list... ({result[1][5]})")
        if result[1][6] == 0:
            data["plate_version"] = 'Old'
        elif result[1][6] == 1:
            data["plate_version"] = 'New'
        elif result[1][6] == -1:
            data["plate_version"] = 'Unknown'
        else: 
            raise ValueError(f"Value for plate type is not in the list... ({result[1][6]})")
        data["letters_prob"] = [round(p[4],4) for p in result[2]]
        data["plate"] = ''.join([letter[-1] for letter in result[2]])
    else:
        raise ValueError("Data is supposed to have a detection but result length doesn't match...")

    return data

def set_folders():
    current_dir = getcwd()
    detections_dir = join(current_dir,"detections")
    try:
        mkdir(detections_dir)
    except:
        pass
        
    return detections_dir

def save_detection_json(data, detections_dir):
    # Append mode to add new line to detections file
    with open(join(detections_dir,'detections.json'), 'a', encoding='utf-8') as file:
        file.write('\n')
        json.dump(data, file, ensure_ascii=False, indent=4)