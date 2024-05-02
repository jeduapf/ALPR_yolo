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

# Torch 
import torch
import torch.backends.cudnn as cudnn

# YOLOv7 imports 
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path 
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from data_handler import preprocessing, set_folders


def initialize_model(weights, img_size, device):
    # Initialize
    set_logging()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    model = TracedModel(model, device, img_size)
    model.half()  # to FP16
        
    return model, imgsz, stride

# TODO: tentar unificar as duas funçoes de aplicar modelos?
def apply_model(model, source, params):

    device = params["device"]
    conf_thres = params["conf_thres"][0]
    iou_thres = params["iou_thres"][0]
    agnostic_nms = params["agnostic_nms"][0]
    torch_img_shape = params["torch_img_shape"]
    original_img_shape = params["original_img_shape"]

    results = []
    with torch.no_grad():
        # Run inference
        if device != 'cpu':
            for frame,img in source:
                pred = non_max_suppression(model(img)[0], conf_thres, iou_thres, agnostic=agnostic_nms)[0]

                if len(pred) == 0:
                    results.append([frame,[]])
                else:
                    #rescale coordiantes to original image shape
                    #                              ((height,width), (x1,y1,x2,y2), (height0,width0))
                    pred[:,:4] = scale_coords(torch_img_shape[1:], pred[:, :4], original_img_shape).round()
                    results.append([frame,pred.cpu().tolist()])

    return results 

def apply_chain_model(model, torch_img, names, device, conf_thres = 0.25, iou_thres = 0.45, agnostic_nms = False):

    with torch.no_grad():
        # Run inference
        if device != 'cpu':
            pred = non_max_suppression(model(torch_img)[0], conf_thres, iou_thres, agnostic=agnostic_nms)[0]
    return pred 

def second_predictions(cropped_img, redes, names, params, save):
     # crop and preprocess original img for next network format
    torch_img,_ = preprocessing(cropped_img,    params["new_img_sizes"][1], 
                                                params["device"], 
                                                params["half"], 
                                                s = params["strides"][0], 
                                                inter = cv2.INTER_LINEAR,
                                                save = save)
    
    # Apply last network and arrange data into desired results format
    pred = apply_chain_model(redes[1], torch_img, names,    params["device"],
                                                            conf_thres = params["conf_thres"][1], 
                                                            iou_thres = params["iou_thres"][1], 
                                                            agnostic_nms = params["agnostic_nms"][1])
    # rescale coordinates to original image
    pred[:,:4] = scale_coords(torch_img.shape[2:], pred[:, :4], cropped_img.shape).round()

    
    return pred.cpu().tolist()

def initialize_models(weights, img_sizes, device):

    # Loading model 2
    rede2, new_img_size2, stride2 = initialize_model(weights[0], img_sizes[0], device)
    # Loading model 3
    rede3, new_img_size3, stride3 = initialize_model(weights[1], img_sizes[1], device)

    return (rede2,rede3),(new_img_size2,new_img_size3),(stride2,stride3)

def plate_pipeline(cropped_img, pred, result, redes, params, x1,y1,x2,y2,det,names, DEBUG, PRINTS, SAVE, MIN_LEN):

    len_pred = len(pred)
    # Couldn't detect enough characters so just detected plate but not string
    if len_pred < MIN_LEN:
        if PRINTS:
            print(f'\n\tNot enough characters found\n\n') 
        res = [result[0],[x1,y1,x2,y2,round(det[4],3),-1,-1]] # return only plate position

    else:
        # Before angle, always sort by << x1 >>
        pred = sorted(pred,key=lambda l:l[0])
        np_pred = np.array(np.array(pred)[:,:4], dtype=np.float64)



        # CAR
        if car_moto(np_pred, MAGIC_CONST = 20, DEBUG = False):
            angle = check_angle(np_pred, MAX_DEGREE = 5)
            if angle is None:                                # Horizontal image 
                rotated = False
                res = predict_if_car(cropped_img, pred, np_pred, len_pred, 
                                    result, names, det, x1,y1,x2,y2, PRINTS, DEBUG, SAVE)  

            else:                                               # Make it Horizontal image
                rotated = True
                rotated_image = horizontalize(cropped_img, angle)

                # Predict again 
                pred = second_predictions(rotated_image, redes, names, params, SAVE)
                len_pred = len(pred)
                pred = sort_pred(pred, names, by = 'x1')
                np_pred = np.array(np.array(pred)[:,:4], dtype=np.float64)

                if DEBUG:
                    if PRINTS:
                        print(f"\n\tImage must be rotated with angle {angle}\n")
                        print(f"\n\tpred after rotation \n{pred}\n\n")    
                    if SAVE is not None:
                        cv2.imwrite(join(SAVE, 'rotated_img.png'), rotated_image) 

                if len_pred < 7 or check_angle(np_pred, MAX_DEGREE = 5) is not None:
                    res = [result[0],[x1,y1,x2,y2,round(det[4],3),0,-1]] # return only plate position
                else:
                    res = predict_if_car(rotated_image, pred, np_pred, len_pred, 
                                        result, names, det, x1,y1,x2,y2, PRINTS, DEBUG, SAVE)   




        # MOTO 
        else:     
            

            angle_top, angle_down, top, down,np_pred_top, np_pred_down = moto_angle(pred, np_pred, names)
            
            if PRINTS and DEBUG:
                print(f"\n\n\t ENTREI MOTO \n\n")
                print(f"\n top:\n{top} ")
                print(f"\n angle_top:\n{angle_top} ")
                print(f"\n Down:\n{down} ")
                print(f"\n angle_down:\n{angle_down} ")

            if (angle_top is None and angle_down is None):                               # Horizontal image 
                rotated = False
                res = predict_if_moto(cropped_img, names, top, down,np_pred_top, np_pred_down, 
                                                        result, x1,y1,x2,y2, det, SAVE, PRINTS, DEBUG)

            elif (angle_top is not None and angle_down is not None):

                if np.abs(angle_top - angle_down) > 5:
                    res = [result[0],[x1,y1,x2,y2,round(det[4],3),1,-1]] # return only plate position
                else:
                    angle = (angle_top + angle_down)/2
                    rotated = True
                    rotated_image = horizontalize(cropped_img, angle)

                    if DEBUG:
                        if PRINTS:
                            print(f"\n\tImage must be rotated with angle {angle}\n\n")   
                        if SAVE is not None:
                            cv2.imwrite(join(SAVE, 'rotated_img.png'), rotated_image) 

                    # Predict again 
                    pred = second_predictions(rotated_image, redes, names, params, SAVE)
                    len_pred = len(pred)
                    np_pred = np.array(np.array(pred)[:,:4], dtype=np.float64)
                    top, down = separate_top_down_chars(pred, np_pred)  

                    angle_top, angle_down, top, down,np_pred_top, np_pred_down = moto_angle(pred, np_pred, names)

                    if PRINTS and DEBUG:
                        print(f"\n\n\t RODEI img MOTO \n\n")
                        print(f"\n top:\n{top} ")
                        print(f"\n angle_top:\n{angle_top} ")
                        print(f"\n Down:\n{down} ")
                        print(f"\n angle_down:\n{angle_down} ")

                    if len_pred < 7 or angle_top is not None or angle_down is not None :
                        res = [result[0],[x1,y1,x2,y2,round(det[4],3),1,-1]] # return only plate position
                    else: 
                        res = predict_if_moto(rotated_image, names, top, down,np_pred_top, np_pred_down,
                                                     result, x1,y1,x2,y2, det, SAVE, PRINTS, DEBUG) 

            else:
                res = [result[0],[x1,y1,x2,y2,round(det[4],3),1,-1]] # return only plate position
                
    return res

def run_plate_detector(buffer, redes, source, params, DEBUG = False, PRINTS = True, SAVE = None, MIN_LEN = 5):

    # Apply first network
    results = apply_model(redes[0], source, params) 

    if DEBUG:
        print(f"\nresults after fist rede: {results}\n\n")   

    # Letters and numbers in plate
    names = redes[1].module.names if hasattr(redes[1], 'module') else redes[1].names

    # All detections of plates ( can be repeated frame )
    # update results per frame to detect the plate itselfs
    for i in range(len(results)):

        # result = [frame, tensor(n,6)] => n detections in the frame
        result = results.pop(0)

        # There are PLATE detections
        r = result[-1]
        if len(r) != 0:

            # Cropping the image and preprocessing for new prediction
            for j in range(len(r)):
                det = r[j]
                x1,y1,x2,y2 = int(det[0]),int(det[1]),int(det[2]),int(det[3])
                cropped_img = buffer[i][-1][y1:y2,x1:x2,:]

                # Already rescaled to cropped image values
                pred = second_predictions(cropped_img, redes, names, params, SAVE)

                if DEBUG:
                    print(f"\n\tpred:\n{pred}\n\n") 
                
                res = plate_pipeline(cropped_img, pred, result, redes, params, x1,y1,x2,y2,det,names, DEBUG, PRINTS, SAVE, MIN_LEN)
                results.append(res)   
                             
        else:
            results.append([result[0],None])
    #                                       nothing = -1 / car = 0 / moto = 1
    #                                       nothing = -1 / old = 0 / new = 1
    # result = [frame = buffer(i), [x1,y1,x2,y2,p,'car or moto','new or old'],     => ONE detection in the frame
    #               [[box,prob,letter][box,prob,letter],...,[box,prob,letter]]     
    #           ] 
    return results

def moto_angle(pred, np_pred, names):
    top, down = separate_top_down_chars(pred, np_pred)  

    np_pred_top = sort_pred(top, names, by = 'x1')
    np_pred_top = np.array(np.array(np_pred_top)[:,:4], dtype=np.float64)

    np_pred_down = sort_pred(down, names, by = 'x1')
    np_pred_down = np.array(np.array(np_pred_down)[:,:4], dtype=np.float64)    

    angle_top = check_angle(np_pred_top, MAX_DEGREE = 5)
    angle_down = check_angle(np_pred_down, MAX_DEGREE = 5)

    return angle_top, angle_down, top, down,np_pred_top, np_pred_down

def predict_if_moto(img, names, top, down, np_pred_top, np_pred_down, result, x1,y1,x2,y2, det, SAVE = False, PRINTS = False, DEBUG = False):
    
    print("\nENTREI predict_if_moto\n\n")
    if old_new(img, np_pred_top, True, DEBUG, save = True): # NEW model
        if PRINTS:
            print(f"\n\tIt's a MOTO NEW plate model\n\n") 
        if len(np_pred_top) + len(np_pred_down) < 7:
            if PRINTS:
                print(f'\n\tNot enough characters found\n\n') 
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),1,1]]
        else:
            top = sort_pred(top, names, by = 'x1')
            down = sort_pred(down, names, by = 'x1')
            pred = correct_characters_moto(top, down, np_pred_top, np_pred_down,True)
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),1,1], pred] 

    else: # OLD model
        if PRINTS:
            print(f"\n\tIt's a MOTO OLD plate model\n\n") 
        if len(np_pred_top) + len(np_pred_down) < 7 < 7:
            if PRINTS:
                print(f'\n\tNot enough characters found\n\n') 
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),1,0]] 
        else:
            top = sort_pred(top, names, by = 'x1')
            down = sort_pred(down, names, by = 'x1')
            pred = correct_characters_moto(top, down, np_pred_top, np_pred_down,False)
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),1,0], pred]        

    return res

def predict_if_car(img, pred, np_pred, len_pred, result, names, det, x1,y1,x2,y2, PRINTS, DEBUG, SAVE):

    if type(pred[0][-1]) is float:
        pred = sort_pred(pred, names, by = 'x1')
    if old_new(img, np_pred, True, DEBUG, save = True): # New plate format
        if PRINTS:
            print(f"\n\tIt's a CAR NEW plate model\n\n") 
        if len_pred < 7:
            if PRINTS:
                print(f'\n\tNot enough characters found\n\n') 
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),0,1]]
        else:
            pred = correct_characters(pred,np_pred,True)
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),0,1], pred] 
    else:                                                         # Old plate format
        if PRINTS:
            print(f"\n\tIt's a CAR OLD plate model\n\n") 
        if len_pred < 7:
            if PRINTS:
                print(f'\n\tNot enough characters found\n\n') 
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),0,0]] 
        else:
            pred = correct_characters(pred,np_pred,False)
            res = [result[0], [x1,y1,x2,y2,round(det[4],3),0,0], pred]        

    return res

def car_moto(np_pred, MAGIC_CONST = 20, DEBUG = False):

    # TODO Maybe getting min and max and doing the difference is faster and better??
    pred_std = np.std(np_pred, axis=0)
    if DEBUG:
    	print(f"\n\tpred_std: \n{pred_std}**************\n\n")

    if (pred_std[0]-pred_std[1] + pred_std[2]-pred_std[3] > MAGIC_CONST): # STD of X axis seems a bigger gap beetween veicules
        return True # Car
    else:
        return False # Moto

def old_new(img, np_pred, moto = False, DEBUG = True, save = True, MAGIC_CONST = 1.7):
    kernel = 2
    blue_points = 0
    pu = pixels_up(np_pred) 
    Y,X = img.shape[0],img.shape[1]

    if DEBUG:
        print(f"\n\tPixels Up: {pu}\n")
        print(img.shape)

    points = []
    for i in range(len(np_pred)):

        y = int(np_pred[i,1]  - pu)
        if y < 0:
        	y = 0
        x = int( (np_pred[i,0]+np_pred[i,2])/2)
        if y < kernel :
        	if X - x < kernel or x < kernel:
        		blue = int(img[y,x,0]) 
		        green = int(img[y,x,1])  
		        red = int(img[y,x,2])
        	else:
        		blue = int(np.mean(img[y,x-kernel:x+kernel,0])) 
		        green = int(np.mean(img[y,x-kernel:x+kernel,1]))  
		        red = int(np.mean(img[y,x-kernel:x+kernel,2])) 
        else:
        	if X - x < kernel or x < kernel:
        		blue = int(np.mean(img[y-kernel:y+kernel,x,0])) 
		        green = int(np.mean(img[y-kernel:y+kernel,x,1]))  
		        red = int(np.mean(img[y-kernel:y+kernel,x,2]))  
        	else:
        		blue = int(np.mean(img[y-kernel:y+kernel,x-kernel:x+kernel,0])) 
		        green = int(np.mean(img[y-kernel:y+kernel,x-kernel:x+kernel,1]))  
		        red = int(np.mean(img[y-kernel:y+kernel,x-kernel:x+kernel,2]))  
        
        points.append((x,y))

        # to avoid overflow use int type    
        if (blue > (green + red)/MAGIC_CONST):
            blue_points +=1

        if DEBUG:
            print(f"\n\tPoint: ({x},{y})\t Blue value: {blue}\t Green value: {green}\t Red value: {red}\n")

    if save and DEBUG:
        if save:
            save_dir = set_folders()
        else:
            save_dir = None
            
        print(f"\nsave: {save_dir}")
        radius = kernel+1
        color1 = (255,255,255)
        color2 = (0,0,0)
        thick = 1
        I = img.copy()
        for point in points:
            I = cv2.circle(I,point,radius,color1,thick)
            I = cv2.circle(I,point,radius+1,color2,thick)
        cv2.imwrite(join(save_dir, str(time.time())+'color_points.png'),I)


    if moto:
        if blue_points > 1:
            return True # NEW
        else:
            return False # OLD
    else:
        if blue_points > 3:
            return True # NEW
        else:
            return False # OLD

def correct_characters(pred,np_pred,new):
    # pred and np_pred MUST be already ordered by x1

    NEW = ('L','L','L','N','L','N','N')
    OLD = ('L','L','L','N','N','N','N')

    L = len(pred)
    missing, equal = similar_bounding_boxes_rows(np_pred, MIN_DIST = 5)

    if missing:
        return None
    else:
        if len(equal)>0 and L==7:
            # Not enough letters to predict plate since two are the same
            return None
        elif len(equal) == 0 and L ==7:
            if new:
                for typ in NEW:
                    ele = pred.pop(0)
                    pred.append(check_letter_number(ele, typ))
            else:
                for typ in OLD:
                    ele = pred.pop(0)
                    pred.append(check_letter_number(ele, typ))

            return pred
        else:
            # Here take the possible choices for the character and put in a list to choose while it was
            # detected that the next element in the predictions (ordered by x1) is in the same bb as the current
            # analysed choise in pred (pred[count])
            if new:
                pred = more_choices(pred, equal, NEW)
            else:
                pred = more_choices(pred, equal, OLD)

            # There is a detection that shuldn't exist somewhere
            if len(pred) != 7:
                print("Something is wrong, not or more than enough characters detected in correct_characters...")
                return None

            return pred

def correct_characters_moto(top, down, np_pred_top, np_pred_down,new):
    # pred and np_pred MUST be already ordered by x1

    TOP = ('L','L','L')
    NEW = ('N','L','N','N')
    OLD = ('N','N','N','N')

    L_top = len(top)
    L_down = len(down)
    missing_top, equal_top = similar_bounding_boxes_rows(np_pred_top, MIN_DIST = 5)
    missing_down, equal_down = similar_bounding_boxes_rows(np_pred_down, MIN_DIST = 5)

    if missing_top or missing_down:
        return None

    else:
        if (len(equal_top)>0 and L_top==3) or (len(equal_down)>0 and L_down==4):
            # Not enough letters to predict plate since two are the same
            return None
        elif (len(equal_top) == 0 and L_top == 3) and (len(equal_down) == 0 and L_down == 4):
            if new:
                for typ in TOP:
                    top.append(check_letter_number(top.pop(0), typ))
                for typ in NEW:
                    down.append(check_letter_number(down.pop(0), typ))
            else:
                for typ in OLD:
                    for typ in TOP:
                        top.append(check_letter_number(top.pop(0), typ))
                    for typ in NEW:
                        down.append(check_letter_number(down.pop(0), typ))

            pred = top + down
            return pred

        else:
            # Here take the possible choices for the character and put in a list to choose while it was
            # detected that the next element in the predictions (ordered by x1) is in the same bb as the current
            # analysed choise in pred (pred[count])
            if new:
                top = more_choices(top, equal_top, TOP)
                down = more_choices(down, equal_down, NEW)
            else:
                top = more_choices(top, equal_top, TOP)
                down = more_choices(down, equal_down, OLD)

            pred = top + down
            # There is a detection that shuldn't exist somewhere
            if len(pred) != 7:
                print("Something is wrong, not or more than enough characters detected in correct_characters...")
                return None

            return pred

def more_choices(pred, equal, TYPE):
    count = 0
    return_pred = []
    for typ in TYPE:
        to_choose = [pred[count]]
        while count + 1 in equal:
            count += 1
            to_choose.append(pred[count])

        count += 1
        if len(to_choose) == 1:
            return_pred.append(check_letter_number(to_choose[0], typ))
        else:
            return_pred.append(choose(typ, to_choose))
    return return_pred

def check_letter_number(element, char):
    letter = element[-1]
    if not letter.isnumeric() and char == 'L':
        return element
    elif letter.isnumeric() and char == 'N':
        return element
    elif letter.isnumeric() and char == 'L':
        element[-1] = confusing_table(letter, want_a_digit = False)
        return element
    elif not letter.isnumeric() and char == 'N':
        element[-1] = confusing_table(letter, want_a_digit = True)  
        return element
    else:
        raise ValueError(f"\nSomething is really wrong here...\t{element}\t{char}\n\n") 
        return None

def choose(char,lista):
    lista = sorted(lista,key=lambda l:l[4], reverse=True)
    best_option = lista[0]

    for i in range(len(lista)):
        letter = lista[i][-1]
        if not letter.isnumeric() and char == 'L':
            return lista[i]
        if letter.isnumeric() and char == 'N':
            return lista[i]

    letter = best_option[-1]
    if letter.isnumeric() and char == 'L':
        best_option[-1] = confusing_table(letter, want_a_digit = False)
    elif not letter.isnumeric() and char == 'N':
        best_option[-1] = confusing_table(letter, want_a_digit = True)  

    return best_option

def pixels_up(np_pred):
    # 0.35 = 23/65 the distance up divided by the standard heigth of a character according to regulation
    return int(0.3*np.abs(np.mean(np_pred[:,1] - np_pred[:,3])))

def separate_top_down_chars(pred, np_pred):

    # This is initially a simple approach using the mean  TODO
    # ( I guess using otsu would be better to separate the two inner means of the data)
    u1 = np.mean(np_pred[:,1])*np.ones(len(np_pred))
    u2 = np.mean(np_pred[:,3])*np.ones(len(np_pred))

    bool_array1 = np.array([u1 - np_pred[:,1] > 0]).squeeze()
    bool_array2 = np.array([u2 - np_pred[:,3] > 0]).squeeze()

    # print(f"\n\n bool_array1: {bool_array1}\n")
    # print(f"\n\n bool_array1: {bool_array2}\n")
    # print(f"\n\n bool_array1: {np.sum(bool_array2 ^ bool_array1)}\n")

    if np.sum(bool_array2 ^ bool_array1) == 0:
        top = [pred[i] for i in np.where(bool_array1)[0]]
        down = [pred[i] for i in np.where(np.invert(bool_array1))[0]]

        # print(f"\n\n top: {top}\n")
        # print(f"\n\n down: {down}\n")
        return sorted(top,key=lambda l:l[0]),sorted(down,key=lambda l:l[0])

    else:
        raise ValueError(f"\nBool arrays don't match, data seems uncnsistent for Y points\n\tmean1: {u1}\tmean2: {u2}\n\tbool_array1: {bool_array1} \tbool_array2: {bool_array2}\n\n")
        return None

def similar_bounding_boxes_rows(np_pred, MIN_DIST = 5):
    # np_pred MUST be already sorted by x1
    x1 = np_pred[:,0]
    sort_diff=np.diff(x1)
    test = np.where(sort_diff < MIN_DIST)[0]
    invert_test = np.where(sort_diff > MIN_DIST)[0]

    if len(test) > 0: # exist same boundingboxes
        if np.std(np_pred[invert_test]) > MIN_DIST: # Missing boundingboxes
            return False,test+1
        else:   
            return True,test+1

    else:
        if np.std(np_pred[invert_test]) > MIN_DIST: # Missing boundingboxes
            return False,[]
        else:
            return True,[]

def sort_pred(pred, names, by = 'x1'):
    if by == 'x1':
        s = 0
    elif by == 'y1':
        s = 1
    elif by == 'x2':
        s = 2
    elif by == 'y2':
        s = 3
    else: 
        raise ValueError(f"Can't sort by {by}...")

    # Sorting predictions by << s >> 
    pred = sorted(pred,key=lambda l:l[s])
    pred = [ [ele[0],ele[1],ele[2],ele[3],ele[4],names[int(ele[-1])]] for ele in pred]

    return pred

# TODO: still having some miss predictions... Maybe do another method
def check_angle(np_pred, MAX_DEGREE = 5):
    # np_pred must be ordered by x1

    diff_mean = np.mean(np.diff(np_pred, axis = 0), axis=0)
    angle = 180*np.arctan(diff_mean[1]/diff_mean[0])/np.pi

    if np.abs(angle) > MAX_DEGREE:
        return angle
    else:
        return None

def horizontalize(img, angle):

        rows = img.shape[0]
        cols = img.shape[1]
        #  ( x_center, y_center), angle, scale_factor  
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(114,114,114))

        return rotated_image

def confusing_table(s, want_a_digit = False):

    if not want_a_digit:
        if s == '0':
            return 'O'
        if s == '1':
            return 'I'
        if s == '2':
            return 'Z'
        if s == '3':
            return 'E'
        if s == '4':
            return 'A'
        if s == '5':
            return 'S'
        if s == '6':
            return 'G'
        if s == '7':
            return 'T'
        if s == '8':
            return 'B'
        if s == '9':
            return 'P'
        else:
            return '?'
    else:
        if s == 'A':
            return '4'
        if s == 'B':
            return '8'
        if s == 'D':
            return '0'
        if s == 'E':
            return '3'

        else:
            return '?'   # TODO: Finish this table


# Deprecated
# def similar_bounding_boxes_rows(np_pred, MIN_DIST = 20, DEBUG = True):

#     L = len(np_pred)
#     equal = []
#     compare_vect = MIN_DIST* np.ones((1,4))
#     for i in range(L):

#         count = 1
#         while i+count < L and np.sum(np.abs(np_pred[i,:]-np_pred[i+count,:])) < MIN_DIST:
#             count += 1

#         if count > 1:
#             equal.append((i,count-1))

#     if DEBUG:
#         print("\nInside similar bounding boxes")
#         print(f"\nnp_pred:\n{np_pred}\n")
#         print(f"\n\tequal: {equal}\n")

#     return equal