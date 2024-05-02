from os import listdir,rename,remove,mkdir
from os.path import isfile, join,split
import matplotlib.pyplot as plt
import shutil
from random import sample
from tqdm import tqdm

import cv2
import numpy as np
import math

import json


def Copy_Files_From_To(path_from, path_to, if_in=None):

    if if_in != None :

        # Just the PURE NAME of the files to compare 
        files_to_compare = [f.split('.')[0] for f in listdir(if_in) if isfile(join(if_in, f))]
        # Iteration
        files_to_copy_from = iter([f for f in listdir(path_from) if isfile(join(path_from, f))])

        flag = True
        count = 0
        while flag:
            elem = next(files_to_copy_from,None)
            
            if elem is None :
                flag = False

            elif elem.split('.')[0] in files_to_compare:
                shutil.copyfile(join(path_from,elem),join(path_to,elem))
                count +=1

            else:
                pass

        return print(f'{count} elements were copied from:\n\n\t {path_from} \n to \n\t {path_to} \n If they were in \n\t{if_in}\n')
    else:

        files_to_copy_from = iter([f for f in listdir(path_from) if isfile(join(path_from, f))])

        flag = True
        count = 0
        while flag:
            elem = next(files_to_copy_from,None)
            
            if elem is None :
                flag = False

            else:
                shutil.copyfile(join(path_from,elem),join(path_to,elem))
                count +=1
        return print(f'{count} elements were copied from:\n\t {path_from} \n to \n\t {path_to}')
        
def Change_Names(dataset = "D:/rede1_dataset"):

    folders = listdir(dataset)
    assert "images" and "labels" in folders, "\nMissing either the folder 'images' or 'labels' !\n"

    images = [f for f in listdir(join(dataset,'images'))]
    labels = [f for f in listdir(join(dataset,'labels'))]
    images.sort(reverse = True)
    labels.sort(reverse = True)

    if ([f[:-4] for f in images] == [f[:-4] for f in labels]):
        for count in range(len(images)):
            elem = images[count]
            rename(join(join(dataset,'images'),elem), join(join(dataset,'images'),str(hex(count))+elem[-4:]))
            rename(join(join(dataset,'labels'),labels[count]), join(join(dataset,'labels'),str(hex(count))+'.txt'))
    else:
        raise ValueError("Elements in images and labels folder don't match !")

def Delete_If_Classes(classes , dataset = "D:/rede1_dataset"):
    folders = listdir(dataset)
    assert "images" and "labels" in folders, "\nMissing either the folder 'images' or 'labels' !\n"

    image_path = join(dataset,'images')
    labels_path = join(dataset,'labels')
    images = [f for f in listdir(image_path)]
    labels = [f for f in listdir(labels_path)]

    if ([f[:-4] for f in images] == [f[:-4] for f in labels]):
        for k in range(len(images)):
            flag = False
            f = open(join(labels_path,labels[k]), "r")
            print("*****************************************************************")
            print(labels[k])
            print("*****************************************************************\n")

            string = f.read()

            lines = string.split("\n")
            if len(lines[0]) == 0: 
                clas = [0]
            else:       
                clas = [int(line[0]) for line in lines]
            for c in classes:
                if c in clas:
                    flag = True
            f.close()

            if not flag: 
                remove(join(labels_path,labels[k]))
                remove(join(image_path,images[k]))

                print(f"delete files\n\t{join(labels_path,labels[k])}\n\t{join(image_path,images[k])}\n\n")   
        
    else:
        raise ValueError("Elements in images and labels folder don't match !")

def Delete_Labels_If_In(to_delete , dataset = "D:/rede1_dataset"):

    folders = listdir(dataset)
    assert "images" and "labels" and "to_delete" in folders, "\nMissing either the folder 'images' or 'labels' !\n"

    images_to_delete_path = join(dataset,to_delete)
    labels_path = join(dataset,'labels')

    images_to_delete = [f[:-4] for f in listdir(images_to_delete_path)]
    labels = [f for f in listdir(labels_path)]

    for k in images_to_delete:

        label_name = k + '.txt'
        if label_name in labels:
            remove(join(labels_path,label_name))

            print(f"delete file\n\t{label_name}\n in \t{join(labels_path,label_name)}\n\n")   

def Mix_Data(path, N = 3200, T_V_Te = (0.9,0.1,0.0), single_dir = False):

    folders = listdir(path)

    if not single_dir:
        assert "train" and "valid" and "test" in folders, "\nMissing either the folder 'train' or 'valid' or 'test' !\n"

        train_path = join(path,'train')
        assert "images" and "labels"  in listdir(train_path), "\nMissing either the folder 'images' or 'labels' !\n"
        valid_path = join(path,'valid')
        assert "images" and "labels"  in listdir(train_path), "\nMissing either the folder 'images' or 'labels' !\n"
        test_path = join(path,'test')
        assert "images" and "labels"  in listdir(train_path), "\nMissing either the folder 'images' or 'labels' !\n"

        train_images = [join(join(train_path,'images'),f) for f in listdir(join(train_path,'images'))]
        valid_images = [join(join(valid_path,'images'),f) for f in listdir(join(valid_path,'images'))]
        test_images = [join(join(test_path,'images'),f) for f in listdir(join(test_path,'images'))]
        images = train_images + valid_images + test_images

        train_labels = [join(join(train_path,'labels'),f) for f in listdir(join(train_path,'labels'))]
        valid_labels = [join(join(valid_path,'labels'),f) for f in listdir(join(valid_path,'labels'))]
        test_labels = [join(join(test_path,'labels'),f) for f in listdir(join(test_path,'labels'))]
        labels = valid_labels + train_labels + test_labels

        images.sort(reverse = True)
        labels.sort(reverse = True)
        
        sampled = sample(list(range(len(images))), N)
    else:
        assert "images" and "labels"  in listdir(path), "\nMissing either the folder 'images' or 'labels' !\n"
        images = [join(join(path,'images'),f) for f in listdir(join(path,'images'))]
        labels = [join(join(path,'labels'),f) for f in listdir(join(path,'labels'))]

        images.sort(reverse = True)
        labels.sort(reverse = True)
        
        sampled = sample(list(range(len(images))), N)

    if len(images) == len(labels):
        mkdir(join(path,'resampled'))

        New_Path_Valid = join(join(path,'resampled'),'valid')
        New_Path_Train = join(join(path,'resampled'),'train')
        New_Path_Test = join(join(path,'resampled'),'test')

        # " Train folder"
        New_Images_Train = join(New_Path_Train,'images')
        New_Labels_Train = join(New_Path_Train,'labels')
        # " Valid folder"
        New_Images_Valid = join(New_Path_Valid,'images')
        New_Labels_Valid = join(New_Path_Valid,'labels')
        # " Test folder"
        New_Images_Test = join(New_Path_Test,'images')
        New_Labels_Test = join(New_Path_Test,'labels')
        
        mkdir(New_Path_Valid)
        mkdir(New_Path_Train)
        mkdir(New_Path_Test)

        mkdir(New_Images_Train)
        mkdir(New_Labels_Train)
        mkdir(New_Images_Valid)
        mkdir(New_Labels_Valid)
        mkdir(New_Images_Test)
        mkdir(New_Labels_Test)

        print("\n******************************************************************")
        print(f"""\t\t Starting the transfer: \n\t Training folder with {100*(T_V_Te[0])}% of the {N} images 
                                            \n\t Validation folder with {100*(T_V_Te[1])}% of the {N} images
                                            \n\t Test folder with {100*(T_V_Te[2])}% of the {N} images
                                            \n\t Randomly chosen from: \t{path}""")
        print("******************************************************************\n")

        # Training
        for count in tqdm(range(N)):
            k = sampled[count]

            if count < int(N*T_V_Te[0]): 
                shutil.copyfile(images[k], join(New_Images_Train,images[k].split("\\")[-1]) )
                shutil.copyfile(labels[k], join(New_Labels_Train,labels[k].split("\\")[-1]) ) 

            elif count < int( N*(T_V_Te[0]+T_V_Te[1]) ):
                shutil.copyfile(images[k], join(New_Images_Valid,images[k].split("\\")[-1]) )
                shutil.copyfile(labels[k], join(New_Labels_Valid,labels[k].split("\\")[-1]) ) 
            
            else:
                shutil.copyfile(images[k], join(New_Images_Test, images[k].split("\\")[-1]) )
                shutil.copyfile(labels[k], join(New_Labels_Test,labels[k].split("\\")[-1]) ) 

        print(f"\nFrom {len(images)} images and labels, {int(N)} are in the new resampled folder !\n\n")
    else:
        raise ValueError("Images and labels don't match in size!")

def show(img):
    # Display the image
    cv2.imshow("Image", img)
    # Wait for the user to press a key
    cv2.waitKey(0)
    # Close all windows
    cv2.destroyAllWindows()

def Plate_Pre_Processing(path2image):

    img = cv2.imread(path2image, 0)

    # Just bleck and white seems much better than anything
    return img

    # # show(img)

    # # Image shaped based kernel
    # Kernel = math.floor(math.sqrt( (np.shape(img)[0]*np.shape(img)[1])/(1500) ))
    # if Kernel % 2 == 0:
    #     Kernel += 1

    # # computing the histogram  of the image 
    # hist = cv2.calcHist([img],[0],None,[256],[0,256]) 

    # values = np.array(hist).flatten()
    # bins = np.linspace(0,255,256).flatten()
    # Mean_Bin = np.sum(values*bins)/np.sum(values)

    # # TODO Am I calculating this right???
    # probs = np.array(values/np.sum(values)).flatten()
    # std = math.sqrt(np.sum(probs*(bins-Mean_Bin*np.ones(256))*(bins-Mean_Bin*np.ones(256))))

    # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # # Save image info for analysis after
    # image_info = {
    #     'Path' : path2image,
    #     'Height': np.shape(img)[0],
    #     'Width': np.shape(img)[1],
    #     'Mean' : Mean_Bin, 
    #     'Std' : std,
    #     'Otsu' : ret2 
    # }

    # head, tail = split(path2image)
    # try:
    #     mkdir(join(head,'Preprocessed'))
    # except:
    #     pass
    
    # new_path = join(head,'Preprocessed')
    # with open(join(new_path, tail[:-4]+'.json'), 'w') as fp:
    #     json.dump(image_info, fp)
    
    # # Image pre-processing pipeline
    # dilated_img = cv2.erode(img, np.ones((Kernel,Kernel), np.uint8))
    # bg_img = cv2.medianBlur(dilated_img, Kernel)
    # diff_img = 255 - cv2.absdiff(img, bg_img)
    # norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # return norm_img

def Preprocess_folder(path2images):

    try:
        mkdir(join(path2images,'Preprocessed'))
    except:
        pass
    
    new_path = join(path2images,'Preprocessed')
    images = [f for f in listdir(path2images)]

    print(f"\n\t\tStarting to preprocess {len(images)} images from folder {path2images}.\n\t\t Results in 'Preprocessed' folder inside directory!\n")

    for k in tqdm(range(len(images))):

        path2image = images[k]
        path2file = join(path2images,path2image)
        if isfile(path2file):
            preprocessed = Plate_Pre_Processing(path2file)
            # Save image in new folder 
            cv2.imwrite(join(new_path,path2image[:-4])+'.jpg', preprocessed)

def testing(path2images):

    images = [f for f in listdir(path2images) if not f.endswith(".json")]

    for k in images:
        path2image = join(path2images,k) 
        if isfile(path2image):
            
            img = cv2.imread(path2image, 0)

            # computing the histogram  of the image 
            hist = cv2.calcHist([img],[0],None,[256],[0,256]) 

            values = np.array(hist).flatten()
            bins = np.linspace(0,255,256).flatten()
            Mean_Bin = np.sum(values*bins)/np.sum(values)
            probs = np.array(values/np.sum(values)).flatten()
            std = math.sqrt(np.sum(probs*(bins-Mean_Bin*np.ones(256))*(bins-Mean_Bin*np.ones(256))))

            ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            # # If Otsu-Mean < sqrt(std) NAO PRECISA DE EQUALIZACAO?
            # if Mean_Bin < ret2 : 
            #     img = cv2.bitwise_not(img)
            #     print("INVERTEU")
            
            # Image shaped based kernel
            Kernel = math.floor(math.sqrt( (np.shape(img)[0]*np.shape(img)[1])/(1500) ))
            if Kernel % 2 == 0:
                Kernel += 1
            
            print(Kernel)

            print()
            print("Mean: " + str(Mean_Bin))
            print("Std: " + str(std))
            print("Otsu: " + str(ret2))
            print("Mean < Otsu (Need to invert) " + str(Mean_Bin <ret2))
            print("Otsu - Mean: " + str(ret2-Mean_Bin<math.sqrt(std)))
            print()

            equ = cv2.equalizeHist(img)

            # Image pre-processing pipeline
            dilated_img = cv2.erode(img, np.ones((Kernel,Kernel), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, Kernel)
            diff_img = 255 - cv2.absdiff(img, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            
            invert = 255 - norm_img
            invert_add = img + invert

            plt.subplot(231),plt.imshow(img, cmap = 'gray')
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(232),plt.imshow(norm_img, cmap = 'gray')
            plt.title('Preprocess original'), plt.xticks([]), plt.yticks([])
            plt.subplot(233),plt.imshow(equ, cmap = 'gray')
            plt.title('Equalized'), plt.xticks([]), plt.yticks([])
            plt.subplot(234),plt.plot(hist)
            plt.title('Histogram'), plt.xticks([]), plt.yticks([])
            plt.subplot(235),plt.plot(cv2.calcHist([equ],[0],None,[256],[0,256]) )
            plt.title('Histogram depois'), plt.xticks([]), plt.yticks([])
            plt.subplot(236),plt.imshow(invert_add, cmap = 'gray')
            plt.title('add preprocess with original'), plt.xticks([]), plt.yticks([])
            plt.show()

            # f = np.fft.fft2(img)
            # fshift = np.fft.fftshift(f)
            # magnitude_spectrum = 20*np.log(np.abs(fshift))
        
            # plt.subplot(121),plt.imshow(img, cmap = 'gray')
            # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
            # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            # plt.show()

def correct_labels(img_path, label_path):
    images = [f for f in listdir(img_path)]
    labels = [f for f in listdir(label_path)]
    
    images.sort(reverse = True)
    labels.sort(reverse = True)

    image_names = [i[:-4] for i in images]
    label_names = [i[:-4] for i in labels]

    # Missing labels (delete images)
    x = 0
    for i in tqdm(range(len(image_names))):
        if image_names[i] not in label_names:
            x +=1
            remove(join(img_path,images[i]))

    # Missing images (delete labels)
    y = 0
    for i in tqdm(range(len(label_names))):
        if label_names[i] not in image_names:
            y +=1
            remove(join(label_path,labels[i]))


    print(f"\nMissing labels: {x}")
    print(f"Missing images: {y}\n")


if __name__ == "__main__":
    # if_in = "D:/rede1_dataset/images"
    # path_from = "D:/Downloads/Artificial Mercosur License Plates (2)/Artificial Mercosur License Plates/labels"
    # path_to = "D:/rede1_dataset/labels"

    # Copy_Files_From_To(path_from, path_to, if_in)

    # Change_Names(dataset = "D:/rede1_dataset")
    # delete(classes = [3,4] , dataset = "D:/rede1_dataset")
    # Delete_Labels_If_In(to_delete = "D:/rede1_dataset/to_delete" , dataset = "D:/rede1_dataset")

    Mix_Data(path= r"D:\rede3_dataset\best_dataset3", N = 15987, T_V_Te = (0.8,0.2,0.0), single_dir = False)
    # correct_labels(r"D:\rede3_dataset\best_dataset3\train\images", r"D:\rede3_dataset\best_dataset3\train\labels")


    # testing(path2images = r"D:\rede3_dataset\exp1\train\images")
    # testing(path2images = r"C:\Users\jedua\OneDrive\Documents\Python Scripts\yolov7\test_images\Preprocessed")

    # Preprocess_folder(path2images = r"C:\Users\jedua\OneDrive\Documents\Python Scripts\yolov7\test_images")

    # Mix_Data(path = r"D:\rede3_dataset\Preprocessed", N = 533, T_V_Te = (0.8,0.2,0), single_dir = True)




























# # TODO check this after
# def preprocesing_shadow():

    # # If histogram shiffited to the right don't use it
    # colse = cv2.morphologyEx(norm_img, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    # show(colse)

    # rgb_planes = cv2.split(img)
    # show(np.reshape(rgb_planes, np.shape(img)))

    # result_planes = []
    # result_norm_planes = []

    # for plane in rgb_planes:
    #     show(plane)
    #     dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    #     bg_img = cv2.medianBlur(dilated_img, 21)
    #     diff_img = 255 - cv2.absdiff(plane, bg_img)
    #     norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    #     result_planes.append(diff_img)
    #     result_norm_planes.append(norm_img)
        
    # result = cv2.merge(result_planes)
    # result_norm = cv2.merge(result_norm_planes)

    # cv2.imwrite(join(path2image, 'shadows_out.png'), result)
    # cv2.imwrite(join(path2image, 'shadows_out_norm.png'), result_norm)
