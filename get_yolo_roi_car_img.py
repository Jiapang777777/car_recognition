import os
from turtle import circle
import cv2
import re
import numpy as np 
import string
import random

def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters) for _ in range(length))
    return random_string

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath) 
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):   
            if temp.endswith(".jpg"):          
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

if __name__ =="__main__":
    file_list=[]
    # root_path=r"train\dest"  # 图片和标注路径
    root_path=r"train_data/car/11-17-2"  # 图片和标注路径
    save_jpg_folder =r"train_data/plate/11-17-2"
    count=0
    allFilePath(root_path,file_list)
    for pic in file_list:    
        try:
            count+=1   
            print(count,pic)       
            txt_path= pic.replace(".jpg",".txt")
            img = cv_imread(pic)
            pic_name = os.path.basename(pic)
            h,w,c = img.shape 
            with open(txt_path,"r") as f:
                lines = f.readlines() 
                for line in lines:
                    line =line.strip('\n')
                    line_list = re.split(r'\s+',line)
                    label_=int(line_list[0])
                    xywh=line_list[1:5]
                    xywh = [float(x) for x in xywh]
                    # landmarks =line_list[5:]
                    # landmarks=[float(x) for x in landmarks]

                    rect_w=xywh[2]*w
                    rect_h=xywh[3]*h

                    rect_x=xywh[0]*w-rect_w/2
                    rect_y=xywh[1]*h-rect_h/2
                    
                    # for i in  range(4):
                    #     pts_x=landmarks[2*i]*w
                    #     pts_y=landmarks[2*i+1]*h
                        # cv2.circle(img,(int(pts_x),int(pts_y)),4,colors[i],3,-1)
                    # cv2.rectangle(img,(int(rect_x),int(rect_y)),(int(rect_x+rect_w),int(rect_y+rect_h)),colors[label_],2)
                    roi_img = img[int(rect_y):int(rect_y+rect_h),int(rect_x):int(rect_x+rect_w)]
                    new_pic_name = generate_random_string(20)+".jpg"
                    save_jpg_path= os.path.join(save_jpg_folder,new_pic_name)
                    cv2.imwrite(save_jpg_path,roi_img)
        except:
            print("error",pic)
            # continue
                # print(xywh,landmarks)
        # cv2.namedWindow("haha",cv2.WINDOW_NORMAL)
        # cv2.imshow("haha",roi_img)
        # cv2.waitKey(0)
        # # cv2.imwrite("result.jpg",img)  
        