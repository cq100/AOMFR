# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import codecs
import pathlib
import os
import time
import tqdm
import random
from scipy.signal import find_peaks 

# %%
class Image_process(object):
    def read_image(self,path,gray=True):
        image = cv2.imread(path)   ### BGR
        if gray:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
        return image

    def Image_show(self,image,gray=False):
        if not gray:
            plt.imshow(image[:,:,::-1])
            plt.xticks([]),plt.yticks([])
            plt.show()
        else:
            plt.imshow(image,cmap = 'gray')
            plt.xticks([]),plt.yticks([])
            plt.show()
    
    def Image_binary(self,image,threshold=127):
        _, binary = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)
        return binary
    
    #THRESH_OTSU最适用于双波峰
    #THRESH_TRIANGLE最适用于单个波峰，最开始用于医学分割细胞等
    def Image_binary_OTSU_TRIANGLE(self,image,type="OTSU"):
        if type == "OTSU":
            ret, binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            print("OTSU ret:",ret)
        elif type == "TRIANGLE":
            ret, binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY |cv2.THRESH_TRIANGLE)
            print("TRIANGLE ret:",ret)
        else:
            raise ValueError("type must be 'OTSU' or 'TRIANGLE' !")
        return binary

    ###局部二值化
    def Image_local_binary(self,image,type = "mean"):
        if type == "mean":
            cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,10)
        elif type == "gauss":
            binary = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,10)
        else:
            raise ValueError("type must be 'mean' or 'gauss' !")
        return binary 

    def draw_point(self,key_point,image,save=False):
        kp = [cv2.KeyPoint(key_point[i][1],key_point[i][0],1) for i in range(len(key_point))]
        img = cv2.drawKeypoints(image, kp, None, color=(0,255,0))
        if save:
            cv2.imwrite('out1.jpg',img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        self.Image_show(img,gray=False)
        return img

    # bbox是bounding box的缩写
    def bbox_to_rect(self, bbox, color):
        # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
        return plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
            fill=False, edgecolor=color, linewidth=2)

    def draw_bbox(self,image, bbox, color="green"):
        # plt.figure(num=1,figsize=(10, 10))
        fig = plt.imshow(image[:,:,::-1])
        plt.xticks([]),plt.yticks([])
        fig.axes.add_patch(self.bbox_to_rect(bbox, color))
        plt.show()

    def scatter(self,point):
        plt.figure()
        plt.scatter(point[...,1],-point[...,0],s=20,c="#DC143C")
        plt.grid(b=True, ls=':')
        plt.suptitle("scatter", fontsize=12)
        plt.xticks([]),plt.yticks([])
        plt.show()

    def draw_result(self,acc_list,total_acc_list,save=False):
        epochs_range = np.arange(1,len(acc_list)+1,1)
        figure,ax=plt.subplots(dpi=600,num=1,figsize=(9, 3))
        plt.plot(epochs_range, np.array(acc_list),marker='o', linestyle='-',linewidth=1, markersize=2,label='Accuracy')
        plt.plot(epochs_range, np.array(total_acc_list),marker='*', linestyle='-',linewidth=1, markersize=2,label='Img_Accuracy')
        plt.xlabel('Block')
        plt.ylabel('Accuracy')
        plt.xticks(epochs_range)
        plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        # plt.tick_params(labelsize=8)
        plt.legend(loc='upper right')
        if save:  plt.savefig('Result.jpg',bbox_inches="tight",dpi=600)
        plt.show()

    def Connected_Component(self,binary,ori_img,debug=False):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=None)
        center = centroids.tolist()
        del_index = []
        num = []
        for i in range(stats.shape[0]):
            if stats[i][0] == 0:
                del_index.append(i)
            else:
                num.append(stats[i][4])
        for j in range(stats.shape[0]):
            if stats[j][4] <= 0.5416*sum(num)/len(num):   #按照簇内点的数目进行过滤0.328
                del_index.append(j)

        bright = []
        for i in range(len(center)):
            bright.append(ori_img[int(center[i][1])][int(center[i][0])])
        bright_threshold = 0.62*sum(bright)/len(bright)

        new_p = []
        for i in range(len(center)):
            if len(center) - len(del_index) >= 4:
                if i not in del_index and ori_img[int(center[i][1])][int(center[i][0])] > bright_threshold:
                    new_p.append([int(center[i][1]),int(center[i][0])])  #(y,x)
            else:
                new_p.append([int(center[i][1]),int(center[i][0])])  #(y,x)
        if debug:
            print("center:",center)
            print("stats:",stats)
            print("Threshold1:",0.5416*sum(num)/len(num))
            self.ori_center = center
            self.binary = binary
            print("del_index:",del_index)
            print("bright",bright)
            print("bright_threshold:",bright_threshold)
            self.centers = new_p
        return new_p

    def out01_data(self,centers):
        x_axis,y_axis = [],[]
        for i in range(len(centers)):
            x_axis.append(centers[i][1])
            y_axis.append(centers[i][0])
        xmin,xmax,ymin,ymax = min(x_axis),max(x_axis),min(y_axis),max(y_axis)
        xd = np.linspace(xmin,xmax+0.000001,num=5)
        yd = np.linspace(ymin,ymax+0.000001,num=5)
        res = np.zeros((4,4))
        for kt in centers:
            for i in range(4):
                for j in range(4):
                    if xd[i] <= kt[1] < xd[i+1] and yd[j] <= kt[0] < yd[j+1]:
                        res[j][i] = 1
        return res

    def predict(self,path,show=True,debug=False):
        img = self.read_image(path,gray=True)
        if show:
            self.Image_show(img,gray=True)
        binary = self.Image_binary(img,threshold=127)
        if show:
            self.Image_show(binary,gray=True)
        if show: print("连通域标记：")
        centers = self.Connected_Component(binary,img,debug=debug)
        if show: print("Centers:", len(centers))
        return self.out01_data(centers)

    def Canny_predict(self,path,show=True,debug=False):
        img = self.read_image(path,gray=True)
        if show:
            self.Image_show(img,gray=True)
        binary = self.Image_binary(img,threshold=127)
        if show:
            self.Image_show(binary,gray=True)
        edge = cv2.Canny(binary, 120, 255, 3)
        if show:
            self.Image_show(edge,gray=True)
        points = np.argwhere(edge != 0)
        ymax,xmax = points.max(axis=0)
        ymin,xmin = points.min(axis=0)
        if show:
            ori_img = self.read_image(path,gray=False)
            Rbox = [xmin,ymin,xmax,ymax]
            self.draw_bbox(ori_img,Rbox)
        
        xd = np.linspace(xmin,xmax+0.000001,num=5)
        yd = np.linspace(ymin,ymax+0.000001,num=5)
        res = np.zeros((4,4))
        for kt in points:
            for i in range(4):
                for j in range(4):
                    if xd[i] <= kt[1] < xd[i+1] and yd[j] <= kt[0] < yd[j+1]:
                        res[j][i] = 1
        return res

    def Fast_predict(self,path,show=True,debug=False):
        img = self.read_image(path,gray=True)
        binary = self.Image_binary(img,threshold=127)
        fast = cv2.FastFeatureDetector_create(threshold=30,nonmaxSuppression=False,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        kp = fast.detect(binary,None)#描述符

        if show:
            self.Image_show(img,gray=True)
            self.Image_show(binary,gray=True)
            img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0))  
            self.Image_show(img2)
        
        points = []
        for i in range(len(kp)):
            points.append([int(kp[i].pt[1]),int(kp[i].pt[0])])
        points = np.array(points)
        ymax,xmax = points.max(axis=0)
        ymin,xmin = points.min(axis=0)
        if show:
            ori_img = self.read_image(path,gray=False)
            Rbox = [xmin,ymin,xmax,ymax]
            self.draw_bbox(ori_img,Rbox)

        xd = np.linspace(xmin,xmax+0.000001,num=5)
        yd = np.linspace(ymin,ymax+0.000001,num=5)
        res = np.zeros((4,4))
        for kt in points:
            for i in range(4):
                for j in range(4):
                    if xd[i] <= kt[1] < xd[i+1] and yd[j] <= kt[0] < yd[j+1]:
                        res[j][i] = 1
        return res

    def Structuring_predict(self,path,show=True,debug=False):
        img = self.read_image(path,gray=True)
        binary = self.Image_binary(img,threshold=105)
        if show:
            self.Image_show(img,gray=True)
            self.Image_show(binary,gray=True)
        Sy, Sx = np.sum(binary, axis=0), np.sum(binary, axis=1)          
        peak_yid, peak_xid = find_peaks(Sy)[0], find_peaks(Sx)[0]    
        res = np.zeros((4,4))
        if peak_xid !=[] and peak_yid !=[]:
            y = np.round(np.linspace(peak_yid[0], peak_yid[-1], 4)).astype(int) 
            x = np.round(np.linspace(peak_xid[0], peak_xid[-1], 4)).astype(int)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
            dibinary = cv2.dilate(binary, element)
            for i in range(4):
                for j in range(4):
                    ii = x[i]
                    jj = y[j]
                    if dibinary[ii][jj] == 255:
                        res[i][j] = 1
        return res

# %%
class Evaluation_Norm(Image_process):
    def read_label(self,path):
        label = []
        with codecs.open(path,"r",encoding="utf-8") as f:
            for line in f.readlines():
                if line.strip(): label.append(eval(line))
        return label

    def evaluation(self,paths,type="Structuring"):
        pred = []
        data_root = pathlib.Path(paths)
        num = len(list(data_root.glob('*/')))
        for index in tqdm.tqdm(range(1,101),desc="Evaluation"): 
            path = os.path.join(paths,"{}.jpg".format(index))
            if os.path.exists(path):
                if type == "canny":
                    res = self.Canny_predict(path,show=False)
                elif type == "Connected":
                    res = self.predict(path,show=False)
                elif type == "fast":
                    res = self.Fast_predict(path,show=False)
                elif type == "Structuring":
                    res = self.Structuring_predict(path,show=False)
                else:
                    raise ValueError("type must be 'canny' or 'Connected' or 'fast' or 'Structuring' !") 
                res = res.reshape(16,)
                pred.append(res)
        return pred

    ###随机采样###
    def sample(self,N,label,pred):  
        assert len(label) == len(pred)                          
        seed_array = np.arange(len(label))
        print("......Sample......")
        for i in range(N):
            index = random.choice(seed_array)
            print("label:",label[index])
            print("pred :",pred[index])
            print(".........")
        print("......END......")
    
    def get_performance(self,label,pred,erro_name=None,erro_logs=True):
        assert len(label) == len(pred) 
        acc_list = []
        total_acc = 0
        erro = []
        for i in range(len(label)):
            acc = np.sum(np.where(np.array(label[i])==pred[i],1,0))/len(label[i])
            acc_list.append(acc)
            if acc == 1: 
                total_acc += 1
            else:
                if erro_logs:
                    p_list = pred[i].tolist()
                    p_list = list(map(lambda x: int(x),p_list))
                    erro.append([i+1,label[i],p_list])
        if erro_logs:
            if len(erro) > 0:
                with codecs.open(erro_name,"w",encoding="utf-8") as f:
                    for j in range(len(erro)):
                        message1 = "index:" + str(erro[j][0]) + "\n"
                        message2 = "label:" + str(erro[j][1]) + "\n"
                        message3 = "pred :" + str(erro[j][2]) + "\n"
                        f.write(message1)
                        f.write(message2)
                        f.write(message3)
                        f.write("\n")
        return sum(acc_list)/len(acc_list),total_acc/len(label)

# %%
EN = Evaluation_Norm()

# %%
path = r"IMG\2.jpg"
EN.Structuring_predict(path)

# %%
# for index in range(1,4):
#     path_label = r"Label\No.str({}).txt".format(index)
#     path_img = r"IMG\No.str({})".format(index)
#     label = EN.read_label(path_label)
#     pred = EN.evaluation(path_img,type="Structuring")
#     acc, total_acc = EN.get_performance(label,pred,erro_name="Erro_logs2\Erro_{}.txt".format(index))
#     print("index：",path_img)
#     print("acc:",acc)
#     print("total_acc:",total_acc)
#     print("......")

# %%
acc_list = []
total_acc_list = []
for index in range(1,29):
    path_label = r"..\Dataset\Label\No.str({}).txt".format(index)
    path_img = r"..\Dataset\IMG\No.str({})".format(index)
    label = EN.read_label(path_label)
    pred = EN.evaluation(path_img,type="Structuring")
    acc, total_acc = EN.get_performance(label,pred,erro_name="Erro_logs2\Erro_{}.txt".format(index),erro_logs=False)
    acc_list.append(acc)
    total_acc_list.append(total_acc)
    # print("index：",path_img)
    # print("acc:",acc)
    # print("total_acc:",total_acc)
    # print("......")

# %%
EN.draw_result(acc_list,total_acc_list,save=True)

# %%
def draw_result(acc_list,total_acc_list,save=False):
    epochs_range = np.arange(1,len(acc_list)+1,1)
    figure,ax=plt.subplots(dpi=600,num=1,figsize=(9, 3))
    plt.plot(epochs_range, np.array(acc_list),marker='o', linestyle='-',linewidth=1, markersize=2,label='BER')
    plt.plot(epochs_range, np.array(total_acc_list),marker='*', linestyle='-',linewidth=1, markersize=2,label='Img_Accuracy')
    plt.xlabel('Distance(m)')
    plt.ylabel('Accuracy')
    plt.xticks(epochs_range)
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    # plt.tick_params(labelsize=8)
    plt.legend(loc='lower left')
    if save:  plt.savefig('Result.jpg',bbox_inches="tight",dpi=600)
    plt.show()

distance_BER = []
distance_Img_Accuracy = []
for  i in range(10):
    distance_BER.append(np.mean(acc_list[3*i:3*(i+1)]))
    distance_Img_Accuracy.append(np.mean(total_acc_list[3*i:3*(i+1)]))

draw_result(distance_BER,distance_Img_Accuracy,save=False)

print("distance_BER:",distance_BER)
print("distance_Img_Accuracy:",distance_Img_Accuracy)

# %%
print("ACC:",acc_list)
print("Img ACC:",total_acc_list)

# %%
path = r"..\Dataset\IMG\No.str(5)\6.jpg"

N = 100
T = []
for i in range(N):
    start_time = time.time()
    _ = EN.Structuring_predict(path, show=False, debug=False)
    end_time = time.time()
    T.append(round(1000*(end_time-start_time),3))
print("方向投影耗时：{} ms".format(round(sum(T[10:])/len(T[10:]),3)))



