# %%
import os   
import cv2   
import time 
import tqdm  
import torch
import codecs  
import shutil
import random
import pathlib 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
from scipy.signal import find_peaks 
import core.common as common
import core.ori_mAttention as mAttention
import core.involution as Involution2d

# %%
class Data_Read():
    def __init__(self):
        self.label = []
        self.Image_Path = []
        self.len = []

    def read_label(self,path):  
        with codecs.open(path,"r",encoding="utf-8") as f: 
            for line in f.readlines():  
                if line.strip(): self.label.append(eval(line)) 

    def read_image_paths(self,paths):
        data_root = pathlib.Path(paths) 
        num = len(list(data_root.glob('*/'))) 
        self.len.append(num)   
        for index in tqdm.tqdm(range(1,570),desc="Read"): 
            path = os.path.join(paths,"{}.jpg".format(index)) 
            if os.path.exists(path):  
                self.Image_Path.append(path)

    def Image_show(self,path,gray=False): 
        image = cv2.imread(path)          
        if not gray:
            plt.imshow(image[:,:,::-1]) 
            plt.xticks([]),plt.yticks([])
            plt.show()
        else:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
            plt.imshow(image,cmap = 'gray')
            plt.xticks([]),plt.yticks([])
            plt.show()

# %%
DA = Data_Read()
for index in range(1,29):
    path_label = r"..\Dataset\Label\No.str({}).txt".format(index) 
    path_img = r"..\Dataset\IMG\No.str({})".format(index)
    DA.read_label(path_label)
    DA.read_image_paths(path_img)


# %%
print(len(DA.label))
print(len(DA.Image_Path))

print(DA.label[:5])                       
print(DA.Image_Path[:5])

index = 10
DA.Image_show(DA.Image_Path[index])
print(DA.label[index])

# %%
print(DA.len)    
print(sum(DA.len))   
print(len(DA.len))   
print("......")
train_image_paths,train_image_labels,test_image_paths,test_image_labels = [],[],[],[]

total_paths = []
total_labels = []
for i in range(28):   
    if i == 0:
        total_paths.append(DA.Image_Path[:DA.len[i]]) 
        total_labels.append(DA.label[:DA.len[i]])
    else:
        total_paths.append(DA.Image_Path[sum(DA.len[:i]):sum(DA.len[:i+1])])
        total_labels.append(DA.label[sum(DA.len[:i]):sum(DA.len[:i+1])])

for i in range(len(total_paths)):
    train_image_paths.extend(total_paths[i][:-30])
    train_image_labels.extend(total_labels[i][:-30])
    test_image_paths.extend(total_paths[i][-30:])
    test_image_labels.extend(total_labels[i][-30:])
    
print(len(train_image_paths))
print(len(test_image_paths))

index = 103
DA.Image_show(test_image_paths[index])
print(test_image_labels[index])

# %%
class Data_object():
    def __init__(self,path,label):
        self.path = path
        self.label = label

train_ds = [Data_object(train_image_paths[i],train_image_labels[i]) for i in range(len(train_image_paths))]
test_ds = [Data_object(test_image_paths[i],test_image_labels[i]) for i in range(len(test_image_paths))]
val_ds = [Data_object(DA.Image_Path[i],DA.label[i]) for i in range(len(DA.Image_Path))]


for i in range(100):
    random.shuffle(train_ds)

print(len(train_ds))
print(len(test_ds))
print(len(val_ds))

index = 1800
print(train_ds[index].path,train_ds[index].label)
DA.Image_show(train_ds[index].path)

# %%
class Dataset(object):
    def __init__(self,train_ds,batch_size,device=None,shuffle=True):
        self.train_ds = train_ds
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.input_size = 64
        self.channel = 3
        self.num_LED = 16
        self.num_samples = len(self.train_ds)
        print("self.num_samples:",self.num_samples)     
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        print("self.num_batchs:",self.num_batchs)
        self.batch_count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self): 
        batch_image = np.zeros((self.batch_size,self.input_size,self.input_size,self.channel),dtype=np.float32)
        batch_label = np.zeros((self.batch_size,self.num_LED),dtype=np.float32)
        num = 0                                             #重新运行__next__,num归0
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:                    #用于采集一个batch的数据
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples:
                    index -= self.num_samples
                image = self.load_and_preprocess_image(self.train_ds[index].path,N=self.channel)    
                label = self.train_ds[index].label
                batch_image[num, :, :, :] = image
                batch_label[num, :] = label
                num += 1
            self.batch_count += 1
            batch_image, batch_label = torch.from_numpy(np.transpose(batch_image,(0,3,1,2))).float(), torch.from_numpy(batch_label).float()
            if self.device != None:
                batch_image, batch_label = batch_image.to(self.device), batch_label.to(self.device)
            return (batch_image, batch_label)
            
        else:
            self.batch_count = 0
            if self.shuffle:
                random.shuffle(train_ds)            #打乱顺序，停止迭代记忆
            raise StopIteration
    
    def Filter(self, xlist, ylist):
        xlist = [x for x in xlist if 20 < x < 600]
        ylist = [x for x in ylist if 20 < x < 450]
        return xlist, ylist
        
    def load_and_preprocess_image(self,path,N,max_threshold=220,radio=0.3):
        if not os.path.exists(path):
            raise ValueError(path) 
        image = cv2.imread(str(path))[20:,:,:]                              ### BGR
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
        ret, binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY |cv2.THRESH_TRIANGLE)
        index = np.linspace(ret,max_threshold,N).astype(np.int).tolist()
        
        imge_list = []
        for i in range(len(index)):
            if i == 0:
                imge_list.append(np.expand_dims(binary,axis=-1))
            else:
                _, binary = cv2.threshold(image,index[i],255,cv2.THRESH_BINARY)
                imge_list.append(np.expand_dims(binary,axis=-1))
        Sx, Sy = np.sum(binary, axis=0), np.sum(binary, axis=1)          #640,480
        peak_yid, peak_xid = find_peaks(Sy)[0], find_peaks(Sx)[0] 
        peak_xid, peak_yid = self.Filter(peak_xid, peak_yid)
        h, w = peak_yid[-1]-peak_yid[0], peak_xid[-1]-peak_xid[0]
        res = np.concatenate(imge_list,axis=-1)
        ymin = peak_yid[0]-int(radio*h) if (peak_yid[0]-int(radio*h)) >= 0 else 0 
        ymax = peak_yid[-1]+int(radio*h) if (peak_yid[-1]+int(radio*h)) <= res.shape[0]-1 else res.shape[0]-1
        xmin = peak_xid[0]-int(radio*w) if (peak_xid[0]-int(radio*w)) >= 0 else 0
        xmax = peak_xid[-1]+int(radio*w) if (peak_xid[-1]+int(radio*w)) <= res.shape[1]-1 else res.shape[1]-1
        res = res[ymin:ymax,xmin:xmax]
        res = cv2.resize(res,(self.input_size, self.input_size))
        res = res / 255
        return res

# %%
CUDA = True

device = torch.device("cuda:0" if CUDA and torch.cuda.is_available() else "cpu")
print("device:",device)
print()

train_Iterator = Dataset(train_ds, batch_size=32, device=device) 
print()   
val_Iterator = Dataset(val_ds, batch_size=32, device=device, shuffle=False)    #全部数据
print()
test_Iterator = Dataset(test_ds, batch_size=30, device=device, shuffle=False) 

# %%
if os.path.exists("..\mFIDA deep research\Check2\Image"):
    shutil.rmtree("..\mFIDA deep research\Check2\Image")    
    os.makedirs("..\mFIDA deep research\Check2\Image")
    os.remove("..\mFIDA deep research\Check2\label.txt")  


# %%
###数据集检查###
path = r"..\mFIDA deep research\Check2\label.txt"
index = 1

with open(path,"w",encoding="utf-8") as f:
    for num, (batch_image, batch_label) in enumerate(val_Iterator):
        for i in range(batch_image.shape[0]):
            img = np.transpose(batch_image[i].cpu().numpy(),(1,2,0))
            label = batch_label[i].cpu().numpy()
            img_path = "..\mFIDA deep research\Check2\Image\IMG_{}.jpg".format(index)
            img = cv2.resize(img[:,:,::-1]*255,(600, 600))
            cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            text = "Index_{}:\n".format(index) + "{}".format(label.reshape(4,4)) 
            f.write(text)
            f.write("\n")
            index += 1

# %%
class FcaNet(nn.Module):
    def __init__(self):
        super(FcaNet, self).__init__()
        self.CBS1 = common.Tconvolutional(filters_shape=(3, 3, 3, 32), downsample=True, activate_type="silu")
        self.CBS2 = common.Tconvolutional(filters_shape=(3, 3, 32, 16), downsample=True, activate_type="silu")
        self.Attention = mAttention.MultiSpectralAttentionLayer(16,16,16)
        self.CBS3 = common.Tconvolutional(filters_shape=(3, 3, 16, 8), downsample=True, activate_type="silu")
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)    
        self.dense = nn.Linear(512, 16)  

    def forward(self, x):
        x = self.CBS1(x)
        x = self.CBS2(x)
        x = self.Attention(x)
        x = self.CBS3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

CUDA = True
device = torch.device("cuda:0" if CUDA and torch.cuda.is_available() else "cpu")
print("device:",device)
print()
network = FcaNet().to(device)
summary(network, input_size=(3, 64, 64))


# %%
CUDA = False
device_test = torch.device("cuda:0" if CUDA and torch.cuda.is_available() else "cpu")
model = FcaNet().to(device_test)


X = torch.randn([1,3,64,64])
X = X.to(device_test)
N = 100
T = []

for i in range(N):
    start_time = time.time()
    out = model(X)
    end_time = time.time()
    T.append(round(1000*(end_time-start_time),3))
print("device_test:",device_test)
print("batch_size:",X.shape[0])
print("Model 耗时：{} ms".format(round(sum(T[10:])/len(T[10:]),3)))

# %%
opt = torch.optim.Adam(network.parameters())
loss_func = nn.BCEWithLogitsLoss()              #(output, target)


def train_step(images, labels):
    network.train()
    predictions = network(images)
    loss = loss_func(predictions,labels)
    opt.zero_grad()
    loss.backward(retain_graph=True)   
    opt.step()
    train_loss.append(loss.detach().cpu().numpy())
    


def test_step(images, labels, erro_logs=False):
    network.eval()
    with torch.no_grad():
        predictions = network(images)
    pred = torch.where(predictions > 0.5,1,0)
    acc, total_acc, erro = get_performance(images, labels.cpu().numpy().tolist(), pred.cpu().numpy().tolist(), erro_logs)
    return acc, total_acc, erro


def get_performance(images, label, pred, erro_logs):
    assert len(label) == len(pred) 
    acc_list = []
    total_acc = 0
    text = []
    for i in range(len(label)):
        acc = np.sum(np.where(np.array(label[i])==pred[i],1,0))/len(label[i])
        acc_list.append(acc)
        global erro_index
        erro_index += 1
        if acc == 1: 
            total_acc += 1
        else:
            erro_list.append(erro_index)
            if erro_logs:
                img = np.transpose(images[i].cpu().numpy(),(1,2,0))
                img_path = "Erro\Image\IMG_{}.jpg".format(erro_index)
                img = cv2.resize(img[:,:,::-1]*255,(600, 600))
                cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                L_list = list(map(lambda x: int(x), label[i]))
                message = "Index_{}:\n".format(erro_index) + "label: {}".format(L_list) + "\n" + "pred : {}".format(pred[i]) + "\n"
                text.append(message)
    if erro_logs:
        with open("..\mFIDA deep research\Erro\Erro_logs.txt","a",encoding="utf-8") as f:
            for line in text:
                f.write(line)
                f.write("\n")
    return sum(acc_list)/len(acc_list),total_acc/len(label), erro_list

# %%
# model_path = r"save_model\old-25.pth"
# mFida = torch.load(model_path)

epochs = 50
train_loss_result= []
max_total_acc = 0

global erro_index
erro_index = 0
global_steps = 0
warmup_steps = train_Iterator.num_batchs   #1个epoch预热
total_steps = train_Iterator.num_batchs * epochs
lr_init = 1e-3
lr_end = 1e-6
print(warmup_steps,total_steps)

for epoch in range(epochs):
    train_loss = []
    for num, (batch_image,batch_label) in enumerate(train_Iterator):
        global_steps += 1      
        if global_steps < warmup_steps:  
            lr = global_steps / warmup_steps * lr_init  
        else:
            lr = lr_end + 0.5 * (lr_init - lr_end) * (
                (1 + np.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        opt.param_groups[0]["lr"] = lr
        train_step(batch_image, batch_label)
    if epoch == 0: print("num:",num+1)

    acc_list, total_acc_list, erro_list = [], [], []
    for num, (batch_image,batch_label) in enumerate(test_Iterator):
        acc, total_acc, _ = test_step(batch_image, batch_label)
        acc_list.append(acc)
        total_acc_list.append(total_acc)
    print(".........")
    print ('Epoch {}, lr: {:.6f}, Loss: {}, ACC: {}, Total_ACC: {}'.format(epoch, lr, np.mean(train_loss),sum(acc_list)/len(acc_list),sum(total_acc_list)/len(total_acc_list)))
    train_loss_result.append(np.mean(train_loss))
    
    Total_ACC = sum(total_acc_list)/len(total_acc_list)
    if Total_ACC > max_total_acc:
        max_total_acc = Total_ACC
        save_path = os.path.join("..\mFIDA deep research\save_model","epoch-{}.pth".format(epoch))
        torch.save(network, save_path)
    
    if max_total_acc == 1.0:
        print("......Success !......")
        break
print(".........")

# %%
# if os.path.exists("..\mFIDA deep research\Erro\Image"): 
#     shutil.rmtree("..\mFIDA deep research\Erro\Image")    
#     os.makedirs("..\mFIDA deep research\Erro\Image")
#     os.remove("..\mFIDA deep research\Erro\Erro_logs.txt")


model_path = r"save_model\mFCA-34.pth"
mFida = torch.load(model_path)
acc_list, total_acc_list, erro_list = [], [], []

global erro_index
erro_index = 0

for num, (batch_image,batch_label) in enumerate(test_Iterator):
    acc, total_acc, erro = test_step(batch_image, batch_label, erro_logs=False)
    acc_list.append(acc)
    total_acc_list.append(total_acc)

print(num)
print(".........")
print ('ACC: {}, Total_ACC: {}'.format(sum(acc_list)/len(acc_list),sum(total_acc_list)/len(total_acc_list)))


# %%
print(np.mean(total_acc_list))
print("错误图片个数：",len(erro))

T = np.array(total_acc_list)
num = (1 - T)*32
np.sum(num)


print("ACC:",acc_list)
print("Img ACC:",total_acc_list)

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

# %%
distance_BER = []
distance_Img_Accuracy = []
for  i in range(10):
    distance_BER.append(np.mean(acc_list[3*i:3*(i+1)]))
    distance_Img_Accuracy.append(np.mean(total_acc_list[3*i:3*(i+1)]))

# %%
draw_result(distance_BER,distance_Img_Accuracy,save=False)

# %%
print("distance_BER:",distance_BER)
print("distance_Img_Accuracy:",distance_Img_Accuracy)


