#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
import torch
import torch.nn as nn


###1、BN层###
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

###Mish激活函数###
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

###2、卷积层###
###默认：CBL，不进行下采样，无指定名称###
def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky',kernel_regularizer1=tf.keras.regularizers.l2(0.0005),kernel_initializer1=tf.random_normal_initializer(stddev=0.01),name=None):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    if name != None:
        if name[0]==0:
            c_name = 'D_conv2d_%d' %name[1] if name[1] > 0 else 'D_conv2d'
            b_name = 'D_batch_normalization_%d' %name[2] if name[2] > 0 else 'D_batch_normalization'
        elif name[0]==1:
            c_name = 'G_conv2d_%d' %name[1] if name[1] > 0 else 'G_conv2d'
            b_name = 'G_batch_normalization_%d' %name[2] if name[2] > 0 else 'G_batch_normalization'
            
        conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                    use_bias=not bn, kernel_regularizer=kernel_regularizer1,
                                    kernel_initializer=kernel_initializer1,
                                    bias_initializer=tf.constant_initializer(0.),name=c_name)(input_layer)
        if bn: conv = BatchNormalization(name=b_name)(conv)
    else:
        conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                    use_bias=not bn, kernel_regularizer=kernel_regularizer1,
                                    kernel_initializer=kernel_initializer1,
                                    bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn: conv = BatchNormalization()(conv)
        
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
        elif activate_type == "SiLU":
            conv = tf.nn.silu(conv)
    return conv


###3、YOLO V5 Focus结构
def Focus(input_layer,filters_shape,name=None):
    input_layer = tf.concat([input_layer[:, ::2, ::2, :], input_layer[:, 1::2, ::2, :],input_layer[:, ::2, 1::2, :],input_layer[:, 1::2, 1::2, :]], axis=-1)
    return convolutional(input_layer, filters_shape,activate_type='SiLU',name=name)


###4、Res残差结构###
def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky',cut=True,rname=None):
    short_cut = input_layer
    if rname != None:
        conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type,name=[rname[0],rname[1],rname[1]])
        conv = convolutional(conv, filters_shape=(3, 3, filter_num1,filter_num2), activate_type=activate_type,name=[rname[0],rname[2],rname[2]])
    else:
        conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type,name=rname)
        conv = convolutional(conv, filters_shape=(3, 3, filter_num1,filter_num2), activate_type=activate_type,name=rname)
    if cut == True:
        residual_output = short_cut + conv
    else:
        residual_output = conv        
    return residual_output

###6、YOLO V5 C3结构###
def C3(input_layer, c1, c2, n, cut1=True, activate_type="SiLU", C3_name=None):
    if C3_name != None:
        x = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type,name=[C3_name[0],C3_name[1],C3_name[1]])
        input_layer = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type,name=[C3_name[0],C3_name[1]+1,C3_name[1]+1])
        Res_name = np.arange(C3_name[1]+2,C3_name[1]+3+(2*n),1)
        for i in range(n):
            input_layer = residual_block(input_layer,int(c2/2), int(c2/2), int(c2/2), activate_type=activate_type, cut=cut1,rname=[C3_name[0],Res_name[2*i],Res_name[2*i+1]])
        return convolutional(tf.concat([x,input_layer],axis=-1), (1, 1, c2, c2), activate_type=activate_type,name=[C3_name[0],Res_name[-1],Res_name[-1]])
    else:
        x = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type,name=C3_name)
        input_layer = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type,name=C3_name)
        for i in range(n):
            input_layer = residual_block(input_layer,int(c2/2), int(c2/2), int(c2/2), activate_type=activate_type, cut=cut1,rname=C3_name)
        return convolutional(tf.concat([x,input_layer],axis=-1), (1, 1, c2, c2), activate_type=activate_type,name=C3_name)


###7、CSP结构的CBAM
def CSP_CBAM(input_layer,activate_type1="SiLU",CB_name=None):
    if CB_name != None:
        name_num = np.arange(CB_name[1],CB_name[1]+10,1)         
        c1 = c2 = input_layer.shape[3]
        x = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[0],name_num[0]])
        input_layer = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[1],name_num[1]])

        short_cut = input_layer
        input_layer = convolutional(input_layer, filters_shape=(1, 1, int(c2/2), int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[2],name_num[2]])
        input_layer = convolutional(input_layer, filters_shape=(3, 3,int(c2/2),int(c2/2)), activate_type=activate_type1,name=[CB_name[0],name_num[3],name_num[3]])

        F_avg = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
        F_max = tf.keras.layers.GlobalMaxPooling2D()(input_layer)
        F_avg = tf.keras.layers.Reshape((1, 1, F_avg.shape[1]))(F_avg)  # shape (None, 1, 1 feature)
        F_max = tf.keras.layers.Reshape((1, 1, F_max.shape[1]))(F_max)

        F_avg = convolutional(F_avg, filters_shape=(1, 1,int(c2/2),int(c2/4)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=[CB_name[0],name_num[4],name_num[4]])

        F_avg = convolutional(F_avg, filters_shape=(1, 1,int(c2/4),int(c2/2)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=[CB_name[0],name_num[5],name_num[5]])

        F_max = convolutional(F_max, filters_shape=(1, 1,int(c2/2),int(c2/4)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=[CB_name[0],name_num[6],name_num[6]])

        F_max = convolutional(F_max, filters_shape=(1, 1,int(c2/4),int(c2/2)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=[CB_name[0],name_num[7],name_num[7]])

        Channel_Attention = input_layer * (tf.nn.sigmoid(F_avg+F_max))
        avg_out = tf.reduce_mean(Channel_Attention, axis=3)
        max_out = tf.reduce_max(Channel_Attention, axis=3)
        Spatial_factor = convolutional(tf.stack([avg_out, max_out], axis=3), filters_shape=(7, 7,2,1),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            kernel_initializer1='he_normal',         
                            name=[CB_name[0],name_num[8],name_num[8]])
        SpatialAttention = Channel_Attention * Spatial_factor
        input_layer = short_cut + SpatialAttention
        input_layer = convolutional(tf.concat([x,input_layer],axis=-1), (1, 1, c2, c2), activate_type=activate_type1,name=[CB_name[0],name_num[9],name_num[9]])
    else:
        c1 = c2 = input_layer.shape[3]
        x = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type1,name=CB_name)
        input_layer = convolutional(input_layer, (1, 1, c1, int(c2/2)), activate_type=activate_type1,name=CB_name)
        short_cut = input_layer
        input_layer = convolutional(input_layer, filters_shape=(1, 1, int(c2/2), int(c2/2)), activate_type=activate_type1,name=CB_name)
        input_layer = convolutional(input_layer, filters_shape=(3, 3,int(c2/2),int(c2/2)), activate_type=activate_type1,name=CB_name)

        F_avg = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
        F_max = tf.keras.layers.GlobalMaxPooling2D()(input_layer)
        F_avg = tf.keras.layers.Reshape((1, 1, F_avg.shape[1]))(F_avg)  # shape (None, 1, 1 feature)
        F_max = tf.keras.layers.Reshape((1, 1, F_max.shape[1]))(F_max)

        F_avg = convolutional(F_avg, filters_shape=(1, 1,int(c2/2),int(c2/4)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=CB_name)
        F_avg = convolutional(F_avg, filters_shape=(1, 1,int(c2/4),int(c2/2)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=CB_name)
        F_max = convolutional(F_max, filters_shape=(1, 1,int(c2/2),int(c2/4)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=CB_name)
        F_max = convolutional(F_max, filters_shape=(1, 1,int(c2/4),int(c2/2)),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            name=CB_name)

        Channel_Attention = input_layer * (tf.nn.sigmoid(F_avg+F_max))
        avg_out = tf.reduce_mean(Channel_Attention, axis=3)
        max_out = tf.reduce_max(Channel_Attention, axis=3)
        Spatial_factor = convolutional(tf.stack([avg_out, max_out], axis=3), filters_shape=(7, 7,2,1),bn=False,
                            activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                            kernel_initializer1='he_normal',         
                            name=CB_name)
        SpatialAttention = Channel_Attention * Spatial_factor
        input_layer = short_cut + SpatialAttention
        input_layer = convolutional(tf.concat([x,input_layer],axis=-1), (1, 1, c2, c2), activate_type=activate_type1,name=CB_name)
    return input_layer

###8、上采样###
##def upsample(input_layer,activate_type=None,name=None):
##    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
def upsample(input_layer, activate_type='leaky',name=None):
    if name != None:
        if name[0]==0:
            c_name = 'D_conv2d_%d' %name[1] 
            b_name = 'D_batch_normalization_%d' %name[2] 
        elif name[0]==1:
            c_name = 'G_conv2d_%d' %name[1] 
            b_name = 'G_batch_normalization_%d' %name[2] 
        conv = tf.keras.layers.Conv2DTranspose(input_layer.shape[3], 3,strides=(2, 2),padding='same',name=c_name)(input_layer)
        conv = BatchNormalization(name=b_name)(conv)
    else:
        conv = tf.keras.layers.Conv2DTranspose(input_layer.shape[3], 3,strides=(2, 2),padding='same')(input_layer)
        conv = BatchNormalization()(conv)
    
    if activate_type == "leaky":
        conv = tf.nn.leaky_relu(conv, alpha=0.1)
    elif activate_type == "mish":
        conv = mish(conv)
    elif activate_type == "SiLU":
        conv = tf.nn.silu(conv)
    return conv

###9、SPP###
def SPP(input_layer, channel, activate_type="SiLU", SPP_name=None):
    input_data = convolutional(input_layer, (1, 1, input_layer.shape[3], int(channel/2)),activate_type=activate_type,name=SPP_name)
    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = convolutional(input_data, (1, 1, channel*4, channel), activate_type=activate_type,name=SPP_name)
    return input_data


###10、CBAM实验
def CBAM_E(input_layer,activate_type1="SiLU",CB_name=None):
    input_layer = convolutional(input_layer, (3, 3, input_layer.shape[3], 32),downsample=True, activate_type=activate_type1,name=CB_name)

    c  = input_layer.shape[3]
    input_layer = convolutional(input_layer, filters_shape=(3, 3, c, c//2), downsample=True, activate_type=activate_type1,name=CB_name)
    
    short_cut = input_layer
    F_avg = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
    F_max = tf.keras.layers.GlobalMaxPooling2D()(input_layer)
    F_avg = tf.keras.layers.Reshape((1, 1, F_avg.shape[1]))(F_avg)  # shape (None, 1, 1 feature)
    F_max = tf.keras.layers.Reshape((1, 1, F_max.shape[1]))(F_max)

    F_avg = convolutional(F_avg, filters_shape=(1, 1, c//2, c//2),bn=False,
                        activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                        name=CB_name)
    F_max = convolutional(F_max, filters_shape=(1, 1, c//2, c//2),bn=False,
                        activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                        name=CB_name)

    Channel_Attention = input_layer * (tf.nn.sigmoid(F_avg+F_max))
    avg_out, max_out = tf.reduce_mean(Channel_Attention, axis=3), tf.reduce_max(Channel_Attention, axis=3)
    Spatial_factor = convolutional(tf.stack([avg_out, max_out], axis=3), filters_shape=(7, 7, 2, 1),bn=False,
                        activate_type=activate_type1,kernel_regularizer1=regularizers.l2(5e-4),
                        kernel_initializer1='he_normal',         
                        name=CB_name)

    input_layer = convolutional(tf.concat([short_cut, Channel_Attention * Spatial_factor],axis=-1), (3, 3, c, c//4),downsample=True,activate_type=activate_type1,name=CB_name)
    return input_layer


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


###11、torch卷积层###
###默认：CBL，不进行下采样，无指定名称###
def Tconvolutional(filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    backbone = []
    if downsample:
        # input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 1
        strides = 2
    else:
        padding = 0
        strides = 1
        
    backbone.append(nn.Conv2d(in_channels=filters_shape[2], out_channels=filters_shape[-1], kernel_size=filters_shape[0], stride=strides, padding=padding,bias=not bn))
    
    if bn: backbone.append(nn.BatchNorm2d(num_features=filters_shape[-1]))
        
    if activate == True:
        if activate_type == "leaky":
            backbone.append(nn.LeakyReLU(0.1))
        elif activate_type == "silu":
            backbone.append(nn.SiLU())
        elif activate_type == "mish":
            backbone.append(nn.Mish())
    return nn.Sequential(*backbone)


class PFocus(nn.Module):
    def forward(self, x): 
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, activate_type='leaky'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if activate_type == "leaky":
            activate = nn.LeakyReLU(0.1)
        elif activate_type == "silu":
            activate = nn.SiLU()
        elif activate_type == "mish":
            activate = nn.Mish()

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes, 1, bias=False),
                                activate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)*x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)*x