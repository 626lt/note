# Recognition

## Semantic Segmentation 语义分割

### sliding window 分割每一块

- Very inefficient:逐个像素进行分类，速度慢，过神经网络开销还是大的
- Limited receptive fields:感受野很小，每一次只看到很小的局部信息，容易出错

### Fully Convolutional Networks 全卷积神经网络

+ 一次就做完所有预测，整个图片作为输入过一遍神经网络
+ 因为是语义分割，分类出最接近的类别，所以可以用 Per-pixel cross-entropy loss 作为损失函数
问题是：只在原来分辨率做卷积，感受野还是有限制的；而且在高分辨率上 expensive 

<center><img src=./figures/2024-11-22-14-43-02.png width=600></center>

解决的方案就是先变小再变大
+ 变小的方法：Downsampling,pooling,strided convolution

+ 变大的方法用的是 unpooling，本质上是插值
  + Bed of Nails: 用0填充，然后卷积
  + Nearest Neighbor: 用最近的像素填充，最近邻插值
  + Bilinear Interpolation: 用周围的像素加权平均，双线性插值
  + Bicubic Interpolation: 三次插值

<center><img src=./figures/2024-11-22-18-07-39.png width=600></center>

上采样的过程也叫做解卷积(transposed convolution)

<center><img src=./figures/2024-11-22-18-08-39.png width=600></center>

但是单纯是变小还是会损失掉一些信息，如果能把前面的信息也结合起来就好了，U-Net就是这样做的。

+ Skip connections: 把前面同样大小的层跟后面的叠在一起(concat)再做卷积。因为是把原来的特征跳过中间的卷积，所以叫做skip connections

<center><img src=./figures/2024-11-22-18-11-12.png width=600></center>

Deep Lab 简介：

+ 用 atrous convolution 来增大感受野（卷积范围大但是卷积核不大来提高计算效率），而不是用 pooling
+ 用 CRF 来做后处理，在最后加了一个条件马尔可夫随机场，加入了空间相关性的约束 

<center><img src=./figures/2024-11-22-18-13-36.png width=600></center>

!!! note Conditional random field 条件随机场
    Energy function: $E(x) = \sum_{i}\theta_i(x_i) + \sum_{ij}\theta_{ij}(x_i,x_j)$
    Unary potential: $\theta_i(x_i) = -\log P(x_i)$ score given by the network
    Pairwise potential: $\theta_{ij}(x_i,x_j) = \mu(x_i,x_j)[w_1 \exp(-\dfrac{\left\|p_i-p_j\right\|^2}{2\sigma_\alpha^2}-\dfrac{\left\|I_i-I_j\right\|^2}{2\sigma_\beta^2})+w_2\exp(-\dfrac{\left\|p_i-p_j\right\|^2}{2\sigma_\gamma^2})]$，
    其中$\mu(x_i,x_j)=1\;\text{if}\;x_i \neq x_j\;\text{zero otherwise}$

### Evaluation metrics 评价指标

+ Per-pixel Intersection-over-union (IoU)

<center><img src=./figures/2024-11-22-18-27-50.png width=600></center>

## Object Detection 目标检测

+ 输入：RGB image
+ 输出：bounding box + class label 框

> Bounding box (bbox):Class label, Location(x,y), Size(w,h)    

这个任务相对于之前的语义分割最大的不同在于输出不再是单一的类别，输出不统一了，要输出很多东西，如果还是用单一网络输出 location，可以考虑跟之前一样的方法，做 sliding window，但是这样效率太低，改进的方式是 region proposal

对图像分割生成大量候选的区域，经常是基于 heuristics，相对快很多。

### R-CNN

+ Run region proposal method to compute ~2000 region proposals
+ Resize each region to 224x224 and run through CNN
+ Predict class scores and bbox transform
+ Use scores to select a subset of region proposals to output

<center><img src=./figures/2024-11-22-18-41-25.png width=600></center>

### Evaluation metrics

IoU: Intersection over Union

+ IoU > 0.5: "decent"
+ IoU > 0.7: "pretty good"
+ IoU > 0.9: "almost perfect"

这样划分了很多框，可能会导致重复检测，所以引入了 Non-maximum suppression

+ Select the highest-scoring box
+ Eliminate lower-scoring boxes with IoU > threshold
+ If any boxes remain, goto 1

检测重复度，重复度大就丢掉。

!!! note
    对于检测的任务一般都会用最大值抑制，一个区域内只检测出一个结果

以上的任务还是慢，主要有两个原因：

+ 2000个候选区域，每个区域都要做卷积，计算量大（传统方法）—— 减少框的数量
+ 框和框之间重复性很大 —— 避免重复计算

### Fast R-CNN

开始把整张图过一个小一点的卷积层，提取出一些特征图，然后把这些特征图上对应每个框的特征做一个pooling，然后再去过后面的层，这样前面的层就可以共享。前面提取特征的层叫做 backbone（主干网络）。ROI pooling: Crop and resize feature maps to fixed size，得到感兴趣的区域

<center><img src=./figures/2024-11-22-18-53-14.png width=600></center>

### Faster R-CNN

在提取特征的基础上用 CNN 去选择 proposals

<center><img src=./figures/2024-11-22-19-23-11.png width=600></center>

#### RPN

在特征图上的每个点都有一个预设好大小的框，然后额外输出四个变量的偏移量，这样解决了前面输出不统一的问题。

<center><img src=./figures/2024-11-22-19-25-25.png width=600></center>

Faster R-CNN 是比较准确的，但是不够快，从过程分析来看，这是两阶段的方法

+ First stage:run once per image
  + Backbone network
  + RPN
+ Second stage:run once per proposal
  + Crop features: RoI pool / align
  + Predict object class
  + Predict bbox offset

Region Proposal Network 如果粒度够细，本身就可以作为一个分类，这就诞生了下面的方法。

Single-stage object detection:

<center><img src=./figures/2024-11-22-19-29-06.png width=600></center>

输出的时候就带着类别(C+1)，+1 代表背景

!!! example YOLO
    <center><img src=./figures/2024-11-22-19-30-29.png width=600></center>

### 比较

+ Two-stage is generally more accurate
+ Single-stage is generally faster

## Instance Segmentation 实例分割

Faster R-CNN + additional head(Mask R-CNN) 再多输出一个 mask 就可以了 

+ Snake 先给一个框，然后用优化的方式来画出框。
+ Deep Snake 用神经网络来做这个优化

### Panoptic Segmentation 全景分割

语义分割和实例分割的结合

+ Beyond instance-segmentation
  + Label all pixels in the image (both things and stuff)
  + For “thing” categories also separate into instances

## Human pose estimation 人体姿态估计

用关键点来表示人体的姿态

### Single human

用神经网络标记关键点的位置，但是这样不太准确，改进的方法使用热力图，每一个关键点对应一个热力图，好处是可以用全卷积网络来输出。

<center><img src=./figures/2024-11-22-20-33-06.png width=600></center>

### Multiple humans

+ Top-down: Detect humans first, then estimate keypoints(Mask R-CNN) 速度慢，人重叠时不准确，大部分情况准确
+ Bottom-up: Detect keypoints and group keypoints to form humans (OpenPose) 只用过一遍网络，对有遮挡的情况更准确

affinity field: 对每个关键点都输出一个向量场，代表更有可能往哪个方向去连

## Video 视频分析

### video classification 视频分类

+ 3D CNN

### Temporal action localization 时间动作定位

Given a long untrimmed video sequence, identify frames corresponding to different actions

视频切成不同的片段，然后对每个片段进行分类

### Spatial-temporal action localization 时空动作定位

先提 proposal，然后对每个 proposal 进行分类

### Multi-object tracking

Identify and track objects belonging to one or more categories without any prior knowledge about the appearance and number of targets.

每一帧得到框，然后对框进行跟踪，特征跟踪或者光流，也可以训练一个网络去判断前后的框是不是同一个物体