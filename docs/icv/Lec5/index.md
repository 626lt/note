---
counter: true
---
# Image Matching and Motion Estimation

## Image Matching

!!! exampe applications of feature matching
    + Image alignment
    + Image Stitching
    + 3D reconstruction
    + Motion Tracking
    + Object recognition
    + Indexing and database retrieval
    + Robot navigation

整体步骤：

<center><img src=./figures/2024-12-01-22-03-38.png width=60%></center>

### Detection

+ Feature points or interest points
  + uniqueness
    + 如何描述？

<center><img src=./figures/2024-12-01-22-07-34.png width=60%></center>
<center><img src=./figures/2024-12-01-22-07-45.png width=60%></center>

#### principal Component Analysis 主成分分析

<center><img src=./figures/2024-12-01-22-08-30.png width=60%></center>

上图中绿色的就是协方差矩阵对应的特征向量

#### Corner Detection 角点检测

1. Compute the covariance matrix at each point

<center><img src=./figures/2024-12-01-22-09-46.png width=60%></center>

2. Compute the eigenvalues of the covariance matrix

<center><img src=./figures/2024-12-01-22-09-59.png width=60%></center>

3. Classify points using eigenvalues of H:

<center><img src=./figures/2024-12-01-22-10-51.png width=60%></center>

##### Harris Operator

计算更加简化，不用求出特征值具体的值

<center><img src=./figures/2024-12-01-22-11-15.png width=60%></center>

Harris detector

1. Compute derivatives at each pixel
2. Compute covariance matrix H om a Gaussian window around each pixel
3. Compute corner response function f
4. Threshold f (阈值化，相当于激活函数)
5. Find local maxima of response function(Non-maximum suppression) 只保留最大的

现在检测出了特征点，要保证特征点在两个图中可重复。在数学上定义就是

+ 我们希望 response value f at the corresponding pixesl to be invaraint to image transformations
对于图像的变换，我们希望特征点的响应值是不变的

##### Image transformations

<center><img src=./figures/2024-12-01-22-19-20.png width=60%></center>

+ Invariance properties
  + photometric transformation

<center><img src=./figures/2024-12-01-22-21-07.png width=60%></center>

+ Image translation，平移具有不变性

<center><img src=./figures/2024-12-01-22-21-39.png width=60%></center>

+ Image rotation，旋转也具有不变性

<center><img src=./figures/2024-12-01-22-21-55.png width=60%></center>

+ Image scaling，缩放不是无关的

<center><img src=./figures/2024-12-01-22-22-53.png width=60%></center>

如何找到合适的scale，对每一张图，去尝试不同大小的尺度，然后找到最大的响应值 f。
+ scale 是指角点检测的尺度，框的大小 

<center><img src=./figures/2024-12-01-22-33-35.png width=60%></center>

###### Automatic Scale Selection

<center><img src=./figures/2024-12-01-22-34-01.png width=60%></center>

这个曲线是不同的窗口大小对应的相应值，两张图得到两个曲线，对应的峰值就是对应的scale

<center><img src=./figures/2024-12-01-22-35-34.png width=60%></center>

以上是一种实现方式，但是实际上我们往往不去变窗口的大小而是图像的大小，这样就可以用图像金字塔，这样更高校

we can implement using a fixed window size with an image pyramid

<center><img src=./figures/2024-12-01-22-37-20.png width=60%></center>

两种方式是等价的。

!!! Abstract 角点检测
    + 如何描述角点
    + 整个计算过程
    + 对哪些性质具有很好的不变性
    + 对缩放不具有不变性但是可以通过尺度的选择，也具有不变性

#### Blob detectior

斑点检测，斑点也是很好的特征，斑点在两个方向的二阶导都很大

<center><img src=./figures/2024-12-01-22-41-48.png width=60%></center>

!!! note Laplacian of Gaussian (LoG)
    <center><img src=./figures/2024-12-01-22-43-11.png width=60%></center>
    由于二阶导对噪声敏感，所以在做 Laplacian 算子卷积之前要先做高斯滤波来平滑一下噪声
    <center><img src=./figures/2024-12-01-22-43-31.png width=60%></center>
    由于卷积的结合律，可以先把卷积核和高斯核结合在一起(LoG)，然后再卷积。
    但同样也有尺度的问题，尺度受高斯函数的 $\sigma$ 控制，尺度的选择与前面类似
    <center><img src=./figures/2024-12-01-22-46-38.png width=60%></center>

### Difference of Gaussian (DoG)

由于 LoG 的计算量大，所以我们可以用 DoG 来近似，DoG 是两个高斯核的差

<center><img src=./figures/2024-12-01-22-50-11.png width=60%></center>

用不同的高斯核去卷积图像等价于用同一个高斯核去卷积不同大小的图像，本质上就是图像金字塔对同一个高斯核做卷积然后两两相减就得到了 DoG

<center><img src=./figures/2024-12-01-22-52-34.png width=60%></center>

因为本来也要做高斯滤波，只要对图形金字塔做差分就可以了，所以计算量会小很多

!!! abstract 
    + 什么是好的 feature point
      + unique
      + Invariant to transformations
    + Popular detectors
      + Harris conner detector
      + Blob detector(LoG, DoG)

### Descriptions

现在我们能够 detect good points，下一个问题是如何匹配他们？
答案是：从每一个点中分离出一个 descriptor，在两张图片中找到相似的 descriptor

对于 descriptor 的要求：
+ Patches with similar content should have similar descriptors.

!!! note Raw patches
    The simplest way to describe the neighborhood around an interest point is to write down the list of intensities to form a feature vector.
    But this is very sensitive (not invariant) to even small shifts, rotations.
    <center><img src=./figures/2024-12-01-23-25-48.png width=60%></center>
    之前的不变性是对 f 而言的，f是通过梯度得到的，所以我们可以考虑用梯度分布来描述

#### SIFT descriptor

Scale Invariant Feature Transform (SIFT)

<center><img src=./figures/2024-12-01-23-28-45.png width=60%></center>

只考虑梯度方向的分布，用直方图作为描述子
+ 平移不变性
+ 旋转；**分布会平移一部分，但是总体的分布不变，为了保持这一不变性，用直方图归一化即可**
+ 对亮度变化不敏感，因为是梯度
+ 对于缩放是会变化的，**但是可以通过尺度选择（即上一步骤的 DoG detector）来保持不变性。**所以 SIFT 是尺度不变的

+ SIFT 是两部分
  + Detection
  + Description

综合以上可以看到，这是一个非常鲁棒的特征描述子

<center><img src=./figures/2024-12-01-23-31-20.png width=60%></center>

!!! note Lowe’s SIFT algorithm
    + Run DoG detector
      + Find maxima in location/scale space
    + Find dominate orientation
    + For each (x,y,scale,orientation), create descriptor

Properties of SIFT

+ Extraordinarily robust matching technique
  + Can handle changes in viewpoint
    + Theoretically invariant to scale and rotation(上一部分解释了)
  + Can handle significant changes in illumination
    + Sometimes even day vs. night 因为对亮度不敏感
  + Fast and efficient—can run in real time
  + Lots of code available

<center><img src=./figures/2024-12-01-23-39-31.png width=60%></center>

### Matching

Feature Matching: Given two sets of feature descriptors, find the best matching pairs.
对于低维的情况用一些方法可以加速，但对高维的情况就很难了，基本上就是两两算一遍，对目前的算力不成问题

Given a feature in $I_1$, how to find the best match in $I_2$?
1. Define distance function that compares two descriptors
2. Test all the features in $I_2$, find the one with min distance

#### Feature distance

How to define the difference between two features f1, f2?
+ Simple approach: L2 distance ||f1 - f2|| 欧氏距离
+ Can give small distances for ambiguous (incorrect) matches 有时候会具有歧义性

#### Ratio Test

+ Ratio score = 1st best distance / 2nd best distance
+ Ambiguous matches have large ratio scores 歧义性越大，这个比值越大

#### Mutual Nearest Neighbors 相互最近邻

+ Another strategy: find mutual nearest neighbors
  + f2 is the nearest neighbor of f1 in I2
  + f1 is the nearest neighbor of f2 in I1

#### Learning-based local features

<center><img src=./figures/2024-12-02-00-23-22.png width=60%></center>

## Motion Estimation

两类问题

+ Feature-tracking
  + Extract feature(interest) points and track them over multiple frames
  + Output: displacement of sparse point 稀疏点的跟踪
+ Optical flow
  + Recover image motion at each pixel
  + Output: dense displacement field (optical flow) 稠密的光流

一个方法：Lucas-Kanade method

<center><img src=./figures/2024-12-02-12-58-49.png width=60%></center>

与特征匹配的区别：
+ 特征匹配是从图像中找特征再去做匹配
+ 运动估计的点是已经给定的，可能不是特征点

Key assumptions of Lucas-Kanade
1. Small motion: points do not move very far away
2. Brigntness constancy: same point looks the same in every frame
3. Spatial coherence: points move like their neighbors

### Lucas-Kanade method

<center><img src=./figures/2024-12-02-13-19-57.png width=60%></center>

首先利用假设2 Brightness constancy，我们可以得到一个方程

$$
I(x,y,t)=I(x+u,y+v,t+1)
$$

再利用假设1 Small motion，我们可以对上式进行泰勒展开

$$
I(x+u,y+v,t+1) \approx I(x,y,t) + \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t}
$$

结合两式我们有：

$$
I_xu + I_yv + I_t = 0 \rightarrow \nabla I \cdot [u,v]^T = -I_t
$$

以上等式存在两个未知数，有无穷解，所以需要更多的等式，利用第三个假设 Spatial coherence，我们可以得到更多的等式

+ Assume the pixel’s neighbors have the same(u,v)
+ If we use a 5x5 window,that gives us 25 equations per pixel

<center><img src=./figures/2024-12-02-13-24-09.png width=60%></center>

+ More equations than variables
+ 所以转化为优化问题 $\min_d \|Ad-b\|^2$
  + Least squares solution for d given by

<center><img src=./figures/2024-12-02-13-25-20.png width=60%></center>

但是上面方程的可解性与 $A^TA$ 有关，如果 $A^TA$ 不可逆，那么就无解，不可逆意味着不满秩
+ 也就是说特征值 $\lambda_1$ 和 $\lambda_2$ 不能够太小
  + 回忆前面特征提取的部分，特征值比较小对应图像的什么情况？
    + 根据 Harris corner detector，对于平坦的区域 flag 和边缘 edge，即非角点的区域，特征值比较小
    + 这意味着光流估计的效果不好，这与直觉是符合的，对于平坦和边缘一般很难分辨这个点是否运动了

<center><img src=./figures/2024-12-02-13-30-51.png width=60%></center>

<center><img src=./figures/2024-12-02-13-31-04.png width=60%></center>

纹理比较丰富的区域能够比较好的估计
<center><img src=./figures/2024-12-02-13-32-31.png width=60%></center>

!!! example The aperture problem
    <center><img src=./figures/2024-12-02-13-31-48.png width=60%></center>
    <center><img src=./figures/2024-12-02-13-31-38.png width=60%></center>

### Errors in Lucas-Kanade

+ 假设 $A^TA$ 是可逆的
+ 假设图像中没有很多的噪声
 
潜在的可能导致 error 的问题：

当我们的推导时的假设不成立时，会导致误差
+ Brightness constancy is not satisfied 亮度有剧烈变化
+ The motion is not small 运动太大
+ A point does not move like its neighbors 空间一致性不成立，遮挡边缘

现在考虑 small motion assumption
如果给定两张图片，但是相对位移比较大，应该如何实现呢？降低分辨率，降低分辨率后运动也变小了。在低分辨率做 LK 算法，再放大回去。

但是该方法的缺点就是在缩小图片的过程中会丢失信息，这样图像移动距离的精度就无法保证。

这就涉及到计算机视觉中很重要的思想 Coarse-to-fine strategy 从粗到细的策略

<center><img src=./figures/2024-12-02-21-29-05.png width=60%></center>

在上层做估计，然后在第二层中恢复这个运动，此时有了第一层粗略估计的结果，再与第二层的 t+1 时刻的图像进行比较，逐层把上一层的结果传递下去，最终得到最终的结果，提升精度。先做简单的粗略的估计，再在更高分辨率上做更精细的估计。

!!! abstract
    对于特征匹配最重要的是其中具有一些的不变性，在图像之间特征点具有不变性
    这两类问题叫做对应关系问题(correspondence problems)