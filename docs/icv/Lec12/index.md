# Lec.13: Computational Photography

+ Data recorded by sensor is not the final image
+ Computational Photography: arbitrary computation to the final image

## High Dynamic Range Imaging(HDR)

<center><img src=./figures/2024-12-06-18-56-43.png width=60%></center>

### Exposure

粗略地讲，曝光是给定场景下被拍到的照片的亮度。

Exposure = Gain x Irradiance x Time

+ Gain: 传感器的增益，由 ISO 控制，ISO 越高越灵敏，但是会引入噪声。同样的 ISO 下，单个像素越大，信噪比越大，噪声就更小
+ Irradiance: 光照强度，由光圈(aperture)控制，一般就是 F 数，F 数越小，光圈越大，光照越强，景深越小
+ Time: 曝光时间由快门速度控制，快门速度还影响图像的模糊程度。

When taking a photo, the averaged exposure should be at the middle of the sensor’s measurement range. So that the photo has both bright and dark parts with details.

过亮或过暗会导致却是细节

### Dynamic range

The ratio between the largest and smallest values of a certain quantity (e.g., brightness)

<center><img src=./figures/2024-12-06-19-11-06.png width=60%></center>

真实世界的动态范围很大，人眼的动态范围也很大，但是传感器的动态范围有限，所以需要 HDR    

### Key Idea

1. Exposure bracketing: Capture multiple LDR images at different exposures
2. Merging: Combine them into a single HDR image

Suppose scene radiance for image pixel (x, y) is L(x, y)

<center><img src=./figures/2024-12-06-19-17-58.png width=60%></center>

<center><img src=./figures/2024-12-06-19-19-38.png width=60%></center>

<center><img src=./figures/2024-12-06-19-19-53.png width=60%></center>

### Tone mapping

<center><img src=./figures/2024-12-06-19-21-45.png width=60%></center>

调整亮度，使得图像看起来更自然，Tone mapping 的方法由相机内置，由相机厂商决定

## Deblurring

### Reason of blurring

+ Defocus: the subject is not in the depth of view
+ Motion blur: moving subjects or unstable camera

### Get a clear image

+ Accurate focus
+ Fast shutter speed
    + Large aperture
    + High ISO
    + One of the reasons for the high price of SLR cameras and lenses
+ Use hardware
    + tripod 三脚架
        + not portable
    + optical image stabilization 相机光学防抖
        + expensive
    + IMU (Inertial Measurement Unit) 加速度计和陀螺仪 比如 云台
        + expensive

### Image Deblurring

如何对模糊建模呢？模糊其实是一种卷积

+ The blurring process can be described by convolution
+ The blurred image is called convolution kernel

<center><img src=./figures/2024-12-06-19-54-12.png width=60%></center>

所以去模糊等于逆卷积

<center><img src=./figures/2024-12-06-19-55-27.png width=60%></center>

任务定义：给定模糊的图像和卷积核，求解原图。

#### Non-blind image deconvolution (NBID)

<center><img src=./figures/2024-12-06-19-59-49.png width=60%></center>

Frequency Domain deconvolution

Convolution in the spatial domain = product in the frequency domain

Spatial deconvolution = division in frequency domain

<center><img src=./figures/2024-12-06-20-01-16.png width=60%></center>


!!! example
    <center><img src=./figures/2024-12-06-20-01-39.png width=60%></center>
    <center><img src=./figures/2024-12-06-20-01-55.png width=60%></center>


但是有一个问题，卷积核一般是一个低通滤波器，因此我们在去卷积的过程中，会乘上一个高通滤波器。所以去卷积我们是在放大高频信息，但与此同时也会相应放大高频噪声。

<center><img src=./figures/2024-12-06-20-02-49.png width=60%></center>

解决这一问题的方法就是调整卷积核。即做inverse fliter的同时抑制高频噪声

<center><img src=./figures/2024-12-06-20-02-59.png width=60%></center>

其他方法：优化方法

+ Variable to be optimized:
    + image to be recovered
+ Objective function:
    + The similarity of the blurred image and the given blurred image (likelihood) 模糊得到的图像和给定的模糊图像的相似性
    + The recovered image looks real (prior)

<center><img src=./figures/2024-12-06-20-09-38.png width=60%></center>

但是 Deconvolution is ill-posed，存在 non-unique solution，所以需要先验信息——自然图像是相对光滑的

<center><img src=./figures/2024-12-06-20-11-12.png width=60%></center>

+ Natural images are generally smooth in segments
+ Gradient map is sparse 也就是说，图像的梯度图是稀疏的
+ Adding L1 regularization makes the image gradient sparse

<center><img src=./figures/2024-12-06-20-12-04.png width=60%></center>

#### Blind image deconvolution (BID)

+ The convolution kernel is also unknown
+ Obviously more difficult——need more prior knowledge

这时候就有两个变量需要优化了，一个是图像，一个是卷积核

+ Kernel Prior
  + Blurn kernel is non-negative and sparse
+ Optimized objective function

<center><img src=./figures/2024-12-06-20-14-50.png width=60%></center>

## Colorization

### Image Colorization

黑白图像变成彩色图像

上色是指在电脑的帮助下给单色图片或视频添加色彩的过程

有两种主要方式给灰度图上色：

+ Sample-based colorization: use sample image 给一个样本
+ Interactive colorization: paint brush interactively 用户告诉计算机哪些部分应该是什么颜色

本质上就是根据输入

<center><img src=./figures/2024-12-06-20-20-03.png width=60%></center>

<center><img src=./figures/2024-12-06-20-25-20.png width=60%></center>

<center><img src=./figures/2024-12-06-20-30-51.png width=60%></center>

权重是灰度图中的像素值差别，约束是用户给定的颜色

Video Colorization is similar to image colorization 当作 (x, y, t) 三维数据处理

### Modern Colorization

<center><img src=./figures/2024-12-06-20-40-16.png width=60%></center>

这里的问题是，一种物体不一定只有一种颜色，这个loss不能处理多种颜色可能性

<center><img src=./figures/2024-12-06-20-40-38.png width=60%></center>

解决的方式是使用 GAN，学习一个 loss function

+ 先用生成图像和真实图像学一个神经网络（判别网络），用于判断图像是不是真实的

<center><img src=./figures/2024-12-06-20-45-37.png width=60%></center>

+ G 尝试生成 fake images that fool D
+ D 尝试区分 real and fake images

<center><img src=./figures/2024-12-06-20-49-36.png width=60%></center>

<center><img src=./figures/2024-12-06-20-49-48.png width=60%></center>

这种 min-max 很难收敛，训练 GAN 非常困难

GAN 最本质的是学了一个 loss function, 叫做 adversarial loss, 用 learned 而不是 hand-crafted loss function,可以用在很多图像合成任务上

如果要考虑用户的输入呢？把用户的输入也作为一个输入

用 GAN 还有一个好处是，可以生成多种可能性

## More Image Synthesis Tasks

+ Super-resolution
+ Image to Image Translation
  + style transfer
  + Text-to-Image
  + Image dehazing
+ Pose and garment transfer