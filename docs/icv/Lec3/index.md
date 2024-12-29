---
counter: true
---
# Lec.03: Image processing

## Image processing basics.(需要清楚这些概念)

Increasing contrast with "S curve"

<center><img src=./figures/2024-11-24-16-41-45.png width=60%></center>

这里代表提高对比度，亮度越大就调的越高。在对角线上代表这个点的亮度不变。

!!! note convolution
    <center><img src=./figures/2024-11-24-23-59-23.png width=60%></center>
    <center><img src=./figures/2024-11-25-00-05-01.png width=60%></center>

### padding

由于卷积的操作会使得图像的尺寸变小，所以需要padding操作，常见的padding操作有：

+ zero padding
+ Edge padding
+ symmetric padding

### blur

#### Gaussian blur

<center><img src=./figures/2024-11-25-00-06-33.png width=60%></center>

$\sigma$ 越小，分布越尖锐，越少模糊

#### Sharpening

突出中心减少周围成分对它的影响，在卷积核中心为正，周围为负。

+ 设 I 是原始图像
+ 高频成分是 I = I - blur(I)
+ sharpened image = I + I - blur(I)

也就是添加了高频成分，让图像变得更加锐利。

想检测什么东西，滤波器的矩阵形式就应该是怎么样的

!!! example edges
    <center><img src=./figures/2024-11-25-00-15-21.png width=60%></center>

+ Gradient detection filter:可以将滤波器视为模式的“检测器”，输出图像中像素的大小是滤波器对输入图像中每个像素周围区域的“响应”
+ Bilateral filter: 双边滤波，对不同的区域有不同的相应，可以保留边缘；kernel depends on image content

## Image sampling.

Resolution: pixels/inch，像素尺寸与物理尺寸的比值

+ Reducing image size - downsampling，但是很容易会出现失真

比如摩尔纹就是对一个连续信号离散化后采样产生的，对于格子很小的衬衫会出现这个问题，一般大格子的衬衫不会出现这个问题。

<center><img src=./figures/2024-11-25-00-24-07.png width=60%></center>

Wagon Wheel Illusion (False Motion)，时域上也会有信号失真

**Aliasing**：由于采样引起的失真，根本原因是 Signals are changing too fast but sampled too slow

信号变化快在数学上描述是频率高，采样太慢就会出现失真。

<center><img src=./figures/2024-11-25-00-27-27.png width=60%></center>

<center><img src=./figures/2024-11-25-00-27-05.png width=60%></center>

对任意信号如何描述频率，但是有频谱——傅立叶变换

!!! note Fourier Transform
    represent a function as a weighted sum of sines and cosines
    <center><img src=./figures/2024-11-25-00-29-29.png width=60%></center>
    频谱就是每一个频率的振幅，也就是频率的权重
    <center><img src=./figures/2024-11-25-00-30-11.png width=60%></center>
    傅立叶变换的本质就是把信号用不同的正弦余弦信号做内积，得到不同频率的权重

!!! example 常见的傅立叶变换
    <center><img src=./figures/2024-11-25-00-36-13.png width=60%></center>
    <center><img src=./figures/2024-11-25-00-36-32.png width=60%></center>
    <center><img src=./figures/2024-11-25-00-36-46.png width=60%></center>
    <center><img src=./figures/2024-11-25-00-36-58.png width=60%></center>
    <center><img src=./figures/2024-11-25-00-37-09.png width=60%></center>
    高斯函数有很好的性质，傅立叶变换后还是高斯函数，变换前后的 $\sigma$ 互为倒数

### Convolution Theorem

<center><img src=./figures/2024-11-25-00-38-08.png width=60%></center>

空间域的卷积就等于频率的乘积，空间域的乘积就等于频率的卷积。

二维的频谱对应的是两个方向上的变化的频率

<center><img src=./figures/2024-11-25-00-41-13.png width=60%></center>

卷积核对应的就是窗口函数，上面的例子里面就是过滤了周围的部分，只留下中间低频的部分，这就是滤波的含义。

<center><img src=./figures/2024-11-25-00-42-51.png width=60%></center>

上面的叫做低通滤波器，卷积核越大，代表着频率越低

### Sampling

<center><img src=./figures/2024-11-25-00-44-09.png width=60%></center>

<center><img src=./figures/2024-11-25-00-44-45.png width=60%></center>

<center><img src=./figures/2024-11-25-00-45-05.png width=60%></center>

采样信号频率低导致间隔大，然后对应到频谱上就是间隔小，做完卷积以后就容易出现重叠，导致失真。

如何才能减少 aliasing？

+ 增加采样率，最少需要多少？至少要大于信号的最高频率的两倍

!!! example Nyquist-Shannon Theorem
    <center><img src=./figures/2024-11-25-00-48-00.png width=60%></center>

+ Anti-aliasing，减少原来信号的频率，滤波把原来高频信号去掉

<center><img src=./figures/2024-11-25-00-49-29.png width=60%></center>

+ Filtering = convolution
+ Steps for anti-alisaing
  1. Convolve image with low-pass filter(e.g., Gaussian) 
  2. Sample it with a Nyquist rate

## Image magnification. 图像缩放

### 放大

插值：通过已知的点估计未知的点
+ 最近邻插值：Nearest-neighbor interpolation：不连续不光滑
+ 线性插值：Linear interpolation：连续但是不光滑
+ 三次插值：Cubic interpolation：连续光滑

二维插值：分别在两个方向上做插值

+ Bilinear interpolation：在两个方向上做线性插值（计算快很多，大部分时候足够好了）
+ Bicubic interpolation：在两个方向上做三次插值

### 缩小

+ Basic idea 去掉相对不重要的像素，比如连续的没有变化的像素
+ Importance of pixels：像素的重要性，如何衡量？

$$
E(I) = \left|\frac{\partial I}{\partial x} + \frac{\partial I}{\partial y}\right|
$$

实现使用卷积核，然后找到每一个行最小的点，**但是这每一行的点是不连续的**，会导致图像变混乱。所以需要 Seam carving：可以理解为最短路径算法，从上至下找一条 $E(I)$ 最小的路径。可以用动态规划实现。

!!! Seam Carving
    Going from top to bottom
    + $M(i,j) =$ minimal energy of a seam going through $(i,j)$
    + $M(i,j) = E(i,j) + \min(M(i-1,j-1), M(i-1,j), M(i-1,j+1))$
    + Solved by dynamic programming

这样可以较完整地保留图像信息，并且把图像变小。

那么对应的，扩大图像也是找到这样的最短路，然后插值即可

<center><img src=./figures/2024-11-30-20-30-19.png width=60%></center>