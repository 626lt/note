---
counter: true
---

# Image Stitching

## Image warping

+ image filtering: change **intensity** of image
+ image warping: change **shape** of image

<center><img src=./figures/2024-12-02-21-38-58.png width=60% /></center>

参数化全局变形，即图像的每一个坐标都遵循同一个变换函数，对于这种全局的变化，可以用一个矩阵来描述 

+ 线性变换
  + scale
  + reflection
  + rotation
  + shear
+ Affine 仿射变换 = 线性变化 + 平移

<center><img src=./figures/2024-12-02-21-41-55.png width=60% /></center>

现在考虑更广义的情况，最后一行的矩阵不是 [0,0,1] 的情况，这种变换称为 perspective 变换 透视变换(Homography/单应变换)

<center><img src=./figures/2024-12-02-21-46-31.png width=60% /></center>

单应变换的自由度是 8，因为是在齐次坐标系下 up to scale(can be multiplied by a scalar)

<center><img src=./figures/2024-12-02-21-51-07.png width=60% /></center>

+ 什么情况下是单应变换（单应变换要求是一个一一对应的变换）
  + 相机旋转但是中心不变
  + 相机中心移动并且 scene 是一个平面

<center><img src=./figures/2024-12-02-21-55-18.png width=60% /></center>

+ Euclidean 3 freedom

Inverse Transform: $T^{-1}$

### Implementing image warping

Given a coordinate transform $(x',y')=T(x,y)$ and a source image $f(x,y)$ 如何得到目标图像 $g(x',y') = f(T(x,y))$

#### Forward warping

Send each pixel $f(x)$ to its corresponding location $(x’,y’) = T(x,y)$ in $g(x’,y’)$。这样的问题在于经过变换后的坐标不一定还是整数，对于计算机这样的情况无法处理。

#### Inverse warping

对于新变换得到的图像的网格，将其逆变换回来会得到原来图像的一个位置，即使这个不是整数，我们可以通过**插值**的方法来得到这个位置的像素值。这不是像之前一样把原来图像变换到新的像素位置，而是在新的位置向原来查询像素值。Get each pixel $g(x’,y’)$ from its corresponding location $(x,y) = T-1(x,y)$ in $f(x,y)$

<center><img src=./figures/2024-12-02-22-06-12.png width=60% /></center>

## Image Stitching

如何计算变换：
1. Image matching(each match gives an equation)
2. Solve T from the obtained matches

### 仿射变换

<center><img src=./figures/2024-12-02-22-10-49.png width=60% /></center>

+ 6 个未知量
+ 每对点给出两个方程
+ 至少 3 对匹配

<center><img src=./figures/2024-12-02-22-19-34.png width=60% /></center>
<center><img src=./figures/2024-12-02-22-20-09.png width=60% /></center>
<center><img src=./figures/2024-12-02-22-32-27.png width=60% /></center>

### 单应变换

<center><img src=./figures/2024-12-02-22-33-22.png width=60% /></center>
<center><img src=./figures/2024-12-02-22-33-37.png width=60% /></center>
<center><img src=./figures/2024-12-02-22-35-22.png width=60% /></center>

+ 这里对 h 一定要有约束，否则没有意义，常见的约束有二范数为1，设置 h 是单位向量
+ 解：$\hat{h} =$ $A^TA$ 的最小特征值的特征向量
+ 实际上只需要 4 对匹配就可以得到单应变换

### Outliers

实际做的时候错误匹配的影响最大

+ 换目标函数
+ RANSAC
  + 每次随机选取 4 对匹配进行拟合得到矩阵
+ Idea
  + All the inliers will agree with each other on the translation vector;
  + The outliers will disagree with each other
    + RANSAC **only has guarantees** if there are < 50% outliers
  +  “All good matches are alike; every bad match is bad in its own way.”
+ General version
1. Randomly choose s samples
   + Typically s = minimum sample size that lets you fit a model
2. Fit a model(e.g.,transformation matrix)to those samples
3. Count the number of inliers that approximately fit the model
4. Repeat N times
5. Choose the model that has the largest set of inliers
6. Final step: least squares fit to all inliers（对于所有投票的点，都进行一次拟合）

### Image Stitching

1. Detect feature points
2. Feature matching
3. Compute transformation matrix with RANSAC
4. Fix image 1 and warp image 2（可能会做亮度的调整）

!!! note Panoramas
    <center><img src=./figures/2024-12-02-22-51-29.png width=60% /></center>

图片拼接一定是有变形的，越往边上变形越大。如果要做 Full Panoramas，解决方案是 Projection Cylinder，把图像投影到柱面上

<center><img src=./figures/2024-12-02-23-05-38.png width=60% /></center>

**相机的旋转在柱面上是平移**，投影到柱面以后，再去做拼接旋转就比较简单了，只要在柱面上做就可以了。但是这样会导致累积误差，求解的时候加上一个约束，所有的平移加起来是 0

<center><img src=./figures/2024-12-02-23-08-47.png width=60% /></center>


