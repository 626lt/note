# Depth estimation and 3D reconstruction

SFM 得到的是稀疏的点云，这次介绍的是三维精细的结构，即稠密的点云，三维网格。这里重要的是要得到每一个点的深度，即深度估计。

## Depth sensing

深度值用灰度图来表示，深度值越大，灰度值越小。用 1/z 来表示灰度，离得越远值越小。这里的 z 是相机到物体的距离。深度也是很多计算机视觉应用的基础，比如虚拟现实，自动驾驶等。

<center><img src=./figures/2024-12-29-20-59-30.png width="80%"></center>

### Active depth sensing 

通过主动地向外传递信号来探测深度，比如激光雷达(LiDAR)，Active stereo，结构光相机(后面两个是通过立体视觉来实现深度估计)等。这些方法都是通过测量光的时间来得到深度信息。

<center><img src=./figures/2024-12-29-21-01-57.png width="80%"></center>

!!! example 激光雷达(LiDAR)
    <center><img src=./figures/2024-12-29-21-03-21.png width="80%"></center>
    首先是360度转动的，然后根据 Time of Flight(ToF) 来计算深度，主要是根据相位差来计算深度。缺点是非常贵，不会特别特别的准确，相对视觉的方案比较准确，分辨率相对较高，但不会特别的高。

### Passive depth sensing

被动式的，传感器没有发出信号，只是接收信号，比如双目视觉，单目视觉。本章主要介绍视觉的双目深度估计。

<center><img src=./figures/2024-12-29-21-06-20.png width="80%"></center>

## Stereo matching

### Stereo vision

单目视觉其实只要相机在动，产生了视差也能够感知到深度，得到位置其实就是通过三角化来得到的。

+ 一个物体点将投影到图像中的某个点上
+ 这个像素点对应世界中的一条光线
+ 两条光线的交点就是这个点的位置，所以我们想要在3D空间中得到这个点的位置，就需要两个视角。

与 SFM 的三角化只是对特征点进行重建不同，这里的三角化要在每一个点上做，得到整张图的深度。这是稠密的深度估计。计算深度需要相机的内参、外参，为了方便我们常使用双目相机来得到深度，这样外参内参都是已知的，只需要进行匹配就可以得到深度。做法就是先做2D-2D的匹配（这等价于光流估计，但是比光流简单，因为存在对极几何的约束，$x_L^T$ \* Fundamental Matrix \* $x_R=0$），然后通过三角化得到深度。

!!! note "Epipolar gemoetry"
    <center><img src=./figures/2024-12-29-21-22-10.png width="80%"></center>

    这里补充一个 Epipolar lines 的概念，即两相机中心和三维点平面与像平面的交线，左图中 x 的对应点一定在右图的 Epipolar line 上，这样就减少了搜索的范围。这条线的方程满足的其实就是 $x_R$ 的方程，即 $x_L^T * F * x_R=0$。这里 F 已知，因为给定了相机的内外参数。


### Basic stereo matching algorithm

+ 对第一张图中的每一个像素点
    + 找到第二张图中对应的线
    + 沿着这条线找到最相似的点

+ 最简单的情况下:epipolar lines fall along the horizontal scan lines of the images
    + Image planes of cameras are parallel to each other and to the baseline(相机中心的连线)
    + Camera centers are on the same Height（这个条件让 Epipolar line 与原像素点同高，对应点匹配点的y值相等）
    + Cameras have the same focal length（焦距一致）

<center><img src=./figures/2024-12-29-21-34-03.png width="80%"></center>

在计算深度之前首先要计算 disparity 视差

<center><img src=./figures/2024-12-29-21-35-13.png width="80%"></center>

注意这里的 $x_2,x_1$ 是相对各自的原点的，不是中间这条线的长度

<center><img src=./figures/2024-12-29-21-38-07.png width="80%"></center>

我们通过相似三角形很容易得到上面的公式，所以视差和深度是成反比的关系。

### Stereo image rectification

实际情况中会更复杂一些，相机拍到的图像不满足上面的条件，我们的做法是把拍到的图片投影到一个满足上面条件的平面上，这个过程是两个单应变换（本质上就是调整相机的朝向，相机的中心是不变的），这样就简化到上面的情况。

<center><img src=./figures/2024-12-29-21-42-34.png width="80%"></center>

上面是如何找 Epipolar line 的，下面是如何找对应点的。有很多的方法，最基本的就是比较两个 window 之间的相似度，这里要选一个窗口，去比较窗口内的相似度，主流的匹配值有：SSD, SAD, NCC。ZNCC 的好处是不受光照的影响，因为是归一化的，减去均值除以标准差。

<center><img src=./figures/2024-12-29-22-08-57.png width="80%"></center>

上面提到的 windows size 也有影响

+ 小窗口：细节多，噪声多
+ 大窗口：细节少，噪声少

<center><img src=./figures/2024-12-29-22-11-10.png width="80%"></center>

上面的匹配结果存在很多的噪声，一个直接的想法是做滤波去模糊掉这些噪声，但这又会造成误差，不精确了。其实滤波之所以能生效是因为我们假设了相邻位置的深度是相近的。我们可以在更早的时候就利用这个先验来提高匹配的精度，即马尔可夫随机场。

!!! note 马尔可夫随机场
	<center><img src=./figures/2024-12-29-22-13-42.png width="80%"></center>
	<center><img src=./figures/2024-12-29-22-15-24.png width="80%"></center>
	<center><img src=./figures/2024-12-29-22-19-51.png width="80%"></center>

	上面的公式是一个能量函数，我们的目标是最小化这个能量函数，第一部分就是匹配得到的结果，第二部分是平滑项，利用周围像素的空间一致性关系，来提高匹配的精度。这个问题可以通过动态规划来独立最小化每一行。

	<center><img src=./figures/2024-12-29-23-13-31.png width="80%"></center>

总结:Stereo reconstruction pipeline

1. Calibrate cameras 相机标定
2. Rectify images 图像矫正
3. Compute disparity 计算视差
4. Estimate depth 估计深度

### Choose the stereo baseline

相机的基线长度对估计的效果也是有影响的

+ 基线太短了，深度误差很大：目标点实际上就是两个视线的交点，如果基线太短了，这个光线的角度就会很大，一有偏差就会有很大的误差
+ 基线太长了，视差越大，匹配就会越困难；也会有遮挡的问题

<center><img src=./figures/2024-12-30-00-23-35.png width="80%"></center>

引起误差的原因还有

+ Camera calibration errors 相机标定误差
+ Poor image resolution 图像分辨率不够高
+ Occlusions 遮挡，匹配不上
+ Violations of brightness constancy (specular reflections) 违反亮度一致性
+ Textureless regions 纹理不够丰富/透明的区域

最后两个是匹配本身造成的问题

为了解决这些问题，可以**主动**打一个结构光的条纹，这样就有了纹理。这个结构光一般用红外光，因为我们通常希望在得到深度的同时也得到一个 RGB 的图像。

另外这时估计深度只用一个红外相机来接受就可以了，因为这里的 projector 也是一个位姿信息，如果 projector 和第二个相机完全重合，那他接收到的图也就是打出去的图，那么就不用第二个相机了。所以很多的红外光传感器实际上就是一个 RBG 相机（获得色彩） + 一个红外相机（获得深度） + 一个projector

<center><img src=./figures/2024-12-30-00-33-49.png width="80%"></center>

+ passive stereo 被动式的，只有相机，被动地接受光信号，只通过拍照来进行立体匹配
+ Active stereo 主动式的，会主动打一些光，然后通过这些光来进行立体匹配，精度最高，只要图像分辨率够高，就能得到很高的精度
+ Lidar(ToF) 通过激光雷达来进行深度估计，根据光的时间来计算深度，很稳定，但是很难达到亚毫米级

## Multi-view stereo

MVS 本质也是**深度估计问题**，但是是在多个视角下进行的，这样可以得到更多的信息，更精确的深度估计。

<center><img src=./figures/2024-12-30-00-42-41.png width="80%"></center>

相比双目的优势：

+ Can match windows using more than 1 neighbor, giving a stronger constraint
可以使用多个邻居来匹配窗口，提供更强的约束。
+ If you have lots of potential neighbors, can choose the best subset of neighbors to match per reference image
如果有很多潜在的邻居，可以选择最佳的邻居子集来匹配参考图像，就可以舍弃一部分不好的匹配。
+ Can reconstruct a depth map for each reference frame, and the merge into a complete 3D model
可以为每个参考帧重建深度图，然后合并成完整的3D模型

基本思想：正确的深度会给出一致的投影
做法：对参考图中的每一个点的每一个深度计算投影误差，那么误差最小的深度就是正确的深度

<center><img src=./figures/2024-12-30-00-44-32.png width="80%"></center>

接下来的问题就是如何高效地计算深度

!!! note Plane-Sweep
	<center><img src=./figures/2024-12-30-00-46-57.png width="80%"></center>
	
	对于参考图像平面，投影到不同的深度，然后再投影到邻居平面内，计算误差即可

	<center><img src=./figures/2024-12-30-00-48-38.png width="80%"></center>

	Cost volume: 一个三维的矩阵，其中的每一个元素是一个像素点在不同深度下的投影误差
	这种方式求解出来的深度图效果一般比较好，但是开销仍然比较大，本质上是一种穷举法

想要实时地求解深度图，我们会采用 PatchMatch，这是一种提高效率的方法计算深度图

!!! note PatchMatch
	这是一种基于随机猜测的算法，基于的假设是大量的随机采样会有一些好的结果；邻居会有相似的移动，也就是一个区域内的像素变换是相近的。

	<center><img src=./figures/2024-12-30-00-53-07.png width="80%"></center>

	1. random initialization：首先是随机初始化，对每个像素给一个随机的 patch offset 作为初始化
	2. propagation：传递，对每个像素的值，查看邻居的offset是不是更好，如果是就把邻居的 offset 赋值给他。这样对的 offset 就会传递开
	3. local search：基于上一步的offset，在邻域内进行搜索，查找更好的 offset
	4. 然后不断迭代 2，3 步骤，直到收敛

	In MVS, replace patch offsets by depth values in the above algorithm

## 3D reconstruction

pipeline：

1. Compute depth map per image 用前面的方法计算深度图
2. Fuse the depth maps into a 3D surface 融合深度图到一个3D表面，得到一个稠密的点云
3. Texture mapping 纹理映射，把纹理贴到这个表面上

### 3D Representation

<center><img src=./figures/2024-12-30-01-09-08.png width="80%"></center>

+ Point cloud 点云：一堆点的集合，每个点有一个坐标，很粗糙，没有表面信息

<center><img src=./figures/2024-12-30-01-09-32.png width="80%"></center>

+ Volume：体素，三维的格子，每个点的信息都有，包括 Occupancy/signed distance，前面表示这个点有没有被占据这里的值是 0/1，后面表示三维空间的点到物体表面的距离，是连续的值，带符号的距离 SDF: Signed Distance Function

<center><img src=./figures/2024-12-30-01-09-48.png width="80%"></center>

+ SDF volume:
    + Signed Distance Function: 一个点到物体表面的距离，是连续的值，带符号的距离
    + The distance is defined by a metric, usually the Euclidean distance
    + Truncated SDF(TSDF): 一个截断的 SDF，只保留一定范围内的 SDF，这样可以减少计算量，一般是 -1 到 1 之间

<center><img src=./figures/2024-12-30-01-10-29.png width="80%"></center>

+ Mesh：网格，A polygon mesh with vertices and edges，最常用的是三角形网格

<center><img src=./figures/2024-12-30-01-10-42.png width="80%"></center>

### 3D surface reconstruction

MVS 最后得到的是一个稠密的点云（深度图），我们需要把这个点云转换成一个体素，进而转化为表面 Mesh

1. Depth maps -> Occupancy volume
   + Poisson reconstruction
2. Occupancy volume -> mesh
   + Marching cubes

做中间这一步转化的好处是可以去到取噪的过程，而且两部分都有成熟的算法

#### Poisson reconstruction

这个算法是把**深度图**变成点云最后变成**体素**

<center><img src=./figures/2024-12-30-01-16-16.png width="80%"></center>

+ 把深度图转化为点云
+ 计算每个点的法向量：这个点附近点的分布方差最小的方向就是这个点的法向量
+ 然后把这个表面恢复转化为优化问题，最小化这个表面的梯度，这样就可以得到一个平滑的表面，这个优化问题实际上是一个泊松方程（也是最小二乘法）

<center><img src=./figures/2024-12-30-01-18-34.png width="80%"></center>
<center><img src=./figures/2024-12-30-01-19-06.png width="80%"></center>

#### Marching cubes

这个算法是把**体素(Occupancy/SDF)**变成**网格(Mesh)**

现在我们有了三维点的表示，要重建出表面，本质上就是在找出表面的顶点，然后把这些顶点连接起来。

<center><img src=./figures/2024-12-30-01-31-47.png width="80%"></center>

+ 对每个网格，四个顶点的值如果一致，说明不存在表面
+ 如果不一致就说明有表面，然后对于每一条边，如果边的两个顶点值不同，说明这条边上有表面的顶点，如果是 occupancy 就定在中间，如果是 SDF 就有更加精细的插值方法，然后根据这些点的位置，连接这些点，就得到了表面的边

<center><img src=./figures/2024-12-30-01-35-10.png width="80%"></center>

对于 3D 的情况，我们可以用类似的方法

+ For each grid cell with a sign change 对于每个有符号变化的格子
    +  Create one vertex on each grid edge with a sign change 在每个有符号变化的格子的边上创建一个顶点
    +  Connect the vertices to form triangles 连接这些顶点形成三角形
    +  三角形不应该相交
    +  比二维的情况要更复杂

<center><img src=./figures/2024-12-30-01-37-02.png width="80%"></center>
<center><img src=./figures/2024-12-30-01-37-13.png width="80%"></center>

最后就只有 15 种情况，都存在 look-up table。

<center><img src=./figures/2024-12-30-01-38-11.png width="80%"></center>

### Texture mapping

最后一步就是把纹理贴到这个表面上，就是给表面上色。

<center><img src=./figures/2024-12-30-01-40-13.png width="80%"></center>
<center><img src=./figures/2024-12-30-01-40-26.png width="80%"></center>

+ 三维表面上的每个三角形的点都一一对应一个二维的坐标（在纹理图上的uv坐标），通过这个坐标就能把纹理图的三角形映射到三维表面上的三角形上。中间任意一个点的坐标都能通过插值得到，这样就能得到一个纹理映射。
+ 对于三维物体，我们可以将其表面剪开，铺平成二维图像，就形成了纹理图。但这个纹理图还没有颜色，其实是通过图像得到的，把每一个三角形投影到各个图像中，取最清晰看到这一块的图像，就是这个三角形的颜色。最清晰看到其实是相机的轴和面片的法相是最一致的，就是正面看到最清晰的。

## Neural Scene Representations

离散的体素表达存在分辨率问题，但我们实际上需要的是一个对任何一个3D坐标，能给出这个点的体素即可，我们可以用神经网络来做这个事情，这是几何表示，没有颜色。

<center><img src=./figures/2024-12-30-01-50-04.png width="80%"></center>

<center><img src=./figures/2024-12-30-01-51-02.png width="80%"></center>

NeRF: Neural Radiance Fields 神经辐射场，这个模型的输入是三维坐标和视角方向，输出是颜色和密度，这个密度是用来表示这个点是否在表面上的，颜色是这个点的颜色。每个网络对应的是一个辐射场，可以渲染为任意视角的图像（体渲染），渲染为输入视角的图像，然后跟输入图像作 loss 优化，这样就可以得到一个很好的模型。这是一种端到端的优化，不需要中间的深度图，效果比较好。

<center><img src=./figures/2024-12-30-01-52-37.png width="80%"></center>

问题是 NeRF 对表面的重建比较粗糙，而使用 SDF 的表示方法会更好，这就是 Neural SDFs for Surface Reconstruction(NeuS)

<center><img src=./figures/2024-12-30-01-56-32.png width="80%"></center>