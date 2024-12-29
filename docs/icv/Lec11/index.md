# 3D Deep Learning

## 3D reconstruction

### Feature matching

最重要的任务时 estimate pose，估计相机的位姿。

+ Learning to estimate pose

用深度学习改进特征匹配，用神经网络去学习特征匹配。

### Dense Reconstruction

用一张图很难得到准确的深度值，即带尺度的深度很难估计

+ MVSNet:用 feature map 去算cost volume，再用卷积网络去求解 depth
+ 在传统的流程里，使用像素值去算 cost volume，然后求解 depth，上面是用特征图去算 cost volume，然后用网络求解 depth，网络经过学习里面会有先验知识。

得到绝对尺度的方式
+ 相机标定，标定板是有尺度的
+ 其他的传感器，如 IMU，GPS

能否 improve the mesh quality by comparing the rendered images with the input images?

+ 理论可行，但是很难
  + 基于网格渲染过程不可微分
  + 网格表示不是很适合的优化的对象，因为其具有拓扑结构

+ Implicit representation

<center><img src=./figures/2024-11-28-10-25-22.png width="50%"></center>

+ Implicit Neural Representation
  + 用神经网络近似图像，输出 Occupancy/signed distance

使用神经辐射场做三维重建

<center><img src=./figures/2024-11-28-10-31-29.png width="50%"></center>

输入是空间位置和相机位姿，输出是 rgb颜色和 output density，叠加方向上的颜色就可以得到图像。这样输出的就可以跟原来拍摄的图像做 loss，这样就可以优化。

但是 NeRF 对表面的描述很差，后来改进的工作是 NeuS，网络输出是 SDF，这是表面的精确描述，问题在反光的表面

<center><img src=./figures/2024-11-28-10-40-43.png width="50%"></center>

## 3D understanding

pointnet 最早做点云处理的网络模型