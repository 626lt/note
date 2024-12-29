# From Coarse to Fine: Robust Hierarchical Localization at Large Scale

## Abstract

精确鲁棒的视觉定位是自动驾驶、移动机器人或增强现实等应用的基础。但对于大规模场景和显著的外观变化还是有很大的挑战，当时的 sota 不仅难以应对而且开销大。在本文中，作者提出了 HF-Net，这是一种基于整体 CNN 的分层定位方法，可同时预测局部特征和全局描述符，以实现准确的六自由度定位。利用了从粗到细的分层定位（From Coarse to Fine），我们首先执行全局检索以获得位置假设，然后才匹配这些候选位置内的局部特征。这种分层的策略减小了开销，使得其可以实时进行，同时也在外观的大变化中实现了较好的鲁棒性。

## Introduction

第一段讲应用，包括 autonomous driving in GPS-denied environments，consumer devices with augmented reality features；包括 computer vision tasks such as Structure-fromMotion (SfM) or SLAM。

第二段讲目标的重要性：鲁棒性和降低开销

第三段讲现有的方法：当前的领先方法主要依赖于使用局部描述符估计查询中的 2D 关键点与稀疏模型中的 3D 点之间的对应关系。

+ 要么是精度够了但是很难在移动设备上处理
+ 针对效率优化后又会有表现不稳定的情况

作者认为上面两种情况的鲁棒性受到 poor invariance of hand-crafted local features 的限制，本文用了 CNN 的方式来做特征提取。

<center><img src=./figures/2024-11-29-16-54-50.png width="50%"></center>

本文使用 learned feature 来平衡效率和鲁棒性，而学习的关键点由于其更高的可重复性而提高了计算和内存方面的效率。为了进一步提高这种方法的效率，我们提出了一种分层特征网络（HF-Net），一种联合估计局部和全局特征的 CNN，从而最大限度地共享计算。

本文的主要贡献：

+ 我们在大规模本地化的多个公共基准中设定了新的最先进水平，在特别具有挑战性的条件下具有出色的鲁棒性；
+ 我们引入了 HF-Net，这是一种单片神经网络，它可以有效地预测分层特征，以实现快速、鲁棒的定位；
+ 我们展示了多任务蒸馏的实际用途和有效性，以通过异构预测器实现运行时目标。

## Hierarchical Localization

### hierarchical localization framework

#### Prior retrieval 预检索

通过使用全局描述符将查询图片与数据库图像进行匹配来执行地图级别的粗略搜索。 k 最近邻 (NN) 称为先验帧，表示地图中的候选位置。鉴于数据库图像比 SfM 模型中的点少得多，这种搜索是有效的。

#### Covisibility clustering 共视聚类

先前的帧根据它们共同观察到的 3D 结构进行聚类。这相当于在将数据库图像链接到模型中的 3D 点的共视图中查找连接的组件（称为位置）。

#### Local feature matching 局部特征匹配

对于每个位置，我们依次将查询图像中检测到的 2D 关键点与该位置中包含的 3D 点进行匹配，并尝试在 RANSAC 方案内使用 PnP 几何一致性检查来估计 6-DoF 姿态。这种局部搜索也很有效，因为该位置所考虑的 3D 点数量明显少于整个模型。一旦估计出有效的姿势，算法就会停止。

#### Discussion

MobileNetVLAD (MNV) 对模型蒸馏有助于实现给定的运行时约束，同时部分保留原始模型的准确性。但是局部匹配的方式使用的 SIFT，其计算成本昂贵并生成大量特征，使得该步骤特别昂贵。并且导致不能有效地扩展到大场景中。同时 SIFT 和基于学习的特征匹配相比可计算性差许多。这段是在说明为什么用 CNN 来替代 SIFT 做特征匹配。

## Proposed Approach

基于学习的特征提取在关键点重复性和描述符匹配方面优于流行的 baseline 例如(SIFT)。另外，一些学习到的特征比 SIFT 稀疏得多，从而减少了要匹配的关键点数量并加快了匹配步骤。图像检索中最先进的网络和局部特征的结合自然地实现了最先进的定位。这种方法在极具挑战性的条件下尤其出色，例如夜间查询，其性能大大优于竞争方法，并且 3D 模型尺寸更小。HF-Net 只用一次检测关键点并计算本地和全局描述符，从而最大限度地共享计算，但保留 baseline network 的性能。

!!! note
    <center><img src=./figures/2024-11-29-17-23-39.png width="80%"></center>

    上面的 pipline 分成两个部分，offline 的部分构建本地数据库，通过 HF-Net 提取全局描述符和局部特征，前者作为 Global Index，后者用于 SFM 重建得到三维模型。对于 Online，也是通过 HF-Net 提取全局描述符和局部特征，利用前者先进行 KNN search的预检索，这样可以大致划定特征的位置，就大大减少了全局搜索的开销。后者用于局部特征匹配，通过 PnP 估计姿态。


### HF-Net Architecture

CNN 本身就是分层的结构，很适合同时进行局部和全局特征的预测