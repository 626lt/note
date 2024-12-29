---
counter: true
---

# Image formation

## Camera and lens

### pinhole camera model

aperture 小孔成像，得到清晰的图像

+ 要求孔的直径很小，但是亮度会越来越低，而且也会有衍射的影响，所以不能无限小

### Lens

解决的方式是透镜(lens)，透镜的作用是将光线聚焦到一个点上，这个点就是焦点(focus) 

$$
\frac{1}{f} = \frac{1}{o} + \frac{1}{i}
$$

+ $f$ 是焦距(focal length)，平行光线经过透镜后会聚焦到焦点上

### Image Magnification 

图像的放大率：变焦距，焦距越大，放大率越大。 $m = \dfrac{h_i}{h_o}$

### Field of view

<center><img src=./figures/2024-11-24-15-26-31.png width=60%/></center>

底片越大，可以加的传感器越大，视场越大

### F-number

+ More convenient to represent aperture as a fraction
of focal length: $D = \dfrac{f}{N}$ 焦距/F-number
+ F-number: $N = \dfrac{f}{D}$

2.8以下的镜头都是大光圈镜头，2.8以上的镜头都是小光圈镜头 

### Lens Defocus

<center><img src=./figures/2024-11-24-15-32-45.png width=60%/></center>

失焦就会形成一个光斑，光斑的正比于光圈的大小，如果拍合照尽量用小光圈，来让尽可能多的人清晰。这个光斑越大代表越模糊。

### Depth of field 景深

有一定的范围是清晰的，只要光斑的大小不超过像素的大小，就是清晰的。景深越小，虚化越明显。景深与焦距成反比。所以拍特写的时候需要景深小，也就是焦距大，光圈大，物距近一点。

<center><img src=./figures/2024-11-24-15-36-48.png width=60%/></center>

## Geometric image formation

### Perspective projection 透视投影
也就是 pinhole camera model

<center><img src=./figures/2024-11-24-15-40-22.png width=60%/></center>

根据相似三角形，可以得到

$$  
u = \dfrac{f}{z}x, \quad v = \dfrac{f}{z}y
$$

### Homogeneous coordinates 齐次坐标

<center><img src=./figures/2024-11-24-15-43-00.png width=60%/></center>

<center><img src=./figures/2024-11-24-15-43-51.png width=60%/></center>

对于齐次坐标系，乘以一个常数仍然是原来的点。透视投影的原物体有无数可能的形状，所有透视投影一定是不可逆的

<center><img src=./figures/2024-11-24-15-45-27.png width=60%/></center>

透视投影的性质：

+ 直线还是直线，长度丢失了，角度也不保持了。
+ 垂直于光轴/平行于相面的平行线投影后还是平行线

### Vanishing points

任何不平行于相面的平行线投影后不再平行，会汇聚到一个点上，这个点就是消失点(vanishing point)。
+ 三维空间内的平行点汇聚到相面上的同一点，这个点与线的位置无关，与朝向有关。消失点告诉我们线的朝向（相对相机），如果消失点在图像内偏上，说明相机是向下倾斜的。
+ 灭点的位置可能在图像外面或无穷远处，无穷远处说明在图像内还是平行的。

!!! note
    **二维空间中的点，是三维空间中射线的投影，齐次坐标就是这条线对应的朝向**
    <center><img src=./figures/2024-11-24-15-52-14.png width=60%/></center>

### Vanishing Lines

+ Multiple Vanishing Points
  + Any set of parallel lines on the plane define a vanishing point
  + The union of all of these vanishing points is the vanishing line
  + Note that different planes define different vanishing lines
    + The direction of the vanishing line tells us the orientation of the plane

线对应 vanishing point，面对应 vanishing line

!!! example
    <center><img src=./figures/2024-11-24-15-59-29.png width=60%/></center>
    这里的海平面就代表了相机的相对高度，在海平面以下的，说明景中的物的海拔比相机要低，想象有一个相机平面，如果一样高的话会汇聚到海平面，所以这里的人会比相机要低。

### Perspective distortion

透视投影会产生畸变，直线还是直线，解决方案是 lens shifted w.r.t flim 轴移相机。

越远离中心，畸变越严重，所以移动相机的时候，要尽量保持相机的中心在物体的中心。

### Radial distortion

透镜的形状不是完美的，会产生径向畸变，这个畸变是非线性的，所以不能用一个矩阵来表示，而是用一个多项式来表示。这种畸变是取决于镜头的，有枕型畸变(pincushion distortion，长焦易发生)和桶型畸变(barrel distortion，短焦易发生)。

<center><img src=./figures/2024-11-24-16-04-38.png width=60%/></center>

$$
\begin{align*}
r^2 &= x_n'^2 + y_n'^2, \\
x_d' &= x_n' \left( 1 + \kappa_1 r^2 + \kappa_2 r^4 \right), \\
y_d' &= y_n' \left( 1 + \kappa_1 r^2 + \kappa_2 r^4 \right).
\end{align*}
$$

### Orthographic projection

+ Special case of perspective projection，最简单的一种做法

<center><img src=./figures/2024-11-24-16-09-40.png width=60%/></center>

!!! tips
  + 要注意缩放、旋转、透视投影等矩阵的公式，齐次坐标下的表示，要理解

## Photometric image formation

### Shutter

+ Shutter speed controls exposure time 曝光时间取决于快门速度
+ The pixel value is equal to the integral of the light intensity within the exposure time

### Rolling shutter effect

理论上的曝光应该是全局曝光，一次完成成像，但是这需要良好的机械结构，受此限制，很多时候采用卷帘快门，逐行曝光，这就导致了每一行得到的图像是不同时刻的，这就是卷帘快门效应。（螺旋桨）

### Color

+ RGB

<center><img src=./figures/2024-11-24-16-17-45.png width=60%/></center>

+ HSV (Hue, Saturation, Value) 色调饱和度亮度

<center><img src=./figures/2024-11-24-16-18-14.png width=60%/></center>

### Practical Color Sensing: Bayer filter

<center><img src=./figures/2024-11-24-16-20-27.png width=60%/></center>

因为人眼敏感，所以对绿色的记录会多，绿色滤镜会多一点。

+ shading：着色

<center><img src=./figures/2024-11-24-16-22-11.png width=60%/></center>

+ Compute light reflected toward camera at a specific point.
+ Inputs
  + Viewer direction, v
  + Surface normal, n
  + Light direction, l
  + Surface parameters BRDF = $f_r(\hat{v}_i,\hat{v}_r,\hat{n};\lambda)$

<center><img src=./figures/2024-11-24-16-23-33.png width=60%/></center>

<center><img src=./figures/2024-11-24-16-24-04.png width=60%/></center>

对朝向 $\hat{V_r}$，颜色为 $\lambda$ 有多少的光强被反射实际上就是入射的光强，乘以 BRDF 再乘以角度（入射光跟法向的夹脚），对所有的入射光积分。

漫反射(Diffuse (Lambertian) reflection)的 BRDF 比较简单，每个角度发出的光都是一样的。

<center><img src=./figures/2024-11-24-16-26-20.png width=60%/></center>

镜面反射(specular reflection)

<center><img src=./figures/2024-11-24-16-27-33.png width=60%/></center>

p 描述镜面反射的属性到底有多强，p越大，镜面反射越强；镜面反射系数(specular coefficient) $k_s$，$k_s$ 越小，镜面反射的越亮。

<center><img src=./figures/2024-11-24-16-31-15.png width=60%/></center>

Ambient reflection 环境反射，没有光源的时候，也会有一些光，这个光就是环境反射。

<center><img src=./figures/2024-11-24-16-32-43.png width=60%/></center>

BRDF 本身的输入很复杂，所以用漫反射、镜面反射、环境反射来近似。