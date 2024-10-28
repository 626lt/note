# 2D Graphics

### 引入
convert 2D primitives into a raster image
+ for output on a monitor/printer
+ rasterization or scan conversion

CRT:阴极射线管 

<img src="./figures/2024-09-18-08-10-36.png" width="75%">

光栅是为了减少相邻电子之间的干扰

LCD:

<img src="./figures/2024-09-18-08-19-01.png" width="75%">   

efficiency is most important to the performance of a display system

高性能的算法一般都固化在硬件中,Modern PCs (video cards)

## Line Segments

坐标系：笛卡尔坐标系，定义在整数点上

### Scan converting a line segment
线是建模世界很有力度元素。线段是由两个端点定义的，端点的 pixels 和 color。

<img src="./figures/2024-09-18-08-25-37.png" width="75%">   

Requirement:
+ the selected pixels should lie as close to the ideal line as possible
+ the sequence of pixels should be as straight as possible
+ all lines should appear to be of constant brightness independent of their length and orientation
+ should start and end accurately
+ should be drawn as rapidly as possible
+ should be possible to draw lines with different width and line styles

How to draw a line:

<img src="./figures/2024-09-18-08-35-07.png" width="75%">

+ Equation of a line: $y = mx +c$ , for line starting at $(x_0, y_0)$ and ending at $(x_1, y_1)$, we have $m = \frac{y_1 - y_0}{x_1 - x_0} = \frac{\Delta y}{\Delta x}$

+ Digital Differential Analyzer (DDA) Algorithm:

<img src="./figures/2024-09-18-08-38-22.png" width="75%">

如何要获得只用整数运算的算法，从而完全避免浮点数？

+ Bresenham's Line Drawing Algorithm

<img src="./figures/2024-09-18-08-56-37.png" width="75%">

Some equations:

$$
\begin{aligned}
& y = m(x_i + 1) + b \qquad & dx= x_2-x_1  \\
& d_1 = y - y_i  \qquad & dy = y_2-y_1 \\
& d_2 = y_i + 1 - y  \qquad & m=dy/dx
\end{aligned}
$$

if $d_1 - d_2 > 0$,then $y_{i+1} = y_i + 1$, otherwise $y_{i+1} = y_i$
$d_1-d_2=2y-2y_i-1=2dy/dx*x_i+2dy/dx+2b-2y_i-1$
both $*dx$, donate $(d_1-d_2)dx$ as $p_i$, we have 
$p_i=2x_{i+1}dy - 2y_{i+1}dx+2dy+(2b-1)dx$
$p_{i+1}=p_i+2dy-2(y_{i+1}-y_i)dx$

<img src="./figures/2024-09-18-10-00-06.png" width="75%">


### 3D lines

## Circles
A circle with center $(x_c, y_c)$ and radius $r$ is defined by the equation $(x - x_c)^2 + (y - y_c)^2 = r^2$

用类似的直角坐标计算依然需要根据45度进行划分，这里考虑用极坐标convert

<img src="./figures/2024-09-18-10-02-51.png" width="75%">

问题在于这里使用增量的形式会导致误差的累积

## Polygons
### Filling Polygons 填充多边形
+ even-odd test / winding number test

<img src="./figures/2024-09-18-09-18-42.png" width="75%">

scan-line algorithm
+ Use intersections between region boundaries and scan lines to identify pixels that are inside the area
+ Exploit the coherence
  + Spatial coherence: Except at the boundary edges, adjacent pixels are likely to have the same characteristics
  + Scan line coherence: Pixels in the adjacent scan lines are likely to have the same characteristics

<img src="./figures/2024-09-18-09-21-54.png" width="75%">

<img src="./figures/2024-09-18-09-25-23.png" width="75%">

<img src="./figures/2024-09-18-09-26-14.png" width="75%">

Seed fill algorithm
+ Assumes that at least one pixel inside the polygon is known
+ A recursive algorithm
+ Useful in interactive painting packages

<img src="./figures/2024-10-22-10-54-36.png" width="75%">

## clipping

Removal of content that is not going to be displayed

在convert之前做，是为了efficient