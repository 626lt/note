---
counter: true
---
# Lec.04: Model Fitting and Optimization

## Optimization

<center><img src=./figures/2024-11-30-20-31-44.png width=60% ></center>

+ 优化变量：$x$
+ 目标函数：$f_0(x)$
+ 不等约束：$f_i(x) \le 0\; i=1,\dots,m$
+ 等式约束：$g_i(x) = 0\; i=1,\dots,p$

!!! example Image deblurring
    <center><img src=./figures/2024-11-30-20-36-36.png width=60% ></center>

### Model Fitting

+ 数学模型 $b = f_x(a)$ 描述了输入 $a$ 和输出 $b$ 之间的关系，其中 $x$ 是模型参数。
  + 例如线性模型 $b = a^Tx$。
+ 那么如何从数据中估计模型参数 $x$ 呢？（这通常叫做 learning）
+ 经典的方法是 Minimize the Mean Square Error (MSE) 均方差
  + $\hat{x}=\arg \min_x \sum_i(b_i-a_i^Tx)^2$ 模型的输出与 ground truth 之间的差异。 
+ 为什么 MSE? 

!!! note 统计解释

    均方误差(MSE) = 最大似然估计(MLE) + 高斯噪声假设
    最大似然估计：找到一组模型参数，使得观察到的数据的概率最大。
    + 我们假设数据中存在高斯噪声

    $$
    b_i = a_i^T + n, n\sim G(0,\sigma)
    $$

    + 对于给定 x，likelihood of observing $(a_i,b_i)$ 是观察到这组数据的概率

    $$
    P[(a_i,b_i)|x] = P[b_i - a_i^Tx] \propto \exp-\frac{(b_i-a_i^Tx)^2}{2\sigma^2}\end{aligned}
    $$

    + 假设数据是相互独立的

    $$
    \begin{aligned}
    &P[(a_1,b_1)(a_2,b_2)...|x] \\
    &\Large=\prod_iP[(a_i,b_i)|x] \\
    &\begin{aligned}=\prod_iP[b_i-a_i^Tx]\end{aligned} \\
    &\begin{aligned}\propto\exp-\frac{\sum_i(b_i-a_i^Tx)^2}{2\sigma^2}=\exp-\frac{\|Ax-b\|_2^2}{2\sigma^2}\end{aligned}
    \end{aligned}
    $$

    + 残差向量 $r = Ax-b$
    + 最大似然估计(MLE) = Maximize the likelihood to find the best x

    $$
    \begin{aligned}
    \hat{x} &= \arg\max_x P[(a_1, b_1)(a_2, b_2) \dots | x]\\
    &= \arg\max_x \exp\left(-\frac{\|Ax - b\|_2^2}{2\sigma^2}\right)\\
    &= \arg\min_x \|Ax - b\|_2^2
    \end{aligned}
    $$

    等价于最小化均方误差(MSE)。
    + MSE = MLE with Gaussian noise assumption

## Numerical Methods

+ Some problems have analytical solution 一些问题有解析解，可以得到解的数学表达式
  + Linear MSE $A^TAx = A^Tb$
+ 如果没有解析解
  + 找到近似解——solution path
    + $F(x_0) > F(x_1) > \dots > F(x_k) > \dots$

<center><img src=./figures/2024-11-30-20-59-50.png width=60% ></center>

梯度下降法 Gradient Descent

<center><img src=./figures/2024-11-30-21-00-19.png width=60% ></center>

!!! note Taylor expansion
    <center><img src=./figures/2024-11-30-21-03-28.png width=60% ></center>
    <center><img src=./figures/2024-11-30-21-03-41.png width=60% ></center>

<center><img src=./figures/2024-11-30-21-03-59.png width=60% ></center> 

### Steepest descent method 最速梯度下降法

+ $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$
+ $\alpha_k$ 是步长
+ $\nabla f(x_k)$ 是梯度
+ 下降的方向是梯度的负方向

#### step size

<center><img src=./figures/2024-11-30-21-14-48.png width=60% ></center>

转化为步长 $\alpha$ 的单变量函数，找到的方法有

+ Exact line search 精确线搜索
+ Backtracking algorithm 回溯算法（理解即可）
  + Ininialize $\alpha$ with a big value
  + Decrease $\alpha$ until 

$$
\psi(\alpha) \leq \psi(0) + \gamma \psi'(0)\alpha
$$

$\gamma \in (0,1)$ 是预先设定的参数

<center><img src=./figures/2024-11-30-21-18-20.png width=60% ></center>

总结最速梯度下降法
+ advantage 优点
  + Easy to implement
  + Perform well when far from the minimum
+ disadvantage 缺点
  + Converge slowly when near the minimum
  + Waste a lot of computation
+ 为什么收敛慢
  + 只是用了一阶导数信息
  + Does not use curvature

### Newton method 牛顿法

做二阶泰勒展开

$$
F(x_k+\delta x) \approx F(x_k) + J_F\delta x + \frac{1}{2}\delta x^TH_F\delta x
$$

找到最小的 $\delta x$

$$
H_F\delta x + -J_F = 0
$$

所以优化的方向（Newton step）

$$
\delta x = -H_F^{-1}J_F
$$

相当于用二阶导去调整梯度下降的方向。

+ Advantage 优点
  + Fast convergence near the minimum
+ Disadvantage 缺点
  + Hessian requires a lot of computation
+ Can we approximate Hessian?

### Gauss-Newton method 高斯牛顿法

+ Useful for solving nonlinear least squares problems 对非线性最小二乘问题很有效

<center><img src=./figures/2024-11-30-21-24-50.png width=60% ></center>

+ 展开残差向量 $R(x)$ 而不是目标函数 $F(x)$，近似的方式是对里面的进行一阶展开，再取平方，近似二阶展开

<center><img src=./figures/2024-11-30-21-26-43.png width=80% ></center>

+ 优化 $\Delta x$ 满足

$$
J_R^TJ_R\Delta x + -J_R^TR(x_k) = 0
$$

+ 优化方向

$$
\Delta x = -(J_R^TJ_R)^{-1}J_R^TR(x_k)
$$

+ 与 Newton method 比较
  + Newton step : $\Delta x = -H_F^{-1}J_F = -H_F^{-1}J_R^TR(x_k)$
  + Gauss-Newton use $J_R^TJ_R$ 近似 Hessian $H_F$

总结
+ Advantage 优点
  + Fast convergence
  + Avoid computing Hessian 用一阶导来近似二阶导，加速计算（）
+ Disadvantage 缺点
  + If $J_R^TJ_R$ is singular,  the algorithm becomes unstable 如果 $J_R^TJ_R$ 是奇异的，即近似的矩阵不可逆，算法会变得不稳定

### Levenberg-Marquardt method 莱文贝格-马夸特法 

因为 $J_R^TJ_R$ 存在不满秩的问题，我们的解决方式是加上一个正则项

$$
\Delta x = -(J_R^TJ_R + \lambda I)^{-1}J_R^TR(x_k)
$$

对于所有的 $\lambda$，$J_R^TJ_R + \lambda I$ 必须要是正定的

+ $\lambda$ 的作用
  + 当 $\lambda$ 很大时，相当于梯度下降并且 step size 很小
  + 当 $\lambda$ 很小时，相当于高斯牛顿法 

总结
+ Advantage 优点
  + Start quickly ($\lambda$ 增大)
  + Fast convergence ($\lambda$ 减小)
  + Do not degenerate ($J_R^TJ_R + \lambda I$ 必须要是正定的)
  + LM = Gradient descent + Gauss-Newton

### Local minimum and global minimum 局部最小值和全局最小值

+ 梯度下降法 can only find the local minimum
+ For some functions, local minimum is global minimum
+ 构建一个能保证找到最优解的近似问题，来作为初始化/或者多初始化几次
+ 凸优化一定能找到全局最优解（局部最优解 = 全局最优解）

## Robust estimation

### Outliers 异常值

+ Inlier: obeys the model assumption
+ Outlier: differs significantly from the assumption

外点对最小二乘法的影响很大，因为它们的残差很大
+ 使用其他的 Lose fuction:L1,Huber
+ They are called robust functions

<center><img src=./figures/2024-11-30-21-56-49.png width=60% ></center>

### RANSAC

另外一种方法使用 RANSAC (Random Sample Concensus)
+ 最有力的处理外点的方法
+ 核心思想
  + 内点的分布是相似的，但是外点不是
  + Use data point pairs to vote

<center><img src=./figures/2024-11-30-21-58-21.png width=60% ></center>

先用其中的几个点拟合，看看剩下的点有多少在上面，找到最多的就是最准确的拟合。

+ 过拟合和欠拟合

### ill-posed problem

+ The solution is not unique
+ use prior knowledge to add more constraints to make the solution unique

### L2 regularization

+ L2 norm: $\|x\|_2 = \sum_i x_i^2$
+ L2 regularization: 

<center><img src=./figures/2024-12-01-21-57-59.png width=60% ></center>

去除掉一些冗余的无用变量

### L1 regularization

<center><img src=./figures/2024-12-01-21-58-17.png width=60% ></center>

## Graphcut
 