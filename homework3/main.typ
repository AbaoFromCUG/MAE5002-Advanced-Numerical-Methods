#import "template.typ": *
#import "@preview/xarrow:0.3.0": *

#set page(numbering: "1", paper: "a4")
#set math.mat(delim: "[")
#set math.mat(gap: 0.5em)
#init("Homework 3", "Abao Zhang (张宝)", student_number: "12332459")

= Question 1

#set enum(numbering: "(a)")
+ 是对称矩阵
+ 不是对称矩阵
+ 是对称矩阵
+ 不是对称矩阵

= Question 2

== (a)

A为M x N矩阵，X为列向量，A的每一行乘以X，需要N次乘法，共有M行， *总共需要进行$M
dot N$次乘法*

== (b)

A为M x N矩阵，X为列向量，A的每一行乘以X再累加求和，每次求和需要$N-1$次加法，共有M行，
*总共需要进行$M dot (N-1)$次乘法*

= Question 3

$
  mat(
    a_11, a_12, ..., a_(1 (N-2)), a_(1 (N-1)), a_(1 N);0, a_22, ..., a_(2 (N-2)), a_(2 (N-1)), a_(1 N);dots.v, dots.v, dots.down, dots.v, dots.v, dots.v;0, 0, ..., 0, a_((N-1) (N-1)), a_((N-1) N);0, 0, ..., 0, 0, a_(N N);
  )
  mat(x_1;x_2;x_3;dots.v;x_N)
  =
  mat(y_1;y_2;y_3;dots.v;y_N)
$
如上公式，当时使用回代法求解每个变量$x_i$时，需要顺序计算$x_N, x_(N-1),...,x_2, x_1$。容易知道在求$x_N$，需要1次除法，0次乘法，0次加减法，

假设此时已知道$x_N, x_(N-1),x_k$，要求解$x_(k-1)$，需要将$x_N,x_(N-1),..., x_k$的值代入方程$a_(k-1)x_(k-1)+a_k x_k+...+a_(N-1)x_(N-1)+a_N x_N=y_(k-1)$中。此时需要1次除法，$N-k+1$次除法，$N-k+1$次加减法，

归纳并求和：
- 总除法次数
  $=overbrace(1+1+...+1, "N个1")=N$
- 总乘法次数 $=0+1+...+(N-2)+(N-1)=N(N-1)/2$
- 总加减法次数 $=0+1+...+(N-2)+(N-1)=N(N-1)/2$

#pagebreak()
= Question 4

== (a)

在本问中，由于计算较为繁琐，且都是重复性计算，所以我编写了`question-4.py` 进行计算验证，手写了高斯消元法，包括反代法、Partial
pivoting、Scaled partial pivoting。下面仅给出Scaled partial pivoting的实现
```python
def gs_partial_scaled_pivot(A: np.ndarray, B: np.ndarray):
    M, M = A.shape
    AB = np.hstack((A, B))

    for p in range(M - 1):
        k = np.argmax(np.abs(AB[p:, p] / np.max(np.abs(AB[p:, p:]), axis=1))) + p
        if k > p:
            AB[[p, k], :] = AB[[k, p], :]

        for i in range(p + 1, M):
            AB[i, :] = AB[i, :] - AB[p, :] * AB[i, p] / AB[p, p]
    return back_substitution(AB)
```
下面给出每个小题的计算过程:

=== (i) Gaussian elimination with partial pivoting
$
  mat(A|B)&=
  mat(
    underline(2), -3, 100, 1;1, 10, -0.001, 0;3, -100, 0.01, 0;augment: #(-1)
  ) \
          &
  xarrowDouble("pivot 1, 3 row")
  mat(
    underline(3), -100, 0.01, 0;1, 10, -0.001, 0;2, -3, 100, 1;augment: #(-1)
  )
  xarrowDouble(m_21=1/3\, m_31=2/3)
  mat(
    3, -100, 0.01, 0;0, underline(43.3333), -0.0043, 0;0, 63.6667, 99.9933, 1;augment: #(-1)
  )
  \
          &
  xarrowDouble("pivot 2, 3 row")
  mat(
    3, -100, 0.01, 0;0, underline(63.6667), 99.9933, 1;0, 43.3333, -0.0043, 0;augment: #(-1)
  )
  xarrowDouble(m_32=130/191)
  mat(
    3, -100, 0.01, 0;0, 63.6667, 99.9933, 1;0, 0, -68.4208, -0.6842;augment: #(-1)
  )
$
使用反代法，可得
$
  X=mat(0.0000, 0.0001, 0.010)^T
$
=== (ii) Gaussian elimination with scaled partial pivoting

$
  mat(A|B)&=
  mat(
    underline(2), -3, 100, 1;1, 10, -0.001, 0;3, -100, 0.01, 0;augment: #(-1)
  ) \
          &
  xarrowDouble("pivot 1, 2 row")
  mat(
    underline(1), 10, -0.001, 0;2, -3, 100, 1;3, -100, 0.01, 0;augment: #(-1)
  )
  xarrowDouble(m_21=2\, m_31=3)
  mat(
    1, 10, -0.001, 0;0, underline(-23), 100.002, 1;0, -130, 0.013, 0;augment: #(-1)
  )

  \
          &
  xarrowDouble("pivot 2, 3 row")
  mat(
    1, 10, -0.001, 0;0, underline(-130), 0.013, 0;0, -23, 100.002, 1;augment: #(-1)
  )
  xarrowDouble(m_32=23/130)
  mat(
    1, 10, -0.001, 0;0, underline(-130), 0.013, 0;0, 0, 99.9997, 1;augment: #(-1)
  )
$
使用反代法，可得
$
  X=mat(0.0000, 0.0000, 0.010)^T
$

== (b)

=== (i) Gaussian elimination with partial pivoting

$
  mat(A|B) &=
  mat(
    underline(1), 20, -1, 0.001, 0;2, -5, 30, -0.1, 1;5, 1, -100, -10, 0;2, -100, -1, 1, 0;augment: #(-1)
  )
  \
           &
  xarrowDouble("pivot 1, 3 row")
  mat(
    underline(5), 1, -100, -10, 0;2, -5, 30, -0.1, 1;1, 20, -1, 0.001, 0;2, -100, -1, 1, 0;augment: #(-1)
  )
  xarrowDouble(m_21=2/5\, m_31=1/5\, m_41=2/5)
  mat(
    5, -1, -100, -10, 0;0, -4.6, 70, 3.9, 1;0, 20.2, 19, 2.001, 0;0, -99.6, 39, 3, 0;augment: #(-1)
  )
  \
           &
  xarrowDouble("pivot 2, 4 row")
  mat(
    5, -1, -100, -10, 0;0, -99.6, 39, 3, 0;0, 20.2, 19, 2.001, 0;0, -4.6, 70, 3.9, 1;augment: #(-1)
  )
  xarrowDouble("消元")
  mat(
    5, -1, -100, -10, 0;0, -99.6, 39, 3, 0;0, 0, 26.9096, 2.6094, 0;0, 0, 68.1988, 3.7614, 1;augment: #(-1)
  )
  \
           &
  xarrowDouble("pivot 3, 4 row")
  mat(
    5, -1, -100, -10, 0;0, -99.6, 39, 3, 0;0, 0, 68.1988, 3.7614, 1;0, 0, 26.9096, 2.6094, 0;augment: #(-1)
  )
  xarrowDouble("消元")
  mat(
    5, -1, -100, -10, 0;0, -99.6, 39, 3, 0;0, 0, 68.1988, 3.7614, 1;0, 0, 0, 1.1253, -0.3946
  )
$

使用反代法，可得
$
  X=mat(-0.0207, 0.0028, 0.0340, -0.3508)^T
$
=== (ii) Gaussian elimination with scaled partial pivoting

$
  mat(A|B)&=
  mat(
    underline(1), 20, -1, 0.001, 0;2, -5, 30, -0.1, 1;5, 1, -100, -10, 0;2, -100, -1, 1, 0;augment: #(-1)
  ) \
          &
  xarrowDouble("pivot 1, 2 row")
  mat(
    2, -5, 30, -0.1, 1;1, 20, -1, 0.001, 0;5, 1, -100, -10, 0;2, -100, -1, 1, 0;augment: #(-1)
  )
  xarrowDouble("消元")
  mat(
    2, -5, 30, -0.1, 1.;0, 22.5, -16, 0.051, -0.5;0, 11.5, -175, -9.75, -2.5;0, -95., -31, -0.9, -1.;augment: #(-1)
  )
  \
          &
  xarrowDouble("消元")
  mat(
    2, -5, 30, -0.1, 1;0, 22.5, -16, 0.051, -0.5;0, 0, -166.8222, -9.7761, -2.2444;0, 0, -98.5556, -0.6847, -3.1111;augment: #(-1)
  )
  \
          &
  xarrowDouble("消元")
  mat(
    2, -5, 30., -0.1, 1;0, 22.5, -16., 0.051, -0.5;0, 0, -166.8222, -9.7761, -2.2444;0, 0, 0, 5.0909, -1.7851;augment: #(-1)
  )
$

使用反代法，可得
$
  X=mat(-0.0207, 0.0028, 0.03400, -0.3507)^T
$

#pagebreak()
= Question 5

$
  A=mat(1, 1, 0, 4;2, -1, 5, 0;5, 2, 1, 2;-3, 0, 2, 6)
$
根据按列消元的逻辑有
$
  A-mat(1;2;5;-3)mat(1, 1, 0, 4)        &=mat(0, 0, 0, 0;0, -3, 5, -8;0, -3, 1, 18;0, 3, 2, 18) eq.def A_1 \
  A_1-mat(0;1;1;-1)mat(0, -3, 5, -8)    &=mat(0, 0, 0, 0;0, 0, 0, 0;0, 0, -4, -10;0, 0, 7, 10) eq.def A_2 \
  A_2-mat(0;0;1;-7/4)mat(0, 0, -4, -10) &=mat(0, 0, 0, 0;0, 0, 0, 0;0, 0, 0, 0;0, 0, 0, -15/2) eq.def A_3 \
  A_3-mat(0;0;0;1)mat(0, 0, 0, -15/2)   &= Omicron \
$

也就是
$
  A=mat(1;2;5;-3)mat(1, 1, 0, 4)
  +mat(0;1;1;-1)mat(0, -3, 5, 8)
  +mat(0;0;1;-7/4)mat(0, 0, -4, -10)
  +mat(0;0;0;1)mat(0, 0, 0, -15/2)
$
将右侧改写成矩阵相乘形式即为
$
  L&=mat(1, 0, 0, 0;2, 1, 0, 0;5, 1, 1, 0;-3, -1, -7/4, 1) \
  U&=mat(1, 1, 0, 4;0, -3, 5, -8;0, 0, -4, -10;0, 0, 0, -15/2)
$

#pagebreak()
= Question 6

本文可以使用计算器进行计算，我使用脚本`question-6.py` 进行计算== (a) Jacobi
iteration 根据以下迭代公式：
$
  x_(k+1) &= 1/4(13-y_k+z_k) \
  y_(k+1) &= 1/5(x_k-z_k+8) \
  z_(k+1) &=1/6(2x_k-y_k+2)
$
可以计算（保留至多四位有效小数）：
$
  P_1&=(x_1,y_1,z_1)=(3.25, 1.6, 0.3333) \
  P_2&=(x_2,y_2,z_2)=(2.9333, 2.1833, 1.15) \
  P_3&=(x_3,y_3,z_3)=(2.9916, 1.9567, 0.9472)
$
迭代能收敛

== (b) Gauss-Seidel iteration
根据以下迭代公式：
$
  x_(k+1) &= 1/4(13-y_k+z_k) \
  y_(k+1) &= 1/5(x_(k+1)-z_k+8) \
  z_(k+1) &=1/6(2x_(k+1)-y_(k+1)+2)
$
可以计算（保留至多四位有效小数）：
$
  P_1&=(x_1,y_1,z_1)=(3.25, 2.25, 1.0417) \
  P_2&=(x_2,y_2,z_2)=(2.9479, 1.9812, 0.9858) \
  P_3&=(x_3,y_3,z_3)=(3.0011, 2.0030, 0.9999)
$

迭代能收敛

#pagebreak()
= Question 7
// 对题目所给的公式进行整理得到如下形式：
// $
// // 1/(x-1)A_1 +1/(x-2) A_2+1/(x-3)^2 A_3+1/(x-3)A_4+x/(x^2+1) A_5+1/(x^2+1) A_6=(x^2+x+1)/((x-1)(x-2)(x-3)^2(x^2+1))
//     & (x-2)(x-3)^2(x^2+1)A_1 + (x-1)(x-3)^2(x^2+1)A_2 +\
//     & (x-1)(x-2)(x^2+1)A_3+(x-1)(x-2)(x-3)(x^2+1)A_4 +\
//     & x(x-1)(x-2)(x-3)^2 A_5+(x-1)(x-2)(x-3)^2 A_6=x^2+x+1
// $
// 要求解未知数$AA=(A_1,A_2,A_3, A_4,A_5, A_6)$。由等式可知，这是关于$x$
//

本题核心思路为通分，然后通过合并同类项得到多项式方程，通过多项式对应系数建立方程组。

在计算中，使用`python`中的`sympy` 库进行通分、合并同类型的计算。 具体代码如下
```python
import sympy as sp
x, A1, A2, A3, A4, A5, A6 = sp.symbols("x, A1, A2,A3, A4, A5, A6")

left = (x**2 + x + 1) / ((x - 1) * (x - 2) * (x - 3) ** 2 * (x**2 + 1))

right = (
    (1 / (x - 1)) * A1
    + (1 / (x - 2)) * A2
    + (1 / (x - 3) ** 2) * A3
    + A4 / (x - 3)
    + (A5 * x + A6) / (x**2 + 1)
)
# 同时乘以分母并简化表达式
left = sp.simplify(left * ((x - 1) * (x - 2) * (x - 3) ** 2 * (x**2 + 1)))
right = sp.simplify(right * ((x - 1) * (x - 2) * (x - 3) ** 2 * (x**2 + 1)))
# 等式右边按照x的多项式进行表达
right = right.expand().collect(x)
```
结果如下：
#figure(
  image("left_right.png"), caption: [通分后的关于$x$的多项式表达式（Jupyter Notebook渲染）],
)
根据多项式对应系数相等构造关于$AA=(A_1,A_2,A_3, A_4,A_5, A_6)^T$的线性方程组，这里简单使用`sympy`的API即可

```python
A = [A1, A2, A3, A4, A5, A6]

b = [left.coeff(x, i) for i in reversed(range(6))]
E = [right.coeff(x, i) for i in reversed(range(6))]

```
#figure(
  image("equation.png"), caption: [关于$AA$的线性方程组（Jupyter Notebook渲染）],
)
转换为矩阵表达
```python
M = [[right.coeff(x, i).coeff(A[j]) for j in range(6)] for i in reversed(range(5))]

```
得到：
$
  M &=mat(
    1, 1, 0, 1, 1, 0;-8, -7, 1, -6, -9, 1;22, 16, -3, 12, 29, -9;-26, -16, 3, -12, -39, 29;21, 15, -3, 11, 18, -39;-18, -9, 2, -6, 0, 18
  ) \
  b &=mat(0,0,0,1,1,1)^T
$
也就是求解$M AA = b$，由于`Program 3-3`为Matlab代码，这里使用`sympy.solve`进行矩阵的方程组的求解

```python
result = sp.solve([e - i for e, i in zip(E, b)], A)
```
结果为
$
  AA = mat(A_1;A_2;A_3;A_4;A_5;A_6) =mat(-3/8;7/5;13/20;-203/200;-1/100;-3/100)
$
具体代码请参考`question-7.py`

#pagebreak()
= Question 8
先介绍算法实现，采用$X_0=0$作为初始化解。：
```python
def gs_iteration(A: np.ndarray, B: np.ndarray, max_iter=10000, max_residual=0.0001):
    X = np.zeros(N)  # 初始化X_0
    # 最多迭代max_iter次
    for i in range(max_iter):
        X = X.copy()  # 拷贝一份Xi
        for j in range(N):
            # 顺序更新xi (即每个未知数)
            X[j] = (
                B[j] - np.sum(A[j, :j] * X[:j]) - np.sum(A[j, j + 1 :] * X[j + 1 :])
            ) / A[j, j]
        # 所有的未知数都更新了一遍，计算误差（2-norm）
        residual = np.linalg.norm(A @ X - B)
        # 当误差小于要求值
        if max_residual > residual:
            return X

    return X
```

对于题目给出的方程组，可以发现其是对称带状矩阵，根据其特性可以用代码推算出矩阵$A$：
```python

N = 50
A = np.zeros((N, N))
for j in range(N):
    A[j][j] = 12
    if j > 0:
        A[j][j - 1] = -2
    if j > 1:
        A[j][j - 2] = 1

    if j < N - 1:
        A[j][j + 1] = -2
    if j < N - 2:
        A[j][j + 2] = 1


B = np.ones(N) * 5
```

调用`gs_iteration()`函数： 结果如下
#figure(
  image("gs_output.png"), caption: [高斯-塞德尔迭代法在给定误差/最大迭代次数下的$X$结果],
)
