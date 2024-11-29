#import "template.typ": *
#import "@preview/xarrow:0.3.0": *

#set page(numbering: "1", paper: "a4")
#set math.mat(delim: "[")
#set math.mat(gap: 0.5em)
#init("Homework 5", "Abao Zhang (张宝)", student_number: "12332459")

= Question 1
最小二乘法的目标是找到直线$y=A x+B$ 使得所有数据点$(x_i, y_i)$ 到直线之间的误差平方和最小化。误差平方和表示为

$
  S(A, B) = sum_(k=1)^N ((A x_k + B) - y_k )^2
$
对$A$和$B$ 分别求偏导数：
$
  frac(partial S, partial A) &= 2 sum_(k=1)^N x_k (A x_k + B - y_k) \
  frac(partial S, partial B) &= 2 sum_(k=1)^N (A x_k + B - y_k)
$

分别另两者等于0，可以联立方程组

$
  2 sum_(k=1)^N x_k (A x_k + B - y_k) &= 0\
  2 sum_(k=1)^N (A x_k + B - y_k) &= 0
$

分别展开
$
  A sum_(k=1)^N x_k^2 +B sum_(k=1)^N x_k - sum_(k=1)^N x_k y_k &= 0\
  A sum_(k=1)^N x_k + N B - sum_(k=1)^N y_k &= 0
$

使用$N accent(x, macron)=sum_(k=1)^N x_k$，$N accent(y, macron)=sum_(k=1)^N y_k$ 替换，可以简化为
$
  A sum_(k=1)^N x_k^2 + N B accent(x, macron)- sum_(k=1)^N x_k y_k &= 0\
  A accent(x, macron) + B - accent(y, macron) &= 0
$
消元法解方程组：根据第二个等式可以得到$B=accent(y, macron) - A accent(x, macron)$，代入等式一，解得
$
  A = frac(sum_(k=1)^N (x_k-accent(x, macron))(y_k -accent(y, macron)),
  sum_(k=1)^N (x_k-accent(x, macron))^2)
$
命题得证，即
$
  C =sum_(k=1)^N (x_k-accent(x, macron))^2,
  space space
  A = frac(sum_(k=1)^N (x_k-accent(x, macron))(y_k -accent(y, macron)), C),
  space space
  B=accent(y, macron) - A accent(x, macron)
$

= Question 2

== (a)
$
  S(A) = sum_(k=1)^N (A x_k - y_k)^2 = sum_(k=1)^N (A^2 x_k^2 - 2 A x_k y_k + y_k^2)
$

对$A$求导数，并令倒数为0
$
  frac(d S, d A) = 2 sum_(k=1)^N (A x_k^2 - x_k y_k) = 0
$
解得
$
  A= frac(sum_(k=1)^N x_k y_k, sum_(k=1)^N x_k^2)
$

== (b)
$
  S(A) = sum_(k=1)^N (A x_k^2 - y_k)^2 = sum_(k=1)^N (A^2 x_k^4 - 2 A x_k^2 y_k + y_k^2)
$
对$A$求导数，并令倒数为0
$
  frac(d S, d A) = 2 sum_(k=1)^N (A x_k^4 - x_k^2 y_k) = 0
$
解得
$
  A= frac(sum_(k=1)^N x_k^2 y_k, sum_(k=1)^N x_k^4)
$
== (c)
$
  S(A, B) =sum_(k=1)^N (A x_k^2+B-y_k)^2 = sum_(k=1)^N (A^2 x_k^4 + B^2+ y_k^2 + 2 A B x_k^2 - 2A x_k^2 y_k -2 B y_k)
$
对$A$和$B$ 分别求偏导数，并分别令偏导数为0，得到方程组如下：
$
  frac(partial S, partial A) &= 2 sum_(k=1)^N x_k^2 (A x_k^2 + B - y_k)=0 \
  frac(partial S, partial B) &= 2 sum_(k=1)^N (A x_k^2 + B - y_k)=0
$

解二元一次方程组，可得
$
  A &= frac(N sum_i^N x_i^2y - N accent(y, macron) sum_i^N x_i^2, N sum_i^N x_i^4 - (sum_i^N x_i^2)^2) \
  B&=accent(y, macron) -frac( A sum_i^N x_i^2, N)
$


= Question 3
本题解题目思路为使用变量变换将数据变化，然后拟合线性方程$y=A x+B$，然后将系数$A$、$B$变换回原方程。
手动计算较为复杂，这里我使用了代码进行计算，可实际运行`python-3.py`查看结果。注意：#text(fill: red, [我仅使用numpy表示数组，方便计算，并没有直接调库求解])。下面给出部分计算逻辑代码。

首先是均方根误差$E_2$
```python
def E2(table, f):
    """计算均方根误差
    Args:
        table     [N, 2] 表示(xi, yi)
        f    fun: 函数
    """
    X, Y = table[:, 0], table[:, 1]
    f = np.vectorize(f)
    return np.sqrt(np.sum(np.subtract(f(X), Y) ** 2) / len(X))
```
是使用最小二乘法拟合线性方程$y=A x +B$，我们需要计算系数$A$、$B$，这里使用的公式为第一问中得到的公式。
```python
def solve(X, Y) -> Tuple[float, float]:
    """使用最小二乘法进行线性拟合，得到A和B
    Args:
        X    [N]:   数组，N个元素
        Y    [N]:   数组，N个元素
    """
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    C = np.sum((X - x_mean) ** 2)
    A = np.sum((X - x_mean) * (Y - y_mean)) / C
    B = y_mean - A * x_mean
    return A, B
```
下面给出最终的求解函数
```python
def resolve(table, input_mapper, coeff_mapper):
    """
    将数据按照x_mapper，y_mapper后，返回原始系数
    Args:
        table     [N, 2]
        input_mapper (x_mapper, y_mapper) 数据mapper
            x_mapper (fun(x):X): 映射函数由x到X
            y_mapper (fun(y):Y): 隐射函数由y到Y
        coeff_mapper (A_mapper, B_mapper) 系数mapper
            A_mapper (fun(A):raw_A): 映射系数A回到原参数
            B_mapper (fun(B):raw_B): 隐射系数B回到原参数
    """
    x_mapper, y_mapper = input_mapper
    x_mapper, y_mapper = np.vectorize(x_mapper), np.vectorize(y_mapper)
    X, Y = table[:, 0], table[:, 1]
    A, B = solve(x_mapper(X), y_mapper(Y))
    A_mapper, B_mapper = coeff_mapper
    return A_mapper(A), B_mapper(B)
```
调用求解函数，即可得最终结果。以问题（a）为例，下图中还包含了画图代码。
```python
A, C = resolve(tab1, [lambda x: x, np.log], [do_nothing, np.exp])
def f(x):
    return C * np.exp(A * x)
e2 = E2(tab1, f)
plot(tab1, f)
print("(a) i")
print(f" A={A}, C={C}, E2={e2}")
```
== (a)
=== i
$
  A&=-1.058567340022129 ,space space
  C&=2.399508181612176 ,space space
  E_2&=3.4097854679724233 ,space space
$
=== ii
$
  A&=-1.058567340022129 ,space space
  C&=2.399508181612176 ,space space
  E_2&=3.4097854679724233
$

== (b)
=== i

$
  A&=0.7573257893549833 ,space space
  B&=0.7845232804008844 ,space space
  E_2&=669.2218637833518
$
=== ii

$
  A&=0.5776582870479143 ,space space
  B&=0.8498769941596073 ,space space
  E_2&=0.07767026951123651
$

== (c)
基于上面结果，基于$E_2(f)$，我们可以发现：
- 对于数据表1，曲线$f(x)=C e^(A x)$更好
- 对于数据表2，曲线$f(x)=(A x + B)^(-2)$更好

从下表格中的图我们也可以印证

#table(
  columns: (0.6fr, 1fr, 1fr),
  inset: (1pt, 5pt),
  align: horizon + center,
  table.header(
    [],
    [表格1],
    [表格2],
  ),

  [$f(x)=C e^(A x)$], image("./figure-a-i.png"), image("./figure-a-ii.png"),
  [$f(x)=(A x + B)^(-2)$],
  image("./figure-b-i.png"),
  image("./figure-b-ii.png"),
)


#pagebreak()


= Question 4

首先将$E$进行展开，方便后续计算

$
  E(A, B, C)= sum_(k=1)^N (& A^2 x_k^2 + B^2 y_k^2 + C^2 + z_k^2 \
    &+ 2 A B x_k y_k + 2A C x_k - 2 A x_k z_k +2 B C y_k -2B y_k z_k - 2 C z_k)
$
将$E$分别对$A$、$B$、$C$求偏导。

$
  frac(partial E, partial A)&= 2A sum_(k=1)^N x_k^2 + 2B sum_(k=1)^N x_k y_k + 2C sum_(k=1)^N x_k -2 sum_(k=1)^N x_k z_k \
  frac(partial E, partial A) &= 2 B sum_(k=1)^N y_k^2 + 2 A sum_(k=1)^N x_k y_k +C sum_(k=1)^N y_k -2 sum_(k=1)^N y_k z_k \
  frac(partial E, partial A) &= 2 N C + 2 A sum_(k=1)^N x_k + 2B sum_(k=1)^N y_k - 2 sum_(k=1)^N z_k
$
分别另其等于0，可得到方程组
$
   2 A sum_(k=1)^N x_k^2 + 2B sum_(k=1)^N x_k y_k + 2C sum_(k=1)^N x_k -2 sum_(k=1)^N x_k z_k = 0\
   2 B sum_(k=1)^N y_k^2 + 2 A sum_(k=1)^N x_k y_k +C sum_(k=1)^N y_k -2 sum_(k=1)^N y_k z_k = 0\
   2 N C + 2 A sum_(k=1)^N x_k + 2B sum_(k=1)^N y_k - 2 sum_(k=1)^N z_k = 0
$

整理可得
$
    A sum_(k=1)^N x_k^2 + B sum_(k=1)^N x_k y_k + C sum_(k=1)^N x_k &= sum_(k=1)^N x_k z_k \
    A sum_(k=1)^N x_k y_k + B sum_(k=1)^N y_k^2 +  C sum_(k=1)^N y_k &= sum_(k=1)^N y_k z_k \
     A sum_(k=1)^N x_k + B sum_(k=1)^N y_k + N C&=  sum_(k=1)^N z_k 
$

命题得证

#pagebreak()

= Question 5
假设三段曲线由系数$c={c_0, c_1, ..., c_11}$ 进行表征，具体如下

$
f_0(x) &= c_0 + c_1 x+ c_2 x^2 + c_3 x^3, space x in (-3, -2) \
f_1(x) &= c_4 + c_5 x+ c_6 x^2 + c_7 x^3, space x in (-2, 1) \
f_2(x) &= c_8 + c_9 x+ c_10 x^2 + c_11 x^3, space x in (1, 4)
$
求一阶导数
$
f_0 '(x) &= c_1 + 2c_2 x + 3 c_3 x^2, space x in (-3, -2) \
f_1 '(x) &= c_5 + 2c_6 x + 3 c_7 x^2, space x in (-2, 1) \
f_2 '(x) &= c_9 + 2c_10 x +3  c_11 x^2, space x in (1, 4)
$
求二阶导数
$
f_0 ''(x) &= 2 c_2  +  6 c_3 x, space x in (-3, -2) \
f_1 ''(x) &= 2 c_6  +  6 c_7 x, space x in (-2, 1) \
f_2 ''(x) &= 2 c_10  + 6  c_11 x, space x in (1, 4)
$
根据通过给定端点、内部端点二阶导连续、给定边界条件，进行代入，即可构建包含12个未知数、12个方程的方程组，显然，给定的是克拉姆边界条件，方程是有唯一解的
$
f_1(-3) &= 2 \
f_1(-2) &= 0 \
f_2(-2) &= 0 \
f_2(1) &= 3 \
f_3(1) &= 3 \
f_3(4) &= 1 \
f_1 '(-2) &= f_2 '(-2) \
f_1 ''(-2) &= f_2 ''(-2) \
f_2 '(1) &= f_2 '(1) \
f_2 ''(1) &= f_2 ''(1) \
f_1 '(-3) &= -1 \
f_3 '(4) &= 1 \
$


手动计算耗费量巨大，且无意义，这里我使用编程方法进行$A c=B$的计算，这里的$A$也就是由方程组得到的矩阵，$c$即为我们要求的系数，B也就是对于方程组等号右边的数值。下面分别介绍$A$、$B$是如何构造的。

对于任意给定的端点数组$"points"$ (2d数组，shape=[N, 2])和起始点一阶导$"start_der"$ 和最后一个点的一阶导$"end_der"$，第一步我们简单获取参数信息：
```python
point_num = len(points)
interval_num = point_num - 1
coeff_num = (point_num - 1) * 4

```

#pagebreak()
接下来我们定义三个函数，分别返回长度为未知数个数($"coeff_num"$)的数组，用于后续矩阵$A$的构建

```python
def f_i(i, x):
    """f_i(x) = y 对应的系数，即Ac =b的某一行
    Args:
        i (int): 第i条曲线
    """
    row = np.zeros((coeff_num))
    start = i * 4
    row[start : start + 4] = [1, x, x**2, x**3]
    return row
def fd_i(i, x):
    """一阶倒数系数"""
    row = np.zeros(coeff_num)
    start = i * 4
    row[start : start + 4] = [0, 1, 2 * x, 3 * x**2]
    return row
def fdd_i(i, x):
    """二阶倒数系数"""
    row = np.zeros(coeff_num)
    start = i * 4
    row[start : start + 4] = [0, 0, 2, 6 * x]
    return row
```
对于$f_i(x)=y$的方程，我可以得到将如下方程系数和$b$加入到我们的$A$和$b$中
```python
A, b = [], []
for i, (x, y) in enumerate(points):
    if i != point_num - 1:
        print(f"point{i} 通过 spine{i}")
        A.append(f_i(i, x))
        b.append(y)
    if i != 0:
        print(f"point{i} 通过 spine{i-1}")
        A.append(f_i(i - 1, x))
        b.append(y)
```
对于中间端点一阶导数和二阶导数连续（相等）的方程，我们使用如下代码，注意，我们这里将$f_(i-1) '(x)=f_i '(x)$ 转换为了$f_(i-1) '(x)-f_i '(x)=0$，二阶导数同理

```python
for i in range(1, point_num - 1):
    (x, y) = points[i]
    print(f"point{i} spine{i-1} 和 spine{i} 一阶导数连续（相等）")
    A.append(fd_i(i - 1, x) - fd_i(i, x))
    b.append(0)
    print(f"point{i} spine{i-1} 和 spine{i} 二阶导数连续（相等）")
    A.append(fdd_i(i - 1, x) - fdd_i(i, x))
    b.append(0)
```


两个边界条件对应的两个方程
```python
A.append(fd_i(0, points[0, 0]))
b.append(start_der)
A.append(fd_i(interval_num - 1, points[-1, 0]))
b.append(end_der)
```
最后，我们获得了$A$ 和$B$，只需要解方程组就可以求到所有的$c_i$，我这里使用的使用我在作业三-题目四中手撸的`gs_partial_scaled_pivot(A, B)`方法。

我还编写了最终的曲线函数，对于给定$x$都可以计算响应的值
```python
def S(x):
    if x < points[0, 0] and x > points[-1, 0]:
        assert False, "out of range"
    for i in range(interval_num):
        if x <= points[i + 1, 0]:
            return np.sum(C[i * 4: i * 4 + 4] @ [1, x, x**2, x**3])

```
最后我给出所有的系数
$

c_0=16.12903226, c_1=23.48387097, c_2= 10.61290323, c_3=1.4516129 \
c_4=1.70609319, c_5=1.84946237, c_6=-0.20430108, c_7= -0.35125448 \
c_8=1.0525687,   c_9=3.81003584, c_10= -2.16487455, c_11= 0.30227001
$

即对应
$
f_1(x)=16.12903226+ 23.48387097 x+   10.61290323 x^2+  1.4516129 x^3 \
f_2(x)=1.70609319+ 1.84946237 x  -0.20430108 x^2    -0.35125448 x^3 \
f_3(x)=1.0525687+   3.81003584 x   -2.16487455 x^2+   0.30227001 x^3 

$
最后，我绘制了曲线进行可视化，具体代码可以参考`question-5.py`
#figure(
  image("./figure-cubic-spine.png", width: 100%),
    caption: [可视化曲线]
)


= Question 6
代码从question 5 修改得到，代码可以查看`question-6.py`具体有以下不同

#strike[
```python
A.append(fd_i(0, points[0, 0]))
b.append(start_der)
A.append(fd_i(interval_num - 1, points[-1, 0]))
b.append(end_der)
```
]
```python
A.append(fdd_i(0, points[0, 0]))
b.append((points[1, 1] - 2 * points[0, 1]) / ((points[1, 0] - points[0, 0]) ** 2))
A.append(fdd_i(interval_num - 1, points[-1, 0]))
b.append(
    (points[-2, 1] - 2 * points[-1, 1]) / ((points[-2, 0] - points[-1, 0]) ** 2)
)
```
这里使用公式
$
f''(x_0) approx frac(y_1- 2 y_0, (x_1-x_0)^2)
$
作为边界二阶导数的近似。具体结果如下
$

f_1(x) &=13.563218390804604 + 20.02681992337165 x + 9.086206896551724 x^2 + 1.2318007662835249 x^3 \
f_2(x) &=1.4423158790974888 + 1.845466155810983 x + -0.00446998722860803 x^2 + -0.2833120476798638 x^3 \
f_3(x) &=1.0578969774372076 + 2.998722860791826 x + -1.157726692209451 x^2 + 0.1011068539804172 x^3
$

#figure(
  image("./figure-extra.png", width: 100%),
    caption: [可视化曲线]
) <label>

= Question 7

== (a)
对于$f(x)=x^3-x$，求其一阶导、二阶导数分别为
$
f'(x) & = 3 x^2-1 \
f''(x) & = 6 x \
$
可以发现$f(x)$、$f'(x)$ 和$f''(x)$在$x_0=-2$和$x_1=0$处均连续，所以$f(x)$可以表示区间$[-2, 0]$上的三次样条曲线
== (b)

同样的，$f(x)$、$f'(x)$ 和$f''(x)$在$x_0=-2$、$x_1=0$和$x_2=2$处均连续，命题得证
== (c)

对于任意三次多项式$f(x)=a_0+a_1 x + a_2 x^2 +a_3 x^3$ ，在闭区间$[a, b]$是是否可以作为对应的样条曲线取决于其在端点$x=a$和$x=b$上是否存在一阶导数、二阶导数并连续，
具体而言就是下面$f'(x),f''(x)$存在，且在a和b上连续
$
f'(x)&= a_1 + 2 a_2 x +3 a_3 x^2 \
f''(x)&=  2 a_2  +6 a_3 x \
$


= Question 8
首先，根据傅里叶级数的定义，对于一个周期为 $2 P$的周期函数 $f(x)$，其傅里叶级数展开


= Question 9
对于贝塞尔曲线 $P(t)$，我们有一组控制点 $P_0, P_1, dot, P_N$ 和贝塞尔曲线的参数方程：

$
P(t) = sum_( i=0 )^N B_(i, N)(t) P_i
$

其中，$B_( i,N )(t)$ 是贝塞尔基函数，定义为：

$
B_( i,N )(t) = binom(N, i) t^i (1-t)^( N-i )
$

贝塞尔曲线的一阶导数和二阶导数分别是：

$
frac(d P, d t) &= sum_(i=0)^N frac(d B, d t) P_i \
frac(d^2 P, d t^2) &= sum_(i=0)^N frac(d^2 B, d t^2) P_i

$
// 其基函数对应的一阶导数和二阶导数分别是：
// $
//
// frac(d B_(i,N),d t)  = binom( N, i ) ( i t^( i-1 ) (1-t)^( N-i ) - (N-i) t^i (1-t)^( N-i-1 ) ) \
//  
//    frac( d^2 B_( i,N ),d t^2)  = binom( N,i ) ( i (i-1) t^( i-2 ) (1-t)^( N-i ) - 2i (N-i) t^( i-1 ) (1-t)^( N-i-1 ) + (N-i)(N-i-1) t^i (1-t)^( N-i-2 ) )
// $
在t=0时，只有控制点$P_0, P_1, P_2$对其有贡献，其他项均为0，
$
P''(0) = N (N-1) P_0 -2 N (N-1) P_1+N (N-1) P_2 = N(N-1)(P_2-2P_1+P_0)
$
同理在t=1时，只有控制点$P_N, P_(N-1), P_(N-2)$对其有贡献，其他项均为0，
$
P''(1) =  N(N-1)(P_N-2P_(N-1)+P_(N-2))

$
命题得证


= Question 10
这里调用第5问中手撸的程序即可，这里不过多赘述，具体可以查看`question-10.py`，下面给出调用的代码和
```python
p = np.array(
    [
        [0, 0],
        [2, 40],
        [4, 160],
        [6, 300],
        [8, 480],
    ]
)
model = solve_cubic_coeff(p, 0, 98)

x_vals = np.linspace(0, 8, 100)

plt.plot(x_vals, model(x_vals), label="Time-Distance")
plt.scatter(p[:, 0], p[:, 1], color="red", label="data points")
plt.legend()
plt.show()

```
下面函数的对应区间不言自明
$
f_0(x)&=0.0 + 0.0 x + 8.374999999999993 x^2 + 0.8125000000000023 x^3 \
f_1(x)&=26.000000000000043 + -39.00000000000006 x + 27.87500000000002 x^2 + -2.437500000000002 x^3 \
f_2(x)&=-222.00000000000023 + 147.00000000000014 x + -18.62500000000003 x^2 + 1.4375000000000018 x^3 \
f_3(x)&=263.9999999999999 + -95.99999999999994 x + 21.874999999999986 x^2 + -0.812499999999999 x^3
$

#figure(
  image("./figure-time-distance.png", width: 80%),
    caption: [时间-距离曲线]
) <label>
#pagebreak()

= Question 11
== (a) 自然边界条件
即$S''(x_1)=0$ 和$S''(x_n)=0$
在第5问中手撸的程序中稍微修改最后两个方程（边界条件）即可，下面给出 需要修改的部分代码，完整版本可以查看`question-11-a.py`

#strike[
```python

A.append(fd_i(0, points[0, 0]))
b.append(start_der)
A.append(fd_i(interval_num - 1, points[-1, 0]))
b.append(end_der)
```
]
```python
A.append(fdd_i(0, points[0, 0]))
b.append(0)
A.append(fdd_i(interval_num - 1, points[-1, 0]))
b.append(0)
```
#figure(
  image("./figure-11-a.png", width: 80%),
    caption: [时间-距离曲线]
)

#pagebreak()
== (b) 外推边界条件
跟题目6一样，下面仅给出结果
#figure(
  image("./figure-11-b.png", width: 80%),
    caption: [时间-距离曲线]
)
