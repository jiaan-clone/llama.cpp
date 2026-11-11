# llama.cpp 学习计划数学参考

适用读者: 已学习本科高等数学、线性代数、概率论, 希望复习这些知识在 llama.cpp 推理中的具体用途。

对应范围: [8 周学习总计划](llama-cpp-8-week-learning-plan.md)、[Transformer 概念桥接](000-transformer-concept-bridge.md)及各周学习路线。本文重点是公式、shape、数值含义与工程对应关系。源码相关细节按本地版本 `23d38c44a` 核对。

## 目录

- [1. 阅读方式与符号](#1-阅读方式与符号)
- [2. 向量、矩阵与张量布局](#2-向量矩阵与张量布局)
- [3. 函数、均方根与归一化](#3-函数均方根与归一化)
- [4. Attention 的完整数学过程](#4-attention-的完整数学过程)
- [5. RoPE 与二维旋转](#5-rope-与二维旋转)
- [6. FFN、残差与完整前向计算](#6-ffn残差与完整前向计算)
- [7. 自回归概率与采样](#7-自回归概率与采样)
- [8. 熵、交叉熵与困惑度](#8-熵交叉熵与困惑度)
- [9. 浮点数、量化与误差](#9-浮点数量化与误差)
- [10. 存储容量、索引与状态共享](#10-存储容量索引与状态共享)
- [11. 计算量、带宽与性能统计](#11-计算量带宽与性能统计)
- [12. 微积分的使用边界](#12-微积分的使用边界)
- [13. 按周复习与自测](#13-按周复习与自测)
- [14. 文档与源码入口](#14-文档与源码入口)

## 1. 阅读方式与符号

### 1.1 数学范围与学习深度

| 领域 | 本计划使用的知识 | 学习深度 |
| --- | --- | --- |
| 线性代数 | 点积、矩阵乘法、转置、线性映射、逐元素运算、范数 | 主线必会, 能手算小例子并追踪 shape |
| 初等函数 | 指数、对数、平方根、三角函数、函数复合 | 主线必会, 能解释数值变化 |
| 概率论 | 离散分布、条件概率、期望、方差、随机抽样 | 理解 Attention scaling 和采样 |
| 信息论 | 熵、负对数概率、交叉熵、困惑度 | 熵为第 6 周必做; 困惑度为第 7 周选做 |
| 数值计算 | 浮点精度、舍入、数值稳定性、量化误差 | 第 6-7 周重点 |
| 描述统计 | 均值、标准差、分位数、重复测量 | 第 7-8 周实验分析 |
| 离散结构与数量估算 | 集合、索引、依赖图、计数、容量与复杂度 | 第 4-5 周和第 7-8 周工程分析 |
| 微积分 | 极限、导数和链式法则的直觉 | 辅助理解; 主线不要求训练和反向传播推导 |

你已有的本科课程基础足以开始。本计划最需要重新熟悉的是矩阵运算的轴语义, 以及概率公式怎样作用于真实候选数组。

阅读每个公式时回答四个问题: 输入是什么, 沿哪个轴计算, 输出 shape 是什么, 结果供谁使用。

### 1.2 统一符号

| 符号 | 含义 |
| --- | --- |
| $B$ | 纸面模型中等长序列的 batch size |
| $T$ | 纸面模型中的序列长度 |
| $N$ | 本次处理的输入 token 数; 单序列分析时为当前 query 数 |
| $S$ | 当前 attention 包含的历史与当前 K/V 位置总数; 每个 query 还受 mask 约束 |
| $D$ | hidden size |
| $H$ | Query head 数 |
| $K_v$ | Key/Value head 数, 对应计划中的 `KvH` |
| $d_h$ | 每个 head 的维度, 对应 `Dh`; 本文普通 Llama 例子取 $D=Hd_h$ |
| $I$ | FFN 中间维度 |
| $V_{\mathrm{vocab}}$ | 词表大小; 与 Attention 的 Value 矩阵 $V$ 区分 |
| $L$ | Transformer block 层数 |
| $O$ | 请求输出 logits 的位置数 |
| $\tau$ | 采样温度, 避免与序列长度 $T$ 混淆 |
| $C$ | KV cache 分配的物理 cell 数 |
| $\odot$ | 逐元素乘法 |
| $\mathcal H(p)$ | 分布的熵, 避免与 head 数 $H$ 混淆 |

默认使用 token 在前、特征在后的逻辑 shape, 线性层权重写成 `[out,in]`, 因而投影为 $Y=XW^\top$。概念桥接中也使用过 $Y=XW$, 那时权重定义为 `[in,out]`; 两种约定都成立, 不能混用。

本文多数推导固定 $B=1$。实际 `llama_batch` 可以混合多个 sequence 和不同长度, 不能总把它看成规则的 `[B,T]` 数组。

## 2. 向量、矩阵与张量布局

对应第 1-3 周、第 7 周。

### 2.1 点积、范数和加权求和

对 $x,y\in\mathbb R^d$:

$$
x^\top y=\sum_{i=1}^{d}x_i y_i,\qquad
\|x\|_2=\sqrt{\sum_{i=1}^{d}x_i^2}.
$$

点积输出一个标量, 会同时受到向量方向和长度影响。Attention 使用点积分数, 并不默认把 Q/K 归一化为单位向量, 因此不能直接把它称为余弦相似度。

给定向量 $v_1,\ldots,v_S$ 和权重 $p_j$, 加权求和为:

$$
y=\sum_{j=1}^{S}p_jv_j.
$$

若 $p_j\geq0$ 且 $\sum_jp_j=1$, 这是凸组合, 也可看成随机下标 $J\sim p$ 时向量 $v_J$ 的期望。Attention 实际计算这个加权和, 并不需要先随机抽取一个位置。

例如:

```text
0.25 * [1,2] + 0.75 * [3,4] = [2.5,3.5]
```

### 2.2 矩阵乘法与转置

若 $A\in\mathbb R^{m\times k}$、$B\in\mathbb R^{k\times n}$, 则:

$$
C=AB\in\mathbb R^{m\times n},\qquad
C_{ij}=\sum_{r=1}^{k}A_{ir}B_{rj}.
$$

内维度 $k$ 必须一致。转置交换矩阵的两个轴, 且:

$$
(AB)^\top=B^\top A^\top.
$$

普通投影:

$$
X\in\mathbb R^{N\times D_{\mathrm{in}}},\quad
W\in\mathbb R^{D_{\mathrm{out}}\times D_{\mathrm{in}}},\quad
Y=XW^\top\in\mathbb R^{N\times D_{\mathrm{out}}}.
$$

第 3 周的小例子:

```text
X = [[1,2,3], [4,5,6]]
W = [[1,0,1], [0,1,-1]]
X W^T = [[4,-1], [10,-1]]
```

工程中的“线性投影”通常指学习到的线性映射, 不要求权重满足正交投影矩阵的条件。

### 2.3 逐元素运算与广播

同 shape 张量的逐元素乘法为:

$$
(A\odot B)_{ij}=A_{ij}B_{ij}.
$$

它不对中间维度求和, 与矩阵乘法含义不同。

广播的常见情况是将 `[D]` 参数作用于 `[N,D]` 的每一行:

$$
Y_{nd}=X_{nd}\gamma_d.
$$

RMSNorm 的缩放参数就是这种用法。广播是轴上的复用规则, 不一定真的复制参数数据; 在 GGML 中仍需检查具体 op 的 shape 条件。

### 2.4 Reshape、轴交换与 stride

| 操作 | 数学或逻辑含义 | 要检查的事实 |
| --- | --- | --- |
| reshape | 按新的轴长度解释同一批元素 | 元素总数不变, 元素到新索引的对应正确 |
| transpose / permute | 改变轴的顺序 | shape 和 stride 都可能改变 |
| view | 使用原存储的一个视图 | 起始偏移、轴长度和 stride |
| contiguous copy | 按目标连续布局整理数据 | 是否产生复制以及额外 buffer |

例如, `[N,H*d_h]` 可 reshape 为 `[N,H,d_h]`, 然后交换轴得到 `[H,N,d_h]`。只把数字重新分组, 不等于已经交换了 token 轴和 head 轴。

GGML 的普通二维投影对应:

```text
逻辑: X [N,Din], W [Dout,Din], Y [N,Dout]
GGML: X ne=[Din,N], W ne=[Din,Dout]
      ggml_mul_mat(ctx,W,X) 的结果 ne=[Dout,N]
```

`ne[i]` 是轴长度, `nb[i]` 是字节 stride。对普通非量化 tensor, 给定 GGML 索引 $(i_0,i_1,i_2,i_3)$, 相对数据起点的字节偏移为:

$$
\mathrm{offset}=\sum_{r=0}^{3}i_r\,nb[r].
$$

连续 F32 tensor 通常有 `nb[0]=4`, `nb[1]=4*ne[0]`。量化 tensor 按 block 编码, 不能把这个标量寻址公式原样套到 block 内的每个权重上。

## 3. 函数、均方根与归一化

对应第 1-3 周、第 6 周。

### 3.1 指数、对数与求和

常用性质:

$$
e^{a+b}=e^ae^b,\quad e^0=1,\quad
\ln(ab)=\ln a+\ln b,\quad \ln(e^x)=x.
$$

自然对数 $\ln$ 的底是 $e$。概率连乘常转成对数求和, 可避免很多小概率相乘导致的数值下溢。

Mask 中写 $e^{-\infty}=0$, 对应的数学含义是:

$$
\lim_{x\to-\infty}e^x=0.
$$

### 3.2 均值、均方、方差和均方根

对 $x\in\mathbb R^D$:

$$
\mu=\frac1D\sum_i x_i,\qquad
m_2=\frac1D\sum_i x_i^2,\qquad
\sigma^2=\frac1D\sum_i(x_i-\mu)^2=m_2-\mu^2.
$$

$$
\mathrm{RMS}(x)=\sqrt{m_2}=\frac{\|x\|_2}{\sqrt D}.
$$

这里是一个向量内部的统计量, 方差分母为 $D$。它与估计总体方差时使用分母 $n-1$ 的样本方差用途不同。

### 3.3 RMSNorm 与 LayerNorm

RMSNorm 对每个 token 的特征轴计算:

$$
y_i=\gamma_i\frac{x_i}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}.
$$

$\gamma$ 是学习到的逐维缩放参数, $\epsilon>0$ 用于避免分母过小。输入和输出 shape 都是 `[N,D]`。

作为对照, 常见 LayerNorm 为:

$$
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

| 比较项 | RMSNorm | LayerNorm |
| --- | --- | --- |
| 是否减均值 | 否 | 是 |
| 分母来自什么 | 均方 | 中心化后的方差 |
| 归一化轴 | 此处是单 token 的特征轴 | 此处是单 token 的特征轴 |

忽略 $\epsilon$ 且令 $\gamma_i=1$, `[3,4]` 的 RMS 为 $\sqrt{12.5}$, 归一化结果约为 `[0.8485,1.1314]`, 结果的均方为 1。

不要把 RMSNorm 理解成“输出概率和为 1”。除以公共正标量本身保留向量方向, 但后续不同维度的 $\gamma_i$ 可以改变方向; 带 $\gamma$ 和 $\epsilon$ 的完整输出也不保证 RMS 恰好为 1。

### 3.4 Softmax 及其稳定计算

对一组 logits 或 attention scores $z$:

$$
p_i=\mathrm{softmax}(z)_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}.
$$

有限输入的输出满足 $p_i>0$、$\sum_i p_i=1$, 且保持大小顺序。相同平移不改变结果:

$$
\mathrm{softmax}(z+c\mathbf1)=\mathrm{softmax}(z).
$$

令 $m=\max_i z_i$, 可使用数值更稳定的等价形式:

$$
p_i=\frac{\exp(z_i-m)}{\sum_j\exp(z_j-m)}.
$$

这样指数的自变量不大于 0, 可避免大正数指数溢出。例如 `[1000,1001]` 可先转换为 `[-1,0]`。

小例子:

```text
softmax([0,ln(3)]) = [1/4,3/4]
softmax([2,-inf])  = [1,0]
```

这里要求至少有一个有限的有效分数。全部候选均为 `-inf` 时没有可归一化的分布; NaN、正无穷也应单独处理。不能在这类输入上直接使用减最大值公式。

必须标明归一化轴: Attention 沿 key 位置轴做 softmax; 输出采样沿词表候选轴做 softmax。

## 4. Attention 的完整数学过程

对应第 1 周 Day 3、第 3 周。

### 4.1 Q/K/V 投影

设归一化后的 hidden state 为 $X_n\in\mathbb R^{N\times D}$:

$$
Q=X_nW_Q^\top,\qquad
K_{\mathrm{cur}}=X_nW_K^\top,\qquad
V_{\mathrm{cur}}=X_nW_V^\top.
$$

普通 GQA 配置下:

| 张量 | 逻辑 shape |
| --- | --- |
| $W_Q$ | `[H*d_h,D]` |
| $W_K,W_V$ | `[K_v*d_h,D]` |
| 投影后的 Q | `[N,H*d_h]` |
| 投影后的当前 K/V | `[N,K_v*d_h]` |
| 拆 head 后的 Q | `[H,N,d_h]` |
| 写入 cache 并读取历史后的 K/V | `[K_v,S,d_h]` |

Q、K、V 使用不同权重, 都由 hidden state 投影得到。V 不由 $QK^\top$ 计算出来。这里暂省略模型可能具有的 bias。

### 4.2 单个 head 的计算

对选定 head, 令 $Q\in\mathbb R^{N\times d_h}$、$K,V\in\mathbb R^{S\times d_h}$:

$$
A=\frac{QK^\top}{\sqrt{d_h}}+M,\qquad
P=\mathrm{softmax}_{\mathrm{key}}(A),\qquad
Y=PV.
$$

| 步骤 | 输出 shape | 元素含义 |
| --- | --- | --- |
| $QK^\top$ | `[N,S]` | 每个 query 对每个 key 的匹配分数 |
| scale + mask | `[N,S]` | 缩放后的分数及可见性限制 |
| softmax | `[N,S]` | 每行对有效 key 位置的权重 |
| $PV$ | `[N,d_h]` | 按权重混合历史和当前 Value |

每个位置输出向量的第 $r$ 个分量为:

$$
Y_{ir}=\sum_{j=1}^{S}P_{ij}V_{jr}.
$$

### 4.3 为什么除以 $\sqrt{d_h}$

在一个便于分析的假设下, 各分量零均值、单位方差, Q/K 分量相互独立, 各维乘积之间也独立:

$$
s=\sum_{r=1}^{d_h}q_rk_r,\qquad
\operatorname{Var}(s)=d_h.
$$

因此:

$$
\operatorname{Var}\left(\frac{s}{\sqrt{d_h}}\right)=1.
$$

这说明缩放可以控制分数随维度增长的波动, 避免 softmax 容易变得过于尖锐。真实训练后的 Q/K 不严格满足这些假设, 该推导用于说明设计动机, 不是对每层实测方差的保证。

### 4.4 Causal mask 与位置

对从位置 0 开始、没有历史 cache 的单序列:

$$
M_{ij}=
\begin{cases}
0,&j\leq i,\\
-\infty,&j>i.
\end{cases}
$$

两个 token 时:

$$
M=\begin{bmatrix}0&-\infty\\0&0\end{bmatrix}.
$$

行表示 query, 列表示 key。禁止位置的权重为 0, 不能简单把该位置原始 score 改成 0, 因为 $\exp(0)=1$。

有历史 KV 或多个 sequence 时, 可见性要根据逻辑 position、sequence 归属等判断。当前 query 的行号不一定等于它的绝对位置。

### 4.5 可复算的数值例子

令 $d_h=2$, 假定 Q/K 已完成前面的投影和位置处理:

```text
Q = K = [[1,0], [0,1]]
V     = [[1,2], [3,4]]
M     = [[0,-inf], [0,0]]
```

则:

$$
A=\begin{bmatrix}1/\sqrt2&-\infty\\0&1/\sqrt2\end{bmatrix},\qquad
P\approx\begin{bmatrix}1&0\\0.330238&0.669762\end{bmatrix}.
$$

$$
Y=PV\approx
\begin{bmatrix}1&2\\2.339523&3.339523\end{bmatrix}.
$$

第二行是两个 Value 向量的加权和。这个例子省略 RMSNorm 和实际权重投影, 用于单独验证 attention 核心。

### 4.6 Multi-head 与 GQA

各 head 的结果拼接后再投影:

$$
Y_{\mathrm{attn}}=\operatorname{Concat}(Y_0,\ldots,Y_{H-1})W_O^\top.
$$

拼接 shape 为 `[N,H*d_h]`, $W_O$ 的 shape 为 `[D,H*d_h]`, 最终输出为 `[N,D]`。

GQA 令多个 Q heads 共享同一 K/V head。若 $H$ 能被 $K_v$ 整除, 组大小为 $g=H/K_v$, Q head $h$ 使用:

$$
\mathrm{kv\_head}(h)=\left\lfloor\frac{h}{g}\right\rfloor.
$$

例如 $H=8,K_v=2$, 则 Q heads `0..3` 共用 KV head 0, `4..7` 共用 KV head 1。

保持层数、head 维度、容量和数据类型相同, K/V 数据量相对于 $K_v=H$ 的 MHA 为 $K_v/H$。Q head 数仍为 $H$, 因此 attention score 的 head 轴不会同步缩成 $K_v$。

## 5. RoPE 与二维旋转

对应第 1、3、5 周。以下使用标准、固定频率的 RoPE 推导; 长上下文缩放和额外幅度缩放需要再核对模型配置。

### 5.1 旋转矩阵

对二维列向量 $u$, 旋转角度 $\theta$:

$$
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix},\qquad
u'=R(\theta)u.
$$

正交性质与角度相加性质:

$$
R(\theta)^\top R(\theta)=I_2,\quad
\|R(\theta)u\|_2=\|u\|_2,\quad
R(a)R(b)=R(a+b).
$$

例如 `[1,0]` 旋转 $\pi/2$ 后成为 `[0,1]`。

### 5.2 将位置变成旋转角度

设参与旋转的维度 $d_r$ 为偶数, $d_r\leq d_h$。标准频率形式为:

$$
\omega_j=b^{-2j/d_r},\qquad
\theta_{p,j}=p\omega_j,\qquad
j=0,\ldots,d_r/2-1.
$$

$b>1$ 是模型配置中的频率基数, $p$ 为 token position。对第 $j$ 对分量:

$$
\begin{bmatrix}x'_a\\x'_b\end{bmatrix}
=R(p\omega_j)\begin{bmatrix}x_a\\x_b\end{bmatrix}.
$$

标准示意可以取相邻分量配对, 但实际实现也可能采用前后半区配对。不要仅根据二维公式推断 GGML 中的分量布局。

### 5.3 为什么点积包含相对位置

对同一频率下的一对二维 Q/K 分量:

$$
\widetilde q_m=R(m\omega)q,\qquad
\widetilde k_n=R(n\omega)k.
$$

$$
\widetilde q_m^\top\widetilde k_n
=q^\top R(m\omega)^\top R(n\omega)k
=q^\top R((n-m)\omega)k.
$$

位置作用通过差值 $n-m$ 进入点积。完整 head 的点积是各分量对贡献之和; 若只旋转部分维度, 还包含未旋转部分的贡献。Q/K 内容本身依然影响结果。

### 5.4 与 KV cache shift 的关系

在固定频率的标准形式下, 将 position 从 $p$ 改为 $p+\Delta$, 对已旋转 K 的修正满足:

$$
R((p+\Delta)\omega)k=R(\Delta\omega)\left[R(p\omega)k\right].
$$

这解释了为什么普通 RoPE KV 的位置移动还可能需要修正 K, 而 V 不做同样的旋转。位置修正不等于重新计算被删除上下文造成的全部 hidden state 变化。

RoPE 提供位置信息; causal mask 决定哪些位置可见。两个操作解决不同问题。

## 6. FFN、残差与完整前向计算

对应第 1-3 周。

### 6.1 Embedding 与输出词表

Embedding table 为:

$$
E\in\mathbb R^{V_{\mathrm{vocab}}\times D},\qquad
X_{t,:}=E_{\mathrm{token\_id}_t,:}.
$$

它是按 token ID 取行。ID 是索引, 数值接近不代表语义接近。

输出投影为:

$$
Z=X_{\mathrm{final}}W_{\mathrm{out}}^\top,\qquad
W_{\mathrm{out}}\in\mathbb R^{V_{\mathrm{vocab}}\times D}.
$$

若请求 $O$ 个位置的输出, logits shape 为 `[O,V_vocab]`。Logits 可以为任意实数, 不要求非负或总和为 1。

### 6.2 SiLU 与 SwiGLU

Sigmoid 和 SiLU 定义为:

$$
\sigma(x)=\frac1{1+e^{-x}},\qquad
\mathrm{SiLU}(x)=x\sigma(x).
$$

这些函数逐元素作用于张量。SiLU 在大正数处接近 $x$, 大负数处接近 0, 部分负输入会得到负输出。

普通 Llama 风格 gated FFN:

$$
G=\mathrm{SiLU}(XW_{\mathrm{gate}}^\top),\qquad
U=XW_{\mathrm{up}}^\top,\qquad
\mathrm{FFN}(X)=(G\odot U)W_{\mathrm{down}}^\top.
$$

| 对象 | shape |
| --- | --- |
| 输入 $X$ | `[N,D]` |
| $W_{\mathrm{gate}},W_{\mathrm{up}}$ | `[I,D]` |
| $G,U,G\odot U$ | `[N,I]` |
| $W_{\mathrm{down}}$ | `[D,I]` |
| FFN 输出 | `[N,D]` |

FFN 对每个 token 独立执行相同变换; token 之间的信息交换由 Attention 完成。非线性和门控使整个 FFN 不能合并成一个固定的线性矩阵。

### 6.3 Residual 与单层 block

普通 pre-norm block 可以写成:

$$
X_1=X+\mathrm{Attention}(\mathrm{RMSNorm}_1(X)),
$$

$$
X_2=X_1+\mathrm{FFN}(\mathrm{RMSNorm}_2(X_1)).
$$

各次残差相加的两侧都是 `[N,D]`, 不在这里沿轴求和或拼接。两个 RMSNorm 通常有各自的参数。

全流程可以按下表检查:

| 阶段 | 逻辑 shape |
| --- | --- |
| token IDs | `[N]` |
| embedding | `[N,D]` |
| 一层 Attention + residual | `[N,D]` |
| 一层 FFN + residual | `[N,D]` |
| 重复 $L$ 个 block | `[N,D]` |
| 选择输出位置、最终 norm | `[O,D]` |
| 输出投影 | `[O,V_vocab]` |
| 对指定位置采样 | 一个 token ID |

该表是逻辑模型。实现可能提前选取最终需要的 token 行, 也可能融合多个数学操作; 图节点数不一定等于公式步骤数。

## 7. 自回归概率与采样

对应第 1、6 周。

### 7.1 条件概率与自回归分解

给定 prompt $c$, 生成序列 $t_1,\ldots,t_M$ 的概率为:

$$
p(t_1,\ldots,t_M\mid c)=\prod_{r=1}^{M}p(t_r\mid c,t_1,\ldots,t_{r-1}).
$$

这是概率的链式分解, 不是假设各 token 相互独立。每生成一个 token, 下一步的条件就随之改变。实际使用 temperature、过滤和 penalties 后, 抽样分布可能不同于模型原始 softmax 分布。

### 7.2 Greedy 与随机采样

Greedy 取当前分数最大项:

$$
t_{\mathrm{next}}=\operatorname*{argmax}_i z_i.
$$

随机采样则令 $t_{\mathrm{next}}\sim\operatorname{Categorical}(p)$。离散抽样可用累计概率理解: 将 `[0,1)` 分成长度为 $p_i$ 的区间, 均匀随机数落入哪个区间就选哪个候选。

一次抽到相同 token 不能证明两个分布相同。固定 seed 也不保证不同 backend、不同量化模型或不同候选处理顺序逐 token 一致。

### 7.3 Temperature

对 $\tau>0$:

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

若 $p_i(1)$ 为原始概率, 则等价地:

$$
p_i(\tau)=\frac{p_i(1)^{1/\tau}}{\sum_jp_j(1)^{1/\tau}}.
$$

| 温度 | 在固定候选集合上的作用 |
| --- | --- |
| $0<\tau<1$ | 放大分数差异, 分布更集中 |
| $\tau=1$ | 保持原分布 |
| $\tau>1$ | 缩小分数差异, 分布更平坦 |
| $\tau\to0^+$ | 质量集中到最大分数项; 有并列最大值时还涉及平局处理 |
| $\tau\to\infty$ | 有限分数候选趋向均匀分布 |

公式不能直接代入 $\tau=0$。代码中非正温度的处理属于专门分支。

### 7.4 Top-k、top-p 与重新归一化

Top-k 保留分数最高的 $k$ 个候选。Top-p 按概率降序排列, 选取满足累计概率阈值 $\rho$ 的最短前缀:

$$
r=\min\left\{m:\sum_{j=1}^{m}p_{(j)}\geq\rho\right\}.
$$

实现还会受 `min_keep` 等参数约束。设最终保留集合为 $\mathcal A$, 真正抽样前的分布为:

$$
\widetilde p_i=
\begin{cases}
p_i/\sum_{j\in\mathcal A}p_j,&i\in\mathcal A,\\
0,&i\notin\mathcal A.
\end{cases}
$$

截断并不保证 candidate 数组中已有的 `p` 字段立即被重新归一化; temperature 改变 logits 后, 旧 `p` 也可能已经过期。

### 7.5 为什么采样顺序影响结果

沿用第 6 周例子:

```text
p = [0.55,0.25,0.15,0.05]
z_i = ln(p_i)
top_p = 0.85, min_keep = 1, temperature = 0.5
```

| 路径 | 保留集合 | 最终概率 |
| --- | --- | --- |
| top-p 后 temperature | 前三项 | $[121/155,25/155,9/155,0]$ |
| temperature 后 top-p | 前两项 | $[121/146,25/146,0,0]$ |

第一条路径先保留累计概率为 0.95 的前三项, 再平方并归一化。第二条路径先对四个概率平方并归一化, 得到 `[121,25,9,1]/156`, 前两项累计概率已超过 0.85。

因此, 各 sampler 的复合通常不可交换。正温度本身保持分数排序, 但会改变 top-p 使用的累计概率。

### 7.6 重复惩罚与 grammar

当前 penalties 路径对历史窗口中出现过的候选, 先使用 repeat penalty $r>0$:

$$
z'_i=
\begin{cases}
z_i/r,&z_i>0,\\
r z_i,&z_i\leq0.
\end{cases}
$$

再使用 frequency 和 presence penalty:

$$
z''_i=z'_i-\alpha c_i-\beta\,\mathbf1[c_i>0].
$$

$c_i$ 为候选在接受历史窗口中的出现次数, $\alpha,\beta$ 为对应参数。未出现候选不执行上述重复分支。通常 $r>1,\alpha,\beta\geq0$ 用于降低重复倾向; 这些是生成策略, 不是模型训练公式。

Grammar 可理解为先由当前语法状态决定允许集合 $\mathcal A$, 再将非法候选的 logit 设为 $-\infty$。对一个固定集合的屏蔽再归一化, 相当于条件化到该集合; 后续其他 sampler 仍可能继续改变分布。

## 8. 熵、交叉熵与困惑度

对应第 6-7 周。熵用于必做 observer; 困惑度与分布对比属于扩展评估。

### 8.1 信息熵

对离散分布 $p$:

$$
\mathcal H(p)=-\sum_i p_i\ln p_i.
$$

采用约定 $0\ln0=0$, 其依据为 $\lim_{x\to0^+}x\ln x=0$。自然对数对应单位 nat, 以 2 为底对应 bit:

$$
\mathcal H_{\mathrm{bit}}(p)=\mathcal H_{\mathrm{nat}}(p)/\ln2.
$$

对 $m$ 个可能候选, $0\leq\mathcal H(p)\leq\ln m$。确定分布熵为 0, 均匀分布熵为 $\ln m$。

```text
H([1,0])       = 0
H([0.5,0.5])   = ln(2) ~= 0.693147 nat
H([0.25,0.75]) ~= 0.562335 nat
```

熵衡量当前候选分布的分散程度, 不能直接当作答案正确率或模型知识可靠性。第 6 周 observer 应从当前有效 logits 重新计算概率。

### 8.2 负对数概率与交叉熵

模型给真实目标 token $y$ 的概率为 $p_y$, 单个位置的负对数概率为:

$$
\ell=-\ln p_y.
$$

给正确 token 的概率越大, $\ell$ 越小。一般的交叉熵为:

$$
\mathcal H(q,p)=-\sum_i q_i\ln p_i.
$$

若目标 $q$ 是 one-hot 分布, 它就退化为 $-\ln p_y$。这是用来解释评估量的公式, 本计划无需实现训练过程。

### 8.3 困惑度

对实际参与评分的 $M$ 个 token:

$$
\mathrm{NLL}_{\mathrm{avg}}=-\frac1M\sum_{r=1}^{M}\ln p(t_r\mid\mathrm{context}_r),
$$

$$
\mathrm{PPL}=\exp(\mathrm{NLL}_{\mathrm{avg}}).
$$

例如, 两个目标 token 的条件概率为 0.5 和 0.25:

$$
\mathrm{PPL}=\exp\left(-\frac{\ln0.5+\ln0.25}{2}\right)=\sqrt8\approx2.828427.
$$

困惑度是目标 token 概率的几何平均数的倒数, 不是每一步概率倒数的算术平均。若每次都在 $m$ 个 token 中均匀预测, 困惑度为 $m$。

比较困惑度时必须保持 tokenizer、语料、上下文和评分位置规则可比。工具可能按 chunk 组织上下文并跳过部分评分位置, $M$ 应取实际评分数量。小规模 PPL 对照不能代表所有实际任务的质量。

### 8.4 KL 散度, 选读

比较同一输入前缀下的基准分布 $p$ 和近似分布 $q$:

$$
D_{\mathrm{KL}}(p\|q)=\sum_i p_i\ln\frac{p_i}{q_i}
=\mathcal H(p,q)-\mathcal H(p).
$$

KL 非负, 两个分布一致时为 0, 通常不对称。若某项 $p_i>0,q_i=0$, 则该方向 KL 为无穷大。它可用于量化分布对比, 不属于八周必做主线。

对比时应使用相同 token 前缀及相同概率处理口径。两条生成历史已经分叉后, 分布差异同时包含上下文差异。

## 9. 浮点数、量化与误差

对应第 6-7 周。

### 9.1 浮点表示与有限精度

二进制正规浮点数可概括为:

$$
x=(-1)^s(1.f)_2\,2^{e-\mathrm{bias}}.
$$

| 类型 | 符号位 | 指数位 | 小数部分位数 | 存储字节 |
| --- | ---: | ---: | ---: | ---: |
| F32 | 1 | 8 | 23 | 4 |
| F16 | 1 | 5 | 10 | 2 |
| BF16, 补充了解 | 1 | 8 | 7 | 2 |

该表达式省略了非正规数、零、无穷与 NaN 的特殊编码。位数有限意味着多数实数只能近似表示; 存储类型也不一定等于运算累加器类型。

浮点加法不严格满足结合律, $(a+b)+c$ 与 $a+(b+c)$ 可能不同。线程归约顺序、融合运算或 backend 改变, 都可能影响最后几位结果。

### 9.2 误差度量与数值比较

设参考值为 $x_i$, 近似值为 $\widehat x_i$:

$$
e_i=\widehat x_i-x_i,\qquad
\mathrm{MAE}=\frac1n\sum_i|e_i|,\qquad
\mathrm{MSE}=\frac1n\sum_ie_i^2,\qquad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
$$

绝对误差为 $|e_i|$, 非零参考值的相对误差为 $|e_i|/|x_i|$。接近零时, 相对误差会非常敏感。

常见的容差比较形式是:

$$
|\widehat x_i-x_i|\leq\mathrm{atol}+\mathrm{rtol}|x_i|.
$$

这是常见形式, 不表示仓库每个测试都采用同一判据。权重误差、logits 误差和任务质量是不同层次的量。

若最大与次大 logit 的间隔大于 $2\delta$, 且每个 logit 的绝对扰动均不超过 $\delta$, 则最大项不会被颠覆。间隔很小时, 小误差就可能改变 greedy token, 并使后续历史分叉。

### 9.3 均匀量化的基本模型

一般的仿射量化可写成:

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s)+z_0,q_{\min},q_{\max}\right),\qquad
\widehat x=s(q-z_0).
$$

$s>0$ 是步长, $z_0$ 是零点编码。该式用于理解缩放、离散化、截断与重建, 不是所有 GGML 格式的完整实现。

在步长精确、最近舍入、没有截断的理想条件下, $|\widehat x-x|\leq s/2$。实际误差还可能来自 scale 的低精度存储和范围截断。

例: $s=0.5,z_0=0,x=1.3$, 则 $q=3,\widehat x=1.5,e=0.2$。

### 9.4 Q8_0 与 Q4_0 的具体约定

Q8_0 每 32 个值共享一个 FP16 scale。参考实现取:

$$
a_{\max}=\max_i|x_i|,\qquad d=a_{\max}/127,\qquad
q_i=\operatorname{round}(x_i/d).
$$

全零 block 单独处理。重建使用存储后的 scale $\widehat d$, 即 $\widehat x_i=\widehat d\,q_i$。

Q4_0 每 32 个值共享一个 scale, 每个值存为 4 bit 编码。当前参考实现取 block 内绝对值最大的带符号值 $v_{\max}$:

$$
d=v_{\max}/(-8),\qquad
q_i=\min\left(15,\operatorname{trunc}(x_i/d+8.5)\right),\qquad
\widehat x_i=\widehat d(q_i-8).
$$

这里是与参考实现对应的非零 block 示意, 编码使用原始 $d$, 解码使用存为 FP16 后的 $\widehat d$。`trunc` 对应此处的整数转换; 全零 block 的倒数按 0 处理。

Q4_0 的 $d$ 可以为负, 不应直接替换成恒正的 `absmax/7`。例如 $v_{\max}=-8$, 则 $d=1$; $x=1.3$ 编码为 9, 重建为 1, 误差为 -0.3。

### 9.5 Block 开销与有效位数

若一个 block 表示 $b$ 个逻辑值, 存储 $s_b$ 字节:

$$
\mathrm{bits/value}=\frac{8s_b}{b}.
$$

当前计划涉及的基本类型:

| 类型 | 每 block 值数 | 每 block 字节 | 有效 bits/value |
| --- | ---: | ---: | ---: |
| F16 | 1 | 2 | 16 |
| Q8_0 | 32 | 34 | 8.5 |
| Q4_0 | 32 | 18 | 4.5 |
| Q4_K | 256 | 144 | 4.5 |

Q4_0 的 18 字节包含 16 字节 packed quants 和 2 字节 scale。Q4_K 还组织了子块的 scale/min 等信息; 有效位数相同不表示编码和误差相同。

对每行宽度满足 block 要求的连续矩阵, 若逻辑形状为 `[M,K]`:

$$
\mathrm{payload\ bytes}=M\frac Kb s_b.
$$

例如 $M=16,K=256$:

| 类型 | 逻辑 payload 字节 |
| --- | ---: |
| F16 | 8192 |
| Q8_0 | 4352 |
| Q4_0 | 2304 |
| Q4_K | 2304 |

以上不含文件 metadata、alignment 和 backend padding。Q4_K_M 是混合多种 tensor 类型的模型量化预设, 不能将整个模型都按 4 bit 或 4.5 bit/weight 计算。

### 9.6 Imatrix 与加权误差

对某输入通道 $j$, 用 $R$ 个校准样本估计其二阶幅度:

$$
a_j=\frac1R\sum_{r=1}^{R}(x_j^{(r)})^2\approx\mathbb E[x_j^2].
$$

它一般不等于方差, 因为 $\mathbb E[x_j^2]=\operatorname{Var}(x_j)+\mathbb E[x_j]^2$。

对一个输出分量, 权重误差导致:

$$
\Delta y=\sum_j\Delta w_jx_j,\qquad
\mathbb E[(\Delta y)^2]=\Delta w^\top\mathbb E[xx^\top]\Delta w.
$$

忽略跨通道项后, 可得到加权平方误差的直觉:

$$
\mathbb E[(\Delta y)^2]\approx\sum_ja_j(\Delta w_j)^2.
$$

因此, 活跃幅度大的通道可能需要更关注其量化误差。这解释了 importance 信息的用途。普通 dense 路径的逐通道统计不是完整协方差矩阵, 也不能据此断言所有格式都严格优化同一目标。

第 7 周要求理解用途, 不要求推导全部量化算法。Imatrix 校准实验是选做, 应与质量评估使用的数据分开。

## 10. 存储容量、索引与状态共享

对应第 2、4、5、7、8 周。

### 10.1 KV cache 容量

对层结构相同的普通全层 Attention, 设每个 head 的 K/V 维度分别为 $d_k,d_v$, 每个元素存储字节数为 $s_k,s_v$:

$$
\mathrm{KV\ bytes}\approx LC K_v(d_ks_k+d_vs_v).
$$

若 $d_k=d_v=d_h$、$s_k=s_v=s$:

$$
\mathrm{KV\ bytes}\approx2LC K_vd_hs.
$$

这里的 $C$ 是分配的物理 cell 总数, 不一定等于当前已填入的 token 数或单条 sequence 的长度。若每层配置不同, 应逐层求和。

例如 $L=28,C=256,K_v=2,d_h=128$, K/V 都为 F16:

$$
2\cdot28\cdot256\cdot2\cdot128\cdot2
=7\,340\,032\ \mathrm{bytes}=7\ \mathrm{MiB}.
$$

此估算不含 metadata、padding、alignment、compute buffer 和权重。KV 按 block 量化时要按实际 block 存储算字节; recurrent、hybrid 和特殊 cache 布局不能直接套用此式。

权重量化主要改变模型权重部分。K/V 仍用 F16 时, cache 不会随权重 bit 数等比例缩小。

### 10.2 参数量与输出 buffer

矩阵参数量等于各维长度之积。例如 `[Dout,Din]` 权重有 $D_{\mathrm{out}}D_{\mathrm{in}}$ 个参数。普通 gated FFN 的三组权重共有:

$$
ID+ID+DI=3DI.
$$

这省略 bias 和 norm。共享同一权重存储的 tied embedding/output 不能重复计算实际存储。

F32 logits buffer 的逻辑大小为 $4OV_{\mathrm{vocab}}$ 字节, 其中 $O$ 是实际请求输出的位置数。

单位换算: 1 byte = 8 bit, 1 KiB = $2^{10}$ bytes, 1 MiB = $2^{20}$ bytes, 1 GiB = $2^{30}$ bytes。MB/GB 通常使用十进制, 不能与 MiB/GiB 混用。

### 10.3 共享前缀的集合计数

设两个 sequence 的 cell 集合为 $A,B$, 则:

$$
|A\cup B|=|A|+|B|-|A\cap B|.
$$

如果共享前缀有 $P$ 个物理 cells, 两条分支分别新增 $a,b$ 个 cells, 总占用为 $P+a+b$, 而独立保存两份前缀需要 $2P+a+b$。

例如共享 2 个前缀位置, 两个分支各新增 1 个位置, 两条 sequence 各长 3, 总共只占 4 个 cells。相同 position 可以存在于不同私有 cells 中, 是否可见还取决于 sequence 归属。

释放一条 sequence 的引用后, 共享 cell 仍可能被另一条使用; 已分配的物理 buffer 也不一定缩小。

### 10.4 Batch 切分与依赖图

对无失败重试的普通单 sequence 路径, 若应用每次最多提交 $b$ 个 token, 总 prompt 长为 $P$, 则 API 调用数为:

$$
n_{\mathrm{API}}=\left\lceil P/b\right\rceil.
$$

若第 $j$ 次 API 调用有 $n_j$ 个 token, 且内部每个 ubatch 最多容纳 $u$ 个, 在简单顺序切分条件下:

$$
n_{\mathrm{ubatch}}=\sum_{j=1}^{n_{\mathrm{API}}}\left\lceil n_j/u\right\rceil.
$$

例如 $P=8,b=4,u=2$, prefill 有 2 次 API 调用、4 个 ubatch; 再生成并 decode 两个新 token, 总共为 4 次 API 调用、6 个 ubatch。多 sequence、特殊 memory 和重试路径需要依据实际规则分析。

GGML 前向图可理解为有向无环依赖图: 边表示数据依赖, 可执行顺序必须先满足前驱依赖。调度还需考虑设备能力和 buffer, 不必为本计划先学习完整图论或最优化调度理论。

## 11. 计算量、带宽与性能统计

对应第 4、7、8 周。

### 11.1 运算量估算

按照一次乘法和一次加法合计约 2 FLOPs, 普通 dense 运算的估算为:

| 运算 | 主要运算量 |
| --- | --- |
| `[m,k]` 乘 `[k,n]` | $2mkn$ |
| 一个线性投影 | $2ND_{\mathrm{in}}D_{\mathrm{out}}$ |
| 所有 Q heads 的 $QK^\top$ | $2HNSd_h$ |
| 所有 Q heads 的 $PV$ | $2HNSd_h$ |
| gated FFN 的三次矩阵乘法 | $6NDI$ |

这里省略 bias、norm、softmax、激活函数等开销。量化 dot 的实际指令还包含解码、转换和整数运算, 不能仅用这些浮点等价计数预测速度。

完整序列 prefill 中 $N=S=T$, 两次 attention 矩阵乘法的复杂度为 $O(HT^2d_h)$。单 sequence 的一步 decode 中 $N=1$, 对已有长度 $S$ 的 attention 复杂度为 $O(HSd_h)$。这只描述 attention 部分, 模型还需执行各层投影和 FFN。

普通显式 score 张量有 $HNS$ 个元素。Flash Attention 可以通过分块、融合减少中间结果的存储和搬运, 不表示 dense attention 的配对计算量变成线性。

### 11.2 算术强度与带宽模型

令运算量为 $F$, 在选定存储层级上传输的字节数为 $Q$:

$$
\mathrm{AI}=\frac FQ\quad(\mathrm{operations/byte}).
$$

若可用计算速率为 $P_{\max}$, 带宽为 $BW$, 简化模型给出:

$$
t\gtrsim\max\left(\frac F{P_{\max}},\frac Q{BW}\right),\qquad
P_{\mathrm{achievable}}\lesssim\min(P_{\max},BW\cdot\mathrm{AI}).
$$

这是理想化的瓶颈分析框架, 实际还有同步、调度、kernel launch 等成本。$Q$ 必须与所选的 DRAM、cache 或设备互连层级一致, 模型文件大小不等于每步真实传输量。

多 token 投影有机会复用权重, 从而提高算术强度; 单 sequence decode 每次输入较少, 可能更受权重读取带宽影响。具体结论要结合实际 shape、batch、backend 和测量证据。

### 11.3 吞吐、延迟与加速比

$$
\mathrm{throughput}=\frac{\mathrm{completed\ tokens}}{\mathrm{elapsed\ seconds}},\qquad
\mathrm{speedup}=\frac{t_{\mathrm{baseline}}}{t_{\mathrm{candidate}}}.
$$

加速比要求工作量和计时范围一致。PP、TG、模型加载时间、HTTP 端到端延迟分别覆盖不同工作, 不应直接混比。

并发请求的总体吞吐应使用本组 token 总数除以“首请求发出到末请求完成”的墙钟时间, 不把各请求的 tokens/s 简单相加。排队时间、首字节时间和首个生成内容到达时间也需要独立定义。

### 11.4 均值与样本标准差

对 $n$ 次观测 $x_1,\ldots,x_n$, $n\geq2$:

$$
\bar x=\frac1n\sum_i x_i,\qquad
s=\sqrt{\frac1{n-1}\sum_i(x_i-\bar x)^2}.
$$

本地 `llama-bench` 的 `stdev` 使用样本标准差口径。工具在仅一个样本时返回 0, 这不代表已经证明没有波动。

如果每次处理 $K$ 个 token, 耗时为 $t_i$, 平均吞吐是:

$$
\bar r=\frac1n\sum_i\frac K{t_i}.
$$

它一般不等于 $K/\bar t$。记录 `avg_ts` 时应沿用工具的计算口径, 不自行混换。

例如吞吐为 `[10,12,14]` tokens/s, 均值为 12, 样本标准差为 2。标准差描述样本波动, 不是均值的置信区间; 在独立同分布等假设下, 均值标准误才可估为 $s/\sqrt n$。

### 11.5 实验结论的范围

分位数描述分布位置, 如 p99 对应约 99% 的观测不超过该值。三次测量可以做初步对照, 不足以稳定估计尾部 p99。

固定硬件、模型、线程、输入长度和 warmup 口径, 每组只改变一个主要变量。若速度差与波动同量级, 应保留“当前数据不足以区分”的结论。有限次数全部成功也不能证明一般情况下的失败概率为零。

## 12. 微积分的使用边界

本节为选读。主线任务是前向推理, 不要求先掌握完整矩阵求导、反向传播和梯度优化。

本科微积分可帮助理解激活函数和分布敏感性:

$$
\frac{d}{dx}e^x=e^x,\qquad
\frac{d}{dx}\ln x=\frac1x,\qquad
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

$$
\mathrm{SiLU}'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x)).
$$

对有限 logits 的 softmax, 令 $\delta_{ij}$ 在 $i=j$ 时为 1、否则为 0:

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
$$

这说明改变一个 logit 会同时影响多个概率。对含温度的 $\mathrm{softmax}(z/\tau)$, 相对于原始 $z$ 求导还要乘 $1/\tau$。

损失对参数的梯度、优化器、Hessian 或更深入的量化优化理论, 可在以后扩展到训练或算法研究时补充。当前计划无需先复习完整积分学、特征值分解或 SVD。

## 13. 按周复习与自测

### 13.1 按周查阅

| 周次 | 优先查阅 | 应能独立完成 |
| --- | --- | --- |
| 第 1 周 | 第 2-7 节的基础内容 | 小矩阵乘法、softmax、简化 Attention 和 block shape |
| 第 2 周 | 第 2、4、10 节 | 从 hparams 推 Q/K/V shape, 估算参数量和 KV 字节 |
| 第 3 周 | 第 2-6 节 | 将每条公式映射到 GGML op, 区分逻辑 shape 与布局 |
| 第 4 周 | 第 10-11 节 | 预测简单 batch/ubatch 切分, 区分 PP 与 TG |
| 第 5 周 | 第 5、10 节 | 解释共享 cell、position、RoPE 修正和容量 |
| 第 6 周 | 第 3、7-8 节 | 跟踪候选概率, 比较采样顺序, 计算熵 |
| 第 7 周 | 第 8-11 节 | 计算 block 大小、误差、统计量; 提出有条件的性能解释 |
| 第 8 周 | 第 10-11 节 | 正确统计共享内存、并发吞吐和端到端延迟 |

### 13.2 自测题与核对结果

先独立计算, 再查看结果。能够解释轴和计算过程比记住结果更重要。

| 题目 | 核对结果 |
| --- | --- |
| `X=[2,3]`、`W=[4,3]`, `X W^T` 的 shape 是什么? | `[2,4]` |
| 第 2.2 节给定 X/W 的投影结果是什么? | `[[4,-1],[10,-1]]` |
| `softmax([0,ln(3)])` 是什么? | `[0.25,0.75]` |
| `[3,4]` 忽略 epsilon 和权重后的 RMSNorm 是什么? | 约 `[0.8485,1.1314]` |
| 第 4.5 节第二个 query 的 Attention 输出是什么? | 约 `[2.339523,3.339523]` |
| $H=8,K_v=2$, Q head 6 使用哪个 KV head? | 以 0 为起点, 使用 KV head 1 |
| `[0.25,0.75]` 的熵是多少? | 约 0.562335 nat |
| 两个目标概率为 0.5、0.25, PPL 是多少? | $\sqrt8$, 约 2.828427 |
| 32 个 Q4_0 值有多少字节和有效 bits/value? | 18 字节, 4.5 bits/value |
| 28 层、256 cells、2 个 KV heads、head dim 128、F16 K/V 需要多少数据? | 约 7 MiB, 不含其他开销 |
| 吞吐样本 `[10,12,14]` 的均值和样本标准差是什么? | 12 和 2 |

另外应能口述: 为什么 mask 不用 0 替代负无穷, 为什么 RoPE 与 causal mask 不能互换, 为什么低熵不等于答案正确, 为什么 Q4_K_M 不能按统一 4 bit 估算, 为什么一次输出不同不能直接认定量化实现错误。

## 14. 文档与源码入口

以下入口用于将公式对应回本计划的任务。数学基本定义稳定, 具体分支、布局和参数以实际 checkout 为准。

| 主题 | 学习文档 | 源码或工具入口 |
| --- | --- | --- |
| 概念与 shape | [概念桥接](000-transformer-concept-bridge.md)、[第 1 周](001-llama-cpp-week1-daily-learning-route.md) | [Llama 模型](../../src/models/llama.cpp) |
| 权重和配置 | [第 2 周](002-llama-cpp-week2-daily-learning-route.md) | [模型加载](../../src/llama-model.cpp) |
| 公式与计算图 | [第 3 周](003-llama-cpp-week3-daily-learning-route.md) | [graph helpers](../../src/llama-graph.cpp)、[GGML API](../../ggml/include/ggml.h) |
| batch 与索引 | [第 4 周](004-llama-cpp-week4-daily-learning-route.md) | [batch](../../src/llama-batch.cpp) |
| KV 与位置 | [第 5 周](005-llama-cpp-week5-daily-learning-route.md) | [KV cache](../../src/llama-kv-cache.cpp)、[CPU ops 中的 RoPE](../../ggml/src/ggml-cpu/ops.cpp) |
| 采样与熵 | [第 6 周](006-llama-cpp-week6-daily-learning-route.md) | [sampler](../../src/llama-sampler.cpp) |
| 量化与误差 | [第 7 周](007-llama-cpp-week7-daily-learning-route.md) | [quantize/dequantize](../../ggml/src/ggml-quants.c)、[block 定义](../../ggml/src/ggml-common.h) |
| Imatrix | [量化学习笔记](answer/week7/week7-day5.md) | [imatrix](../../tools/imatrix/imatrix.cpp) |
| 困惑度与 KL | 第 7 周选做评估 | [perplexity 文档](../../tools/perplexity/README.md) |
| 性能统计 | 第 7 周性能对照 | [llama-bench](../../tools/llama-bench/llama-bench.cpp) |
| 服务与并发 | [第 8 周](008-llama-cpp-week8-daily-learning-route.md) | [server 文档](../../tools/server/README.md) |
