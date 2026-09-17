# Fluctuation Gradient：独立噪声的二阶平均与梯度

本文对应 `max_order=2, drop_odd_average=True` 的准静态独立噪声模型。
每个通道分别具有给定的标准差，同一通道在一次门操作期间保持不变。
**先保留通道索引做展开，再按二阶矩收缩；不能先合并噪声算符再平方。**

## 1. 统计假设与单步插入

把静态和控制噪声统一记作

$$
H(t;z)=H_{\rm nominal}(t)+\sum_a z_a G_a(t),\qquad
\mathbb E[z_a]=0,\quad \mathbb E[z_a z_b]=\delta_{ab}.
$$

其中静态源的 $G_a=\sigma_a H_a$，相对控制源的
$G_a(t_k)=\sigma_a c_i(k)H_{\chi_i}$。标准差已包含在 $G_a$ 中，
平均时不能再乘一次 $\sigma_a^2$。

$$
W_k=\exp[-i\Delta t H_{\rm nominal}(c_k)],\qquad
V_{k,a}=-i\Delta t G_a(c_k)W_k,
\qquad \delta U_k\approx\sum_a z_a V_{k,a}.
$$

对于任意不含噪声的中间传播 $M$，

$$
\mathbb E[\delta U_k M\delta U_l]
=\sum_{a,b}\mathbb E[z_a z_b]V_{k,a}MV_{l,b}
=\sum_a V_{k,a}MV_{l,a}.
$$

另一侧为伴随插入时同样成立。若先定义 $V_k=\sum_a V_{k,a}$，
再使用 $V_kMV_l$，就会保留 $a\ne b$ 交叉项；这对应完全相关的
$\xi_a=\sigma_a z$，不符合本文的独立通道假设。
一般相关噪声需要显式的协方差 $C_{ab}=\mathbb E[z_a z_b]$，
当前多通道平均接口不提供该功能。

代码入口：[单步构造](../quantum_control/steps/perturbative_step.py)。
`OpenSystem.fluctuation_hamiltonian()` 仍返回算符总和，供单位噪声取值或
诊断使用；它不是统计平均。单步构造器使用分开的
`static_fluctuations` 和 `control_fluctuations`。
控制噪声按控制通道的位置对应，零强度占位项不能随意删除。
只有旧式聚合哈密顿量接口的自定义系统被解释为一个有效噪声源。

## 2. 逐通道前后向递推

名义态在所有噪声通道间共享：

$$
F_k=W_kF_{k-1},\qquad F_0=|\psi_0\rangle,
\qquad B_k=W_{k+1}^\dagger B_{k+1},\quad B_N=|\phi\rangle.
$$

每个通道的一次和两次插入为

$$
SF_{k,a}=W_kSF_{k-1,a}+V_{k,a}F_{k-1},
$$

$$
DF_{k,a}=W_kDF_{k-1,a}+V_{k,a}SF_{k-1,a}.
$$

边界条件是 $SF_{0,a}=DF_{0,a}=0$。一般 $SF_{1,a}=V_{1,a}F_0\ne0$，
而 $DF_{1,a}=0$。只存同一通道的两次插入，因为异通道项的平均为零。

相应的后向态满足

$$
SB_{k,a}=W_{k+1}^\dagger SB_{k+1,a}+V_{k+1,a}^\dagger B_{k+1},
$$

$$
DB_{k,a}=W_{k+1}^\dagger DB_{k+1,a}+V_{k+1,a}^\dagger SB_{k+1,a},
\qquad SB_{N,a}=DB_{N,a}=0.
$$

代码：[展开演化](../quantum_control/evolution/expansion_evolution.py)。
多源时 `V.shape == (n_noise, d, d)`；`components[0]` 为 `(d,)`，
`components[1]` 和 `components[2]` 为 `(n_noise, d)`。
单源或无源时保持原来的矩阵／向量形状；Lindblad 的展开状态仍是向量。
使用按通道广播的矩阵向量乘法，共享每一步的 `W` 和 `dW`，
不会为各噪声源重复计算名义矩阵指数。

## 3. 保真度的二阶平均

令

$$
A_0=\langle\phi|F_N\rangle,\quad
A_{1,a}=\langle\phi|SF_{N,a}\rangle,\quad
A_{2,a}=\langle\phi|DF_{N,a}\rangle.
$$

则当前离散近似给出

$$
\overline F\approx |A_0|^2+
\sum_a\left[|A_{1,a}|^2+2\operatorname{Re}(A_0^*A_{2,a})\right].
$$

闭系统项只计一次，噪声修正逐通道相加。
特别地，$\sum_a|A_{1,a}|^2$ **不是** $|\sum_a A_{1,a}|^2$。
仅丢弃奇数总阶项不能消除两个不同通道各出现一次的二阶项。

[ExpansionFidelity](../quantum_control/objectives/expansion_fidelity.py)
返回的 `amplitudes[0]` 为标量，多源时 `amplitudes[1]` 和 `amplitudes[2]`
分别为通道向量；`contract` 逐元素乘积后对通道求和。
多源平均仅支持 `max_order <= 2` 且 `drop_odd_average=True`；
不满足条件时明确报错。更高阶需要混合通道状态及对应高阶矩，
不能把当前逐通道二阶公式直接推广。

`StateAverageProblem` 是在上述噪声平均之后，对输入／目标态对做加权求和。
它不是噪声平均，也不改变各通道的独立性假设。

## 4. 局部导数与后向收缩

对时间片 $k$ 的控制坐标 $c_i(k)$，只在该时间片求局部导数。
记 $dW_k=\partial W_k/\partial c_i(k)$，则

$$
dV_{k,a}=-i\Delta t\left[(\partial_{c_i(k)}G_a)W_k+G_a dW_k\right].
$$

静态源的显式导数为零；属于控制 $j$ 的相对噪声满足
$\partial_{c_i(k)}G_a=\delta_{ij}\sigma_a H_{\chi_j}$。
通过 $W_k$ 的导数则影响所有通道。

`dW_method="first_order"` 使用 $dW_k\approx-i\Delta t H_iW_k$；
`dW_method="frechet"` 使用矩阵指数的 Fréchet 导数。
`V_method="frechet"` 逐噪声通道计算插入的 Fréchet 导数，
其控制导数仍使用现有的中心有限差分。

局部导数向量为

$$
g_0=dW_k F_{k-1},\qquad
g_{1,a}=dW_k SF_{k-1,a}+dV_{k,a}F_{k-1},
$$

$$
g_{2,a}=dW_k DF_{k-1,a}+dV_{k,a}SF_{k-1,a}.
$$

用已经缓存的后向态收缩：

$$
dA_0=B_k^\dagger g_0,
$$

$$
dA_{1,a}=B_k^\dagger g_{1,a}+SB_{k,a}^\dagger g_0,
$$

$$
dA_{2,a}=B_k^\dagger g_{2,a}+SB_{k,a}^\dagger g_{1,a}
+DB_{k,a}^\dagger g_0.
$$

这里 $B_k$ 只包含时间片 $k$ 之后的传播。代码时间片下标从 0 开始，
因此读取 `backward[step_index + 1]`。
每个乘积保留同一通道索引，最终才求和：

$$
\partial_c\overline F=2\operatorname{Re}\left\{
A_0^*dA_0+\sum_a\left[
A_{1,a}^*dA_{1,a}+(dA_0)^*A_{2,a}+A_0^*dA_{2,a}
\right]\right\}.
$$

代码：[展开梯度](../quantum_control/differentiators/expansion_differentiator.py)。
闭系统梯度只计一次，`value_and_gradient` 与分别求值／求导使用同一平均规则。

## 5. 诊断、验证与近似边界

优化、`noisy_gate_fidelity`、保真度分项日志及误差诊断都使用上述规则。
报告的 $\kappa_2$ 使用各通道谱范数平方和的平方根，再乘门时间并取控制边界上的最大值；
它是通道强度估计，不是把可能互相抵消的算符先相加后取范数，也不是实测截断误差。
逐通道诊断必须保留控制噪声的位置：只开启第二控制噪声时，
第一控制噪声应保留零强度占位。精确评估保留该位置映射，
但只对非零噪声源生成 Gauss–Hermite 积分维度。

日志的 `first_order_sq` 为 $\sum_a|A_{1,a}|^2$，
`second_order_cross` 为 $2\operatorname{Re}(A_0^*\sum_a A_{2,a})$。
为兼容旧 CSV，`a1_real/imag`、`a2_real/imag` 显示通道振幅的总和，
只作辅助诊断；不能用显示的 `a1` 平方重建 `first_order_sq`。
`dropped_order1_cross` 是被丢弃的一阶代数项总和，不是平均噪声修正。

回归验证包括：独立的 $X$ 与 $-X$ 噪声不抵消、修正与梯度可加、
独立源符号翻转不改变平均、多通道有限差分梯度、
优化／评估／诊断一致，以及第二控制噪声的归属。

本次只修正跨通道统计平均。每个单步传播仍仅保留一次插入，
没有补齐同一时间片内部的二阶项，因此结果不是步长精确的二阶展开；
Fréchet 单次插入也不能消除这一缺项。
比较精确 Lindblad／Gauss–Hermite 结果时，应分别检查步长收敛和噪声强度，
不能把已有的时间离散误差归为跨通道平均误差，也不应把近似值裁剪到 [0, 1] 隐藏误差。

2026-09-14 之前生成的实验报告属于旧的聚合噪声实现；本次不改写历史结果。
复现修复前的数值还需要对应旧代码版本，仅复用 YAML 和脉冲不足以保证相同结果。
