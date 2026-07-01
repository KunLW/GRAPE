# Fluctuation Gradient 算法说明

本文说明当前代码中 fluctuation gradient 的核心思路：把带 fluctuation 的传播写成按 fluctuation 插入次数展开的状态；对每个控制参数 \(c_{k,a}\)，只计算它所在时间片 \(k\) 的局部导数；最后用已经缓存好的 forward/backward expansion 做 contraction，得到 fidelity objective 的梯度。

相关代码主要在：

- [`quantum_control/steps/perturbative_step.py`](../quantum_control/steps/perturbative_step.py)
- [`quantum_control/evolution/expansion_evolution.py`](../quantum_control/evolution/expansion_evolution.py)
- [`quantum_control/differentiators/expansion_differentiator.py`](../quantum_control/differentiators/expansion_differentiator.py)
- [`quantum_control/objectives/expansion_fidelity.py`](../quantum_control/objectives/expansion_fidelity.py)

## 1. 单步传播子

第 \(k\) 个时间片的名义传播子是：

$$
W_k = \exp\left(-i\,\Delta t\,H_{\mathrm{nominal}}(c_k)\right)
$$

一阶 fluctuation 插入项写成：

$$
V_k = -i\,\Delta t\,H_{\mathrm{fluc}}(c_k)\,W_k
$$

其中 fluctuation Hamiltonian 是：

$$
H_{\mathrm{fluc}}(c_k)
= \sum_j H^{(j)}_{\mathrm{static\_fluc}}
+ \sum_a c_{k,a} H^{(a)}_{\mathrm{control\_fluc}}
$$

也就是说，\(V_k\) 表示在第 \(k\) 步中插入一次 fluctuation。它不是一个独立传播子，而是 fluctuation Hamiltonian 乘在名义传播子 \(W_k\) 前面。

对应代码在 [`perturbative_step.py`](../quantum_control/steps/perturbative_step.py)：

```python
def build_step(self, system, controls, dt, t=None):
    unitary_step = super().build_step(system, controls, dt, t=t)
    fluctuation_h = system.fluctuation_hamiltonian(controls, t=t)
    return PerturbativeStep(
        W=unitary_step.W,
        V=-1j * dt * fluctuation_h @ unitary_step.W,
    )
```

对控制参数 \(c_{k,a}\) 求导时，名义传播子的导数默认采用一阶近似：

$$
\frac{\partial W_k}{\partial c_{k,a}}
\approx
-i\,\Delta t\,H_a\,W_k
$$

fluctuation 插入项的导数是：

$$
\frac{\partial V_k}{\partial c_{k,a}}
=
-i\,\Delta t\,
\frac{\partial H_{\mathrm{fluc}}}{\partial c_{k,a}}\,W_k
-i\,\Delta t\,
H_{\mathrm{fluc}}(c_k)\,
\frac{\partial W_k}{\partial c_{k,a}}
$$

对应代码在 [`perturbative_step.py`](../quantum_control/steps/perturbative_step.py)：

```python
def derivative_step(self, system, controls, dt, control_index, step, t=None):
    dW = super().derivative_step(system, controls, dt, control_index, step, t=t).W
    dfluc_h = system.fluctuation_control_derivative(
        control_index,
        controls=controls,
        t=t,
    )
    dV = -1j * dt * dfluc_h @ step.W
    if self.dV_method == "include_dW":
        fluctuation_h = system.fluctuation_hamiltonian(controls, t=t)
        dV = dV + -1j * dt * fluctuation_h @ dW
    return PerturbativeStep(W=dW, V=dV)
```

第二项 \(H_{\mathrm{fluc}}\frac{\partial W_k}{\partial c_{k,a}}\) 很关键。它表示 fluctuation insertion 本身依赖名义传播子 \(W_k\)，所以当控制改变 \(W_k\) 时，\(V_k\) 也会随之改变。

## 2. Forward Expansion

传播状态按 fluctuation 插入次数展开。记 \(F_k^{(n)}\) 为传播到第 \(k\) 步后、总共插入 \(n\) 次 fluctuation 的状态分量，则到二阶为止：

$$
F_k^{(0)} = W_k F_{k-1}^{(0)}
$$

$$
F_k^{(1)}
= W_k F_{k-1}^{(1)}
+ V_k F_{k-1}^{(0)}
$$

$$
F_k^{(2)}
= W_k F_{k-1}^{(2)}
+ V_k F_{k-1}^{(1)}
$$

更一般地，对任意阶数 \(n\)：

$$
F_k^{(n)}
= W_k F_{k-1}^{(n)}
+ \mathbf{1}_{n>0} V_k F_{k-1}^{(n-1)}
$$

直观理解：

- \(n=0\)：没有 fluctuation insertion。
- \(n=1\)：插入一次 \(V\)。
- \(n=2\)：插入两次 \(V\)。

对应代码在 [`expansion_evolution.py`](../quantum_control/evolution/expansion_evolution.py)：

```python
def _forward_states(self, steps, initial_state):
    states = [ExpansionState({0: np.asarray(initial_state, dtype=complex)})]
    for order in range(1, self.max_order + 1):
        states[0].components[order] = np.zeros_like(states[0].components[0])

    for step in steps:
        previous = states[-1].components
        components = {}
        for order in range(self.max_order + 1):
            propagated = step.W @ previous[order]
            if order > 0:
                propagated = propagated + step.V @ previous[order - 1]
            components[order] = propagated
        states.append(ExpansionState(components))
    return states
```

代码中这些量存放在：

```python
result.forward[k].components[order]
```

也就是数学记号里的 \(F_k^{(n)}\)。

## 3. Backward Expansion

为了高效计算每个时间片的梯度，代码还会从 target state 反向传播同样的 expansion 阶数。

记 \(B_k^{(n)}\) 为从第 \(k\) 步之后的未来传播反推回来的第 \(n\) 阶 backward state，则：

$$
B_k^{(0)}
= W_{k+1}^{\dagger} B_{k+1}^{(0)}
$$

$$
B_k^{(1)}
= W_{k+1}^{\dagger} B_{k+1}^{(1)}
+ V_{k+1}^{\dagger} B_{k+1}^{(0)}
$$

$$
B_k^{(2)}
= W_{k+1}^{\dagger} B_{k+1}^{(2)}
+ V_{k+1}^{\dagger} B_{k+1}^{(1)}
$$

更一般地：

$$
B_k^{(n)}
= W_{k+1}^{\dagger} B_{k+1}^{(n)}
+ \mathbf{1}_{n>0} V_{k+1}^{\dagger} B_{k+1}^{(n-1)}
$$

对应代码在 [`expansion_evolution.py`](../quantum_control/evolution/expansion_evolution.py)：

```python
def _backward_states(self, steps, target_state):
    states_by_index = [None] * (len(steps) + 1)
    final_components = {0: np.asarray(target_state, dtype=complex)}
    for order in range(1, self.max_order + 1):
        final_components[order] = np.zeros_like(final_components[0])
    states_by_index[-1] = ExpansionState(final_components)

    for step_index in range(len(steps) - 1, -1, -1):
        step = steps[step_index]
        next_components = states_by_index[step_index + 1].components
        components = {}
        for order in range(self.max_order + 1):
            propagated = step.W.conj().T @ next_components[order]
            if order > 0:
                propagated = propagated + step.V.conj().T @ next_components[order - 1]
            components[order] = propagated
        states_by_index[step_index] = ExpansionState(components)
    return states_by_index
```

这样做的目的很简单：当只改变第 \(k\) 步的控制时，第 \(k\) 步之前的传播和第 \(k\) 步之后的传播都已经缓存好了。梯度计算只需要在第 \(k\) 步做局部导数，然后和 future backward state 收缩。

## 4. 局部导数

对某个时间片 \(k\) 和控制通道 \(a\)，只需要计算该时间片的局部 derivative state。记

$$
\delta W_k^{(a)} = \frac{\partial W_k}{\partial c_{k,a}},
\qquad
\delta V_k^{(a)} = \frac{\partial V_k}{\partial c_{k,a}}
$$

到二阶为止：

$$
\delta F_{k,a}^{(0)}
= \delta W_k^{(a)} F_{k-1}^{(0)}
$$

$$
\delta F_{k,a}^{(1)}
= \delta W_k^{(a)} F_{k-1}^{(1)}
+ \delta V_k^{(a)} F_{k-1}^{(0)}
$$

$$
\delta F_{k,a}^{(2)}
= \delta W_k^{(a)} F_{k-1}^{(2)}
+ \delta V_k^{(a)} F_{k-1}^{(1)}
$$

更一般地：

$$
\delta F_{k,a}^{(n)}
= \delta W_k^{(a)} F_{k-1}^{(n)}
+ \mathbf{1}_{n>0}
\delta V_k^{(a)} F_{k-1}^{(n-1)}
$$

对应代码在 [`expansion_differentiator.py`](../quantum_control/differentiators/expansion_differentiator.py)：

```python
@staticmethod
def _local_component_derivatives(derivative_step, previous_forward, max_order):
    derivatives = {}
    for order in range(max_order + 1):
        value = derivative_step.W @ previous_forward[order]
        if order > 0:
            value = value + derivative_step.V @ previous_forward[order - 1]
        derivatives[order] = value
    return derivatives
```

其中：

- `derivative_step.W` 对应 \(\delta W_k^{(a)}\)。
- `derivative_step.V` 对应 \(\delta V_k^{(a)}\)。
- `previous_forward[order]` 对应 \(F_{k-1}^{(n)}\)。

这一步只处理 \(c_{k,a}\) 对当前时间片的影响，不重新传播整条 pulse。

## 5. Amplitude 导数

每个 expansion order 的 fidelity amplitude 定义为：

$$
A_n = \langle \psi_{\mathrm{target}} \mid F_N^{(n)} \rangle
$$

对应代码在 [`expansion_fidelity.py`](../quantum_control/objectives/expansion_fidelity.py)：

```python
def amplitudes(self, result):
    target_state = result.backward[-1].components[0] if result.backward else None
    if target_state is None:
        target_state = result.metadata.get("target_state")
    final_components = result.forward[-1].components
    return {
        order: np.vdot(target_state, final_components[order])
        for order in range(min(self.max_order, result.max_order) + 1)
    }
```

局部导数需要和未来的 backward state 做 contraction。若局部导数贡献了 \(r\) 次 fluctuation，未来 backward state 贡献了 \(s\) 次 fluctuation，那么总阶数是：

$$
m = r + s
$$

因此：

$$
\frac{\partial A_m}{\partial c_{k,a}}
=
\sum_{r+s=m}
\left\langle
B_{k+1}^{(s)}
\middle|
\delta F_{k,a}^{(r)}
\right\rangle
$$

对应代码在 [`expansion_differentiator.py`](../quantum_control/differentiators/expansion_differentiator.py)：

```python
@staticmethod
def _derivative_amplitudes(local_derivatives, next_backward, max_order):
    derivative_amplitudes = {}
    for final_order in range(max_order + 1):
        amplitude = 0.0 + 0.0j
        for local_order in range(final_order + 1):
            future_order = final_order - local_order
            amplitude = amplitude + np.vdot(
                next_backward[future_order],
                local_derivatives[local_order],
            )
        derivative_amplitudes[final_order] = amplitude
    return derivative_amplitudes
```

这里的 `next_backward` 表示第 \(k\) 步之后的 future contraction，也就是 \(B_{k+1}^{(s)}\)，因此刚好可以和第 \(k\) 步产生的 \(\delta F_{k,a}^{(r)}\) 拼起来。

当 `max_order=2` 时，这里会显式用到 higher-order backward：

$$
\frac{\partial A_0}{\partial c_{k,a}}
=
\left\langle B_{k+1}^{(0)} \middle| \delta F_{k,a}^{(0)} \right\rangle
$$

$$
\frac{\partial A_1}{\partial c_{k,a}}
=
\left\langle B_{k+1}^{(1)} \middle| \delta F_{k,a}^{(0)} \right\rangle
+
\left\langle B_{k+1}^{(0)} \middle| \delta F_{k,a}^{(1)} \right\rangle
$$

$$
\frac{\partial A_2}{\partial c_{k,a}}
=
\left\langle B_{k+1}^{(2)} \middle| \delta F_{k,a}^{(0)} \right\rangle
+
\left\langle B_{k+1}^{(1)} \middle| \delta F_{k,a}^{(1)} \right\rangle
+
\left\langle B_{k+1}^{(0)} \middle| \delta F_{k,a}^{(2)} \right\rangle
$$

## 6. Objective 导数

当前常用配置是：

```python
ExpansionFidelity(max_order=2, drop_odd_average=True)
```

它计算的是总阶数不超过 2，并且丢掉奇数总阶项后的 perturbative fidelity：

$$
\mathcal{F}
\approx
|A_0|^2
+ 2\,\mathrm{Re}\left(A_0^* A_2\right)
+ |A_1|^2
$$

注意：这一节的 objective contraction 只负责把 \(A_n\) 和 \(\partial A_n/\partial c_{k,a}\) 组合成 \(\partial \mathcal{F}/\partial c_{k,a}\)。higher-order backward 并不是在这里直接出现，而是在上一节计算 \(\partial A_n/\partial c_{k,a}\) 时已经通过 \(B_{k+1}^{(0)}, B_{k+1}^{(1)}, B_{k+1}^{(2)}\) 进入了。

因此梯度为：

$$
\frac{\partial \mathcal{F}}{\partial c_{k,a}}
=
\frac{\partial |A_0|^2}{\partial c_{k,a}}
+
\frac{\partial}{\partial c_{k,a}}
\left[
2\,\mathrm{Re}\left(A_0^* A_2\right)
\right]
+
\frac{\partial |A_1|^2}{\partial c_{k,a}}
$$

代码实际使用更统一的双重循环。对每一对 \((\ell, r)\)，若满足保留条件，则加上：

$$
\frac{\partial}{\partial c_{k,a}}
\left(A_{\ell}^{*} A_r\right)
=
\left(\frac{\partial A_{\ell}}{\partial c_{k,a}}\right)^* A_r
+ A_{\ell}^{*}
\frac{\partial A_r}{\partial c_{k,a}}
$$

对应代码在 [`expansion_fidelity.py`](../quantum_control/objectives/expansion_fidelity.py)：

```python
def contract(self, amplitudes, derivative_amplitudes=None):
    value = 0.0 + 0.0j
    orders = range(self.max_order + 1)
    for left_order in orders:
        for right_order in orders:
            total_order = left_order + right_order
            if total_order > self.max_order:
                continue
            if self.drop_odd_average and total_order % 2 == 1:
                continue
            left = amplitudes.get(left_order, 0.0)
            right = amplitudes.get(right_order, 0.0)
            if derivative_amplitudes is None:
                value = value + np.conj(left) * right
            else:
                dleft = derivative_amplitudes.get(left_order, 0.0)
                dright = derivative_amplitudes.get(right_order, 0.0)
                value = value + np.conj(dleft) * right + np.conj(left) * dright
    return value
```

保留条件是：

$$
\ell + r \le 2
$$

并且在 `drop_odd_average=True` 时跳过奇数总阶：

$$
\ell + r \equiv 1 \pmod{2}
$$

所以最终保留下来的正是：

- \((0,0)\)：\(|A_0|^2\)
- \((0,2)\) 和 \((2,0)\)：\(2\,\mathrm{Re}(A_0^*A_2)\)
- \((1,1)\)：\(|A_1|^2\)

梯度主循环在 [`expansion_differentiator.py`](../quantum_control/differentiators/expansion_differentiator.py)：

```python
for step_index, step in enumerate(result.steps):
    previous_forward = result.forward[step_index].components
    next_backward = result.backward[step_index + 1].components
    controls = pulse.controls_at(step_index)
    t = step_index * pulse.dt

    for control_index in range(pulse.n_controls):
        derivative_step = self.step_builder.derivative_step(
            system,
            controls,
            pulse.dt,
            control_index,
            step,
            t=t,
        )
        local_derivatives = self._local_component_derivatives(
            derivative_step,
            previous_forward,
            result.max_order,
        )
        derivative_amplitudes = self._derivative_amplitudes(
            local_derivatives,
            next_backward,
            result.max_order,
        )
        derivative_value = self.objective.contract(
            amplitudes,
            derivative_amplitudes=derivative_amplitudes,
        )
        gradient[step_index, control_index] = np.real_if_close(derivative_value).real
```

## 7. 整体数据流

整个 gradient 计算可以理解成下面这条链：

$$
c
\longrightarrow
\{W_k,V_k\}
\longrightarrow
\{F_k^{(n)}\}
\longrightarrow
\{B_k^{(n)}\}
\longrightarrow
\{\delta W_k^{(a)},\delta V_k^{(a)}\}
\longrightarrow
\{\delta F_{k,a}^{(n)}\}
\longrightarrow
\left\{
\frac{\partial A_n}{\partial c_{k,a}}
\right\}
\longrightarrow
\frac{\partial \mathcal{F}}{\partial c_{k,a}}
$$

关键点是：对每个参数 \(c_{k,a}\)，不需要重新计算整条 evolution。前面用 `forward[k]` 里的历史状态，后面用 `backward[k + 1]` 里的 future state，中间只替换第 \(k\) 步的局部导数。

## 8. 最容易出问题的地方

### 8.1 \(\delta W\) 默认不是精确 Frechet 导数

默认情况下：

$$
\delta W_k^{(a)}
\approx
-i\,\Delta t\,H_a\,W_k
$$

这是一阶近似。如果 Hamiltonian 不对易，或者 \(\Delta t\)、控制幅度比较大，gradient 可能会偏。

可以用下面的配置检查：

```python
PerturbativeStepBuilder(dW_method="frechet")
```

如果 Frechet 版本和默认版本差别明显，说明一阶近似可能已经不够准。

### 8.2 优化器看到的符号可能是反的

当前 cost function 的物理含义是：

$$
\mathrm{cost}
=
\mathrm{fidelity}
-
\mathrm{penalty}
$$

但优化器内部通常做最小化，所以传给 SciPy 的可能是它的负数：

$$
\mathrm{loss}
=
-\mathrm{cost}
$$

因此如果直接看 SciPy 输出里的 \(F\) 或 loss，需要注意符号约定：优化器正在最小化的量，未必就是代码里报告的 fidelity objective。

### 8.3 Fluctuation objective 是二阶近似

这里的 fluctuation objective 不是完整的 open gate fidelity，而是二阶 perturbative approximation：

$$
\mathcal{F}
\approx
|A_0|^2
+ 2\,\mathrm{Re}\left(A_0^*A_2\right)
+ |A_1|^2
$$

如果 fluctuation 很大，或者高阶 fluctuation contribution 不可忽略，那么这个近似本身可能失效。此时 gradient 即使和该二阶 objective 一致，也不一定代表真实 fluctuation fidelity 的梯度。

## 9. 一句话总结

这个算法的核心是把 fluctuation effect 拆成按插入次数排列的 expansion components。梯度计算时，每个控制参数只影响它所在的单步 \(W_k\) 和 \(V_k\)；其余时间片通过缓存好的 forward/backward expansion 拼接起来。这样既保留了二阶 fluctuation correction，又避免了对每个参数重新做完整传播。
