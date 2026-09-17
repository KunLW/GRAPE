# 独立涨落修复：方案与执行状态

日期：2026-09-15。相关讨论：优化端（perturbative expansion）曾把所有
`FluctuationTerm` 矩阵求和成单一 `H_fluc` 后做统一二阶展开，等价于全部噪声
源共享一个标准正态变量；`faithful_gate_fidelity` 则对每源独立积分。
X/−X 双源构造暴露差异：优化端 1.000000，精确值 ½(1+e^{−4σ²T²}) = 0.980395
（σT = 0.1）。交叉项 E[ξ_a ξ_b] 被错误地当作 1 而非 δ_ab。

## 决策

两个候选方案：

- **方案 A（已采用）——按项展开**：每个源保留自己的插入链，
  F = F_closed + Σ_a ( |A1_a|² + 2Re(A0*·A2_aa) )，交叉项自然消失。
  语义与 faithful 端已声明的独立模型对齐；不改 YAML/系统声明，向后兼容。
- **方案 B（暂不做）——相关组**：`FluctuationTerm` 加 correlation-group
  标签，组内共享变量、组间独立。是 A 的超集（任意高斯协方差可经 Cholesky
  分解为独立组表达），但接触面大（noise 词汇表 → YAML → 报告全链）。
  在没有真实相关噪声需求前不实现；完全相关的一组噪声可手工把算符加权
  求和声明为单个 term（穷人版表达，代价是 error budget 无法拆看组内单项）。

## 实现（已完成，工作区未提交）

实际实现用**通道轴广播**而非 dict 键扩展，比原草案更紧凑：

- `steps/perturbative_step.py` — `V` 为 `(n_noise, d, d)` 堆叠（0/1 源时保持
  `(d, d)`）；`_fluctuation_hamiltonian` / `_fluctuation_control_derivative`
  按源返回，control 源保留零占位以维持与控制通道的位置对齐。
- `evolution/expansion_evolution.py` — 1/2 阶分量带通道轴 `(n_noise, d)`，
  `apply_operator` einsum 广播；同通道双插入天然成立；多源时
  `max_order > 2` 显式报错。
- `objectives/expansion_fidelity.py` — `contract` 对通道轴求和
  （Σ|A1_a|²，绝不 |ΣA1_a|²）；多源下 `drop_odd_average=False` 显式报错。
- `differentiators/expansion_differentiator.py` — einsum 版收缩，梯度逐通道。
- `evaluation/density_matrix.py` — faithful 端跳过零强度源的积分维
  （节点数 hermite^n_active），并保留位置对齐。
- `diagnostics/error_budget.py` — V 插入校验逐通道比较（合并比较会因
  抵消掩盖误差）。
- `experiments/driver/run_experiment.py` — kappa_2 改为
  sqrt(Σ_a ||G_a||₂²)（范数和的范数同样会假抵消）；
  `perturbative_fidelity_terms` 按新收缩计算，报告标注
  `fluctuation_average: independent channels`。
- `run_error_budget.py` — 单源系统用"其余项系数置零"而非删除项，
  避免 control 源位置重映射（旧写法会把 alpha2 误配到 alpha1）。
- 文档 — README、`doc/fluctuation_gradient.md`（重写为独立噪声二阶平均与
  梯度推导）、`doc/report.tex`（独立通道假设写入公式，注明旧数值表格
  产生于修复前）。`density_matrix.py` 误导性 "same convention" docstring 已随
  改动消除。

**验证状态**：全套 82 个测试通过（原 73 + 新增 `tests/test_independent_fluctuations.py` 9 个），覆盖：

1. X/−X 双源：faithful 精确匹配 ½(1+e^{−4σ²T²})，优化端符合二阶独立公式
   （同片二阶项仍按既有约定省略，测试显式锁定 1 − 2σ²(1−1/n)）；
2. static+control 混合系统梯度 vs 有限差分（leading 与 frechet 两种 V）；
3. 修复后的可加性验收：多源 value 与 gradient 等于单源修正之和
   （1e-14 级），符号翻转与项重排不变；
4. 优化器 / evaluator / 日志 / 诊断四条路径同一平均值；
5. 多源 max_order>2、drop_odd_average=False 显式失败而非静默错误。

## 剩余步骤

- [ ] 提交：15 个修改文件 + `tests/test_independent_fluctuations.py`
      （建议单独分支 + PR，说明数值行为变更）。
- [ ] 用修复后引擎重跑 flattop_122us / constant_pulse 的 error budget 与
      robustness 扫描：确认 additivity residual 降到浮点零（修复前
      0.24% / 0.73% 的交叉项占比即预期数值偏移上限），核对鲁棒性排序。
- [ ] 决定已提交的历史 report（robustness_eval/flattop_122us 等）是否
      重生成或仅按 README 的声明保留原值。
- [ ] `doc/report.tex` 复核新增推导（仓库规则：疑似数学错误用 `% REVIEW`
      标注，不直接改公式）。
- [ ] （远期，仅当出现真实相关噪声需求）方案 B：correlation group +
      faithful 端按组积分（组数 < 项数时积分维数还会下降）。
