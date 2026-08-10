# 模型与工作流

## 1. 输入与编码

输入为有序采样点：

\[
Q\in\mathbb R^{B\times M\times d}.
\]

`GeometryEncoder` 将点坐标、一阶差分和二阶差分拼接后送入一维卷积网络，输出局部特征 \(F\) 和全局特征 \(G\)。

## 2. 参数预测

`ParameterHead` 预测 \(M-1\) 个正间隔并归一化，得到：

\[
0=t_0<t_1<\cdots<t_{M-1}=1.
\]

合成数据中的 `true_params` 直接监督该结果。

## 3. 独立节点 query

局部特征首先加入由预测参数 \(t_i\) 生成的位置编码：

\[
M_i=F_i+\operatorname{PE}(t_i).
\]

模型维护 \(K\) 个可学习节点 query。每个 query 对 \(M\) 做 cross-attention，并独立回归一个内部节点位置：

\[
\widetilde u_j=\delta_u+(1-2\delta_u)\sigma(w_u^Th_j).
\]

随后仅对位置排序，并同步重排 query 特征。排序不改变节点数值，因此删除一个节点不会带动其余节点重新分布。

## 4. 节点匹配与 existence

预测节点和真实内部节点均按位置有序。动态规划寻找一对一、保持顺序的最小 L1 代价匹配：

- 匹配 query：existence 标签为 1，并接受节点位置监督；
- 未匹配 query：existence 标签为 0。

`ActivityHead` 使用以下信息预测每个节点的 `log_alpha`：

- query token；
- 全局特征；
- 节点附近的局部几何特征；
- 节点位置。

Hard-Concrete 将 `log_alpha` 转为非零概率 \(\pi_j=P(z_j>0)\) 和训练门 \(z_j\)。

## 5. 可微拟合代理

训练使用三次截断幂基：

\[
\Phi=[1,t,t^2,t^3,\operatorname{stopgrad}(z_j)(t-u_j)_+^3].
\]

线性系数通过带正则的可微最小二乘求解。`stopgrad` 只切断拟合损失到 ActivityHead 的梯度：

- 拟合损失仍训练 \(t\)、\(U\)、编码器和线性求解路径；
- existence/count 损失训练结构分类；
- Hard-Concrete 门值仍实际决定设计矩阵列是否打开。

## 6. 损失与 checkpoint

当前目标为：

\[
L=L_{fit}
+0.01L_t
+0.005L_{exist}
+0.01L_{knot}
+0.002L_{count}.
\]

其中：

- \(L_t\)：预测参数与 `true_params` 的均方误差；
- \(L_{exist}\)：Hard-Concrete 非零概率 logits 的 BCE；
- \(L_{knot}\)：匹配节点位置的 Smooth-L1；
- \(L_{count}\)：期望节点数与真实节点数的归一化误差。

验证集累计 existence TP、预测数和真值数，计算全局 Precision/Recall/F1。checkpoint 先比较 existence F1，F1 相同时再比较总损失。

## 7. 部署

部署时：

1. `model.eval()` 输出 \(\pi_j\)；
2. 用固定阈值生成严格 0/1 门；
3. 物理删除未保留节点；
4. 构造标准开放三次 B 样条节点向量；
5. 用增广最小二乘重新求解控制点。

训练代理的截断幂系数不会当作最终 B 样条控制点使用。

## 8. 当前边界

- 当前训练数据为合成开放三次 B 样条；尚未接入真实 CAD 数据。
- 模型最多输出固定数量 \(K\) 的候选节点，默认 \(K=8\)。
- 阈值应在验证集确定，不能针对每条测试曲线单独调节。
- 旧 interval、pilot 和 legacy-soft 路径仅用于 checkpoint 兼容。
