# 本地模型论文预览图版

## 先看哪些文件

- `framework.png`：共同网络流水线，另有可编辑SVG。
- `m32/six_methods_6x5.png`、`m16/six_methods_6x5.png`：五数据来源、每源一条随机案例、六方法。
- `m32/ours_showcase_5x5.png`、`m16/ours_showcase_5x5.png`：每来源5条精选Ours案例，非总体统计。优先双网格MSE通过，随后按误差排序，并在同一通过/失败层内优先不同group。
- `m32/core_metrics_plus_max_error.png`、`m16/core_metrics_plus_max_error.png`：各自全61条固定测试曲线的统计，附Markdown/JSON。
- `parameters.png`：实际权重/架构/训练参数；`parameters_losses.png`：loss与teacher参数。
- `evaluation_parameters.png`：本次M32评估设置，M16设置相同但候选上限16；详细配置见各comparison.json。

## 解释边界

两个模型都保存epoch13的best权重，不等于服务器后来没有继续训练。计划60轮不能作为这两份权重的实际epoch。
每容量61条曲线（合成K4..24各1条，四外部来源各10条），不是全数据集/10%测试。
两个模型的61条输入内容hash全部一致；六方法随机对照图使用相同case ID。
每容量366条方法测量，共732条。绘图读取测量的控制点/节点/拟合曲线，不另做拟合。
图中的MISS是真实超阈值，不修改误差。精选图不替代统计表；某来源达标不足5条时仍展示真实的较好未通过案例。

误差为归一化平方欧氏残差，不开方；最大误差表列为所有有限曲线上的最坏点平方残差。
工业等距线为程序生成而非实测。M16的合成K17..24是超容量压力测试。
五文献方法均为仓库适配实现，未被验证为作者原生复现。GTX1070/CPU混合测试、完整时间单次测量，勿当作3090正式性能结论。

所有图片已经视觉检查，无水印。所选样本ID、源geometry哈希和指标均在panel_manifest.json及comparison.json中保留。
原始结果目录：`outputs/comparisons/local_models_paper_20260922_m16/` 与 `_m32/`。
