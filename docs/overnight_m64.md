# M64训练：64个候选内部节点

M64表示最大候选内部节点数64；三次夹持样条完整节点向量最多72项，控制顶点最多68个。最终KeepMask可以保留更少节点。没有将合成源复杂度改成K64：仍用K4..24，以便与M32比较。

新支持 `--capacities 64`，M16/M32默认行为不变。没有可确认的M64初始化权重时，显式 `--from-scratch`，绝不将M32权重假装成M64。

```bash
cd /home/feng/HouCode/SplineFitting_1070_overnight
bash scripts/run_v16_paper_training_linux.sh \
  --capacities 64 \
  --from-scratch \
  --epochs 112 \
  --proposal-epochs 64 \
  --joint-geometry-calibration-epochs 8 \
  --train-size 3000 --val-size 500 --batch-size 32 \
  --device cuda \
  --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --prepare-real-data \
  --real-test-fraction 0.1 \
  --baseline-protocol adaptation \
  --run-prefix paper_coupled_clean_3090_m64_scratch_r1
```

从零初始化因此建议64 Proposal + 48 Joint（共112轮），不是已验证的最优轮数或达标承诺；学习率、teacher、mask选择、2+2耦合、MSE5e-5与峰值目标5e-4沿用当前训练配置。若有已训练M64，可用 `--m64-checkpoint 路径` 替换 `--from-scratch`，并使用原12+48日程。不能同时称从零初始化与warm start。

默认串行训练→合成K4..24每K10条和四外部来源10%test→六方法统计→全部已测案例图。六方法本轮容量都设64。统计包含完整耗时、MSE、通过率、平均K和最大平方误差；network时间另存。绘图含控制顶点、内部节点和曲线，不伪造失败几何。

输出检查点：`outputs/checkpoints/paper_coupled_clean_3090_m64_scratch_r1_m64.pt`。
同名结果存在时拒绝覆盖；用新run-prefix而不是删除旧权重。

已有模型仅评估：

```bash
bash scripts/run_v16_saved_models_evaluation_linux.sh \
  --capacities 64 \
  --m64-checkpoint outputs/checkpoints/paper_coupled_clean_3090_m64_scratch_r1_m64.pt \
  --device cuda --data-root /home/feng/HouCode/SplineSurpervisedFitting/data \
  --real-test-fraction 0.1 --selection-seed 20260922 --tag test10pct_20260922
```

M64随机初始化与M32 warm start不是严格的容量单变量消融。增大候选容量可能提高召回，也可能保留更多冗余；结论须由实际测试给出。
