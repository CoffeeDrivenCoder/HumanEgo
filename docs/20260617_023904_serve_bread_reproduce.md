# HumanEgo serve_bread 复现步骤

目标：基于官方释放的 `serve_bread` 数据集复现 HumanEgo 的
flow-matching 模型训练。当前阶段只需要数据和 GPU，不需要机器人硬件。

## 当前环境

- 项目路径：`/home/ubuntu/projects/wangk/HumanEgo`
- Conda 环境：`/data/wangk/conda/envs/humanego`
- 数据根目录：`/data/wangk/data`
- Checkpoint 目录：`/data/wangk/checkpoints/humanego/serve_bread/HumanEgo`
- 已下载任务：`serve_bread`
- 数据状态：`/data/wangk/data/serve_bread/aria` 下已有 61 条 recording
- 训练配置：`cfg/training/serve_bread/HumanEgo.yaml`

官方配置会将 `mps_serve_bread_000_vrs` 作为 eval，其余 60 条作为 train。

## 1. 检查数据

```bash
cd /home/ubuntu/projects/wangk/HumanEgo

find /data/wangk/data/serve_bread/aria -maxdepth 1 \
  -type d -name 'mps_serve_bread_*_vrs' | wc -l

test -d /data/wangk/data/serve_bread/aria/mps_serve_bread_000_vrs/preprocess/all_data
```

期望结果：

- 第一条命令输出 `61`
- 第二条命令退出码为 `0`

官方数据已经带有预处理结果：
`preprocess/all_data/*/training_data.json`。因此复现官方训练时不需要重新跑
preprocess。

## 2. 可选：数据读取检查

```bash
/data/wangk/conda/envs/humanego/bin/python -m training.FlowMatchingDataloader \
  --mps_path /data/wangk/data/serve_bread/aria/mps_serve_bread_000_vrs
```

只有在训练构建 dataloader 报错时再跑这一步。

## 3. 启动训练

建议放在 `tmux` 里跑，避免 SSH 断开导致训练中断：

```bash
tmux new -s humanego_train
cd /home/ubuntu/projects/wangk/HumanEgo

/data/wangk/conda/envs/humanego/bin/python -m training.FlowMatchingTrainer \
  --task serve_bread \
  --use_cfg \
  --job HumanEgo \
  --data_root /data/wangk/data \
  --out_dir /data/wangk/checkpoints/humanego/serve_bread/HumanEgo
```

退出 tmux 但保持训练运行：

```bash
Ctrl-b d
```

重新进入 tmux：

```bash
tmux attach -t humanego_train
```

## 4. 查看训练状态

```bash
tmux attach -t humanego_train
```

或查看输出文件：

```bash
ls -lh /data/wangk/checkpoints/humanego/serve_bread/HumanEgo
tail -f /data/wangk/checkpoints/humanego/serve_bread/HumanEgo/train_history.json
```

主要输出：

- `latest.pt`：自动恢复用 checkpoint
- `best.pt`：当前最佳 checkpoint
- `config.json`：训练配置，inference 需要
- `dataset_stats.json`：归一化统计，inference 需要
- `train_history.json`：每个 epoch 的指标
- `train_curve.png`、`eval_curve.png`：loss 曲线
- `eval_render/`：teacher-forced 可视化评估结果

## 5. 恢复训练

训练脚本会自动从下面的文件恢复：

```bash
/data/wangk/checkpoints/humanego/serve_bread/HumanEgo/latest.pt
```

如果训练中断，重新执行第 3 步的训练命令即可。

## 6. 说明

- 训练阶段与机器人硬件无关。
- 机器人只在后续 real-world inference / deployment 阶段需要。
- 快速 smoke test 可以在训练命令后加：`--epochs 1 --data_num 2`。
- 完整复现建议保持官方配置：400 epochs、batch size 32、AMP 开启。
