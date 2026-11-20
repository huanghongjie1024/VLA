# HiddenPhys 轴孔插入与 LeRobot ACT

本目录包含 HiddenPhys 轴孔插入的 ACT 训练与评估、π0 LoRA 脚本和回放分析。

使用服务器的 [lerobot 依赖清单](../requirements-lerobot.txt) 准备 Linux Python 环境，再在本目录安装 ForceHarness 的仿真依赖。

    python -m pip install -r ../requirements-lerobot.txt
    python -m pip install -e "./ForceHarness[sim]"

run.sh 默认使用当前环境的 Python 和上一级目录的 LeRobot 源码。可通过 HIDDENPHYS_PYTHON 指定解释器，通过 LEROBOT_ROOT 指定其他 LeRobot 工作树。

- 示范来自 [Jack97/ForceHarness-Data](https://huggingface.co/datasets/Jack97/ForceHarness-Data) 的 `raw/peg_in_hole_hard`。把该任务的 185 个 `episode_*.npz` 放入本目录的 `demos/`。数据可访问时运行 `hf download Jack97/ForceHarness-Data --repo-type dataset --include 'raw/peg_in_hole_hard/*.npz' --local-dir downloads`，文件会写到 `downloads/raw/peg_in_hole_hard/`。
- 仿真使用 [ForceHarness 源码](ForceHarness)，以 [上游提交 733407c](https://github.com/showlab/ForceHarness/commit/733407c1ea90d8b474dff7809d9349d0f5f0b1d5) 为基础。仿真依赖包含 robosuite 1.5.1 和 MuJoCo 3.2.7。
- `experiment.py` 直接实例化 LeRobot 的 `ACTPolicy`。两个模式均输入两路当前 RGB 和当前机器人/场景状态。current 模式将当前六维腕力重复 8 次，history 模式输入最近 8 个控制步的真实腕力，包含当前帧。20 Hz 下第一帧距当前 350 ms。回合开头用首帧填充，窗口不会跨越回合。
- 动作窗口长度为 24 步，每次执行前 8 步。13 维动作保留 7 维 OSC 位姿/夹爪命令和 6 维刚度。归一化从训练示范计算并和权重一起保存到 `checkpoint.pt`。另存的 `lerobot_policy/` 用于读取 ACT 模型参数和模型配置。
- 评估直接使用官方 HiddenPhys rollout 客户端。服务监听回环地址。相同测试种子 9100–9149 用于两组比较，输出每局成功标记、峰值力和配对成功率差。

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh smoke
CUDA_VISIBLE_DEVICES=0 bash run.sh train current
CUDA_VISIBLE_DEVICES=0 bash run.sh train history
CUDA_VISIBLE_DEVICES=0 bash run.sh eval current
CUDA_VISIBLE_DEVICES=0 bash run.sh eval history runs/paired/history 9100-9149 47182
bash run.sh compare
# 或运行整个配对实验，默认分配 GPU 0 和 1
bash pair.sh
```

默认训练 5,000 步、batch 32、训练种子 42，使用同一 ImageNet ResNet-18 初始权重。可用 `STEPS`、`BATCH` 和 `DATA_WORKERS` 调整，并为新实验提供新的输出目录。冒烟训练使用 8 条示范和 12 次更新，评估 1 局，仅验证接口。相同评估目录会跳过已有的种子，重试出错的回合时用 `EVAL_DIR` 指定新目录。结果和日志在 `runs/` 及 `logs/`。


## π0 LoRA 与回放分析

[pi0-lora-peg](pi0-lora-peg) 包含数据格式转换、归一化统计和训练入口。脚本默认在同一仓库的 openpi 目录中查找 OpenPI 环境。可通过 OPENPI_ROOT、PI_PY 和 PI_SIM_PY 选择对应的源码位置与解释器。示范和初始权重按 OpenPI 与 ForceHarness 的说明准备。

[analysis/replay_1119](analysis/replay_1119) 包含回放和对照分析脚本。后续实验代码可以按本目录单独提交，生成的检查点、日志和数据由 .gitignore 排除。
