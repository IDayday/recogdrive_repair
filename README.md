# ReCogDrive 表征修复规律分析代码

本仓库整理自 `recogdrive_repair_pattern_analysis_full.zip`，只提交 Python 源码、依赖说明和路径来源信息。评分矩阵、逐场景表格、修复参数、模型权重、轨迹、图像与原压缩包均保留在本地。

## 做了什么分析

分析对象为本地重建的 **D / 先聚类 / K8 / D48 / 增强教师**：完整 Train 85,813 场景、130 个物理日志，完整 Test 12,146 场景、44 个物理日志，以及 1,654 个成功的源拟合配对。每场景使用固定 S0–S4 五组噪声；先在场景内平均评分，再从 Base 和八个修复候选中选择一个分支。

| 分析 | 代码 | 主要内容 |
|---|---|---|
| 评分与修复收益 | `analysis/outcomes/` | 分支使用率、PDMS 与七分量、安全失败恢复、新增失败、近优候选、噪声稳定性、候选子集消融、评分代数 Shapley 分解、Train/Test 难度组成 |
| 表征几何 | `analysis/representation/analyze_mechanisms.py` | 成功减失败的共享 384 维接口差分、先聚类后投影、D48 能量、方向簇紧致度及算子结构 |
| 实际施加的修复向量 | `analysis/representation/analyze_selected_shifts.py` | 实际裁剪后的 image/text 加法向量、未中心化及中心化谱、固定向量复用、跨 Train/Test 方向一致性 |
| 场景语义与外生分组 | `analysis/analyze_scene_metadata.py`、`semantic_followup.py` | 从原始日志读取速度、导航命令、交通密度、前方停驶车、GT 起步/刹停；关联效应量、完整分组中的固定候选效果、时间戳敏感性 |
| 源簇与最终选择 | `analysis/representation/analyze_fit_cluster_semantics.py` | 拟合簇场景组成、所属簇专家与最终 Oracle 的一致性和 regret |
| 轨迹行为与 GT 偏差 | `analysis/trajectories/` | Base/教师轨迹的纵横向变化、路径长度、转向、速度、ADE/FDE、沿 GT/横 GT 误差、同一转向情境下的不同 Base 误差 |
| 区间、图表、案例和报告 | `bootstrap*.py`、`*bootstrap.py`、`analysis/build_deliverable.py` | 按物理日志重采样的置信区间、Train/Test 对照、确定规则选择的定性案例、PNG/PDF/SVG 和 Markdown/HTML |

编号须区分：**Base=choice0；E1–E8=choice1–8=slot0–7；C0–C7 是源拟合簇**。五组噪声不算五个独立训练种子。统计是固定模型和候选库上的探索性描述；Oracle 教师分数不是已部署路由器或学生模型成绩，选中分支的语义富集也不直接证明因果能力模块。

原报告讨论的主要现象包括：方向簇具有运动情境结构；同一转向情境可有不同规划误差；部分选中分支画像在 Train/Test 重复；无条件应用单个修复可能降低评分；PDMS 提高和更接近人类 GT 轨迹并不等价。完整证据继续从本地原包或原分析目录读取。

## 源码范围与整理方式

从原包提取全部 **30 个 Python 文件**：16 个分析脚本、冻结教师解释器、12 个历史数学源码及原打包脚本。另外保留分析引用的 `native_cap.py` 与 `bank.py` 两个接口源码。

新增 `discover_local_assets.py`、`recogdrive_paths.py` 和 `run_analysis.py`，用于发现本地输入、配置路径、按依赖顺序运行。分析脚本仅调整路径读取和输出位置，保留原统计计算与随机种子。`SOURCE_PROVENANCE.json` 记录原 ZIP 的 SHA256、每份源码在包中的位置、原本地路径与原始 SHA256。

`source_evidence/` 中的教师、拟合、算子及打包代码是历史来源参考；部分模块还依赖原项目结构和外部资产。常规分析使用下面的运行入口。此仓库针对所指定 ZIP 的第一轮分析，后续 `repair_patterns_round2_20261103` 是另一批实验。

## 先让 Codex 查找本地文件

当前机器已定位到：

| 用途 | 本地路径 |
|---|---|
| 原始完整 ZIP | `/mnt/project/VLA-AD/outputs/recogdrive_repair_pattern_evidence_20261103/recogdrive_repair_pattern_analysis_full.zip` |
| 原始分析目录 | `/mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027/engineering/repair_patterns_20261103` |
| 实验根目录 `native_root` | `/mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027` |
| 原代码及场景清单 `code_root` | `/mnt/project/recogdrive-no-routing-round1-20261026` |
| 原模型路径配置 | `/mnt/project/recogdrive-teacher2-a800-20261026/paths.local.json` |

换工作区时，不要求用户逐项提供路径。先运行：

```bash
python discover_local_assets.py --roots /mnt/project /mnt/navsim --output paths.local.json
```

该工具浅层查找实验根、原代码、ZIP 和历史模型路径配置，生成只含路径的本地 JSON，并报告必要输入是否存在；不扫描整个海量缓存树，不加载模型。`paths.local.json` 已被 Git 忽略。已存在配置不会被覆盖；需要重找时另指定输出文件。

若工具没有找到，Codex 应先用 `rg --files` 在项目的 `experiments/`、`outputs/`、`configs/` 和原代码目录中定向搜索，再读取 `BUNDLE.json`、`library.json`、历史 `paths.local.json` 及缓存索引中的引用。需要扩展搜索时给 `--roots` 增加根目录，或直接编辑 JSON。文件名相同并不足以确认模型或教师身份。

## 要读取哪些数据

以下相对路径均相对于 `native_root`，除非另作说明：

| 输入 | 内容与用途 |
|---|---|
| `requests/{train,test}/MATRIX.npz` | `scores[N,9,5,7]` 为 FP64 原始评分，另含 `base_scores`、`indices`；评分、消融、噪声与安全分析读取它 |
| `frozen_teacher/selections/{train,test}.npz` | `tokens`、`indices`、`choice[N]`、`q_points[N,9]`、`selected_actual_role_shifts[N,2,384]`、`selected_actual_RMS` 等；固定选择与实际修复向量 |
| `frozen_teacher/BUNDLE.json`、`library.json` | 教师身份、源模型身份、专家方法、簇/slot 映射、alpha、cap 和外部源码引用 |
| `frozen_teacher/space.npz`、`slot00.npz` 至 `slot07.npz` | D48 基、中心及冻结专家参数；分析读取，不上传 Git |
| `fitting/ALLTRAIN/SPACE_AND_CLUSTER.pt` | 完整源拟合差分、聚类、子空间与几何状态 |
| `fitting/INNER/SPACE_AND_CLUSTER.pt`、`fitting/ALLTRAIN/CLUSTER_TRANSFER.pt` | 内层及最终重拟合聚类的对应关系 |
| `fitting/ORDINARY_LOCAL.npz` | 普通输入池及 `raw_aux24_float32`，用于表征/状态对照；原完整 ZIP 未包含此文件，需要在本地实验目录寻找 |
| `engineering/independent_full_teacher_effects_20261027/{TRAIN,TEST}_SCENES.csv` | token 到物理 `log_id` 的对应关系；逐日志 bootstrap 与评分分析读取 |
| `engineering/candidate_pool_analysis_20261028/recogdrive_nine/CANDIDATES.npz` | Test 九候选、五噪声轨迹 `[12146,9,5,8,3]` 及对应评分和 token |
| `requests/train/REPAIRED_CACHE_INDEX.jsonl` | 指向 Train 每场景原始候选 NPZ，并提供 SHA256；无派生轨迹缓存时逐文件提取 Base/选中教师 |
| `code_root/manifests/{train,test}.csv.gz` | token、原始日志相对路径、帧索引与会话信息 |
| `code_root/interface_teacher/native_cap.py` | 实际修复向量定义的源码来源与哈希 |
| `/mnt/navsim/trainval_navsim_logs`、`/mnt/navsim/test_navsim_logs` | 原始场景 pickle；场景语义、GT 轨迹、对象信息、时间戳及案例 |
| `/mnt/navsim/trainval_sensor_blobs`、`/mnt/navsim/test_sensor_blobs` | 原始图像；定性案例绘制读取前视图 |
| `frozen_teacher/{TRAIN,TEST}_EVALUATION.json` | 报告生成阶段读取的冻结整体成绩 |

评分最后一维顺序为 `PDMS, NC, DAC, TTC, Progress, Comfort, DDC`。`q_points = scores[...,0].mean(axis=2) * 100`；`choice` 是九候选按上述分数执行 Base 优先的精确 argmax。轨迹末两维 `[8,3]` 为八个未来点的局部 `x,y,heading`，名义间隔 0.5 秒，分析另检查实际时间戳。

可复用原分析目录中的 `scene_metadata/{train,test}_metadata.npz` 与 `trajectories/TRAIN_SELECTED_TRAJECTORIES.npz`。后者保存 `[85813,2,5,8,3]`，其中候选轴为 Base 与实际选中教师。复用可免去重新读取全部原始日志和 85,813 份 Train 原始候选文件。案例绘制仍会读取少量原始日志和图像。

JSON 和 JSONL 内可能保留绝对路径。迁移数据时，顶层路径配置不会自动改写这些内部引用；应保留原目录映射，或在独立输入副本中调整引用。不要覆盖原冻结证据。原完整 ZIP 包含部分数值输入和完整派生轨迹，但不包含全部原始日志、图像、模型与 Train 全候选逐文件缓存，不能仅凭 ZIP 承诺完全从原始输入重算。

## 要用什么模型

**离线统计入口不加载规划模型或 VLM，也不执行新推理/评分。** 它读取上述模型已经生成并保存的数值资产。Torch 用于在 CPU 上读取拟合状态 `.pt`。地图和 metric cache 是上游生成评分时使用的输入，本仓库的离线统计不读取它们。

需要追溯或重新生成上游资产时，模型身份为：

- Base：`ReCogDrive-2B-RL`，本地 checkpoint：`/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt`。
- checkpoint SHA256：`23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20`。
- VLM：历史实际配置指向 `/mnt/project/recogdrive-representation-teacher-1/local/models/ReCogDrive-VLM-2B`。
- 本地冻结教师：`native_root/frozen_teacher/BUNDLE.json`，SHA256 为 `ed0a1bef8b27d89e7bb4eb8f32c93134204d66c42184ddecaa1831c5a7188dfc`；不是直接套用旧 PPU 教师参数。
- 上游环境与模型数值协议来源：`/mnt/project/recogdrive-teacher2-a800-20261026/local_rebase_20261027/CONTRACT.json`；上游推理/评分代码继续从原项目寻找。

上游配置还记录了地图 `/mnt/navsim/maps`，Train metric cache `/mnt/project/recogdrive-representation-workspace/115033c_a800_v1/_data/metrics/navtrain`，Test metric cache `/mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/metric_cache_navtest_v1`。路径工具会读取历史配置中的实际值。重生成模型输出、重新拟合或重评分属于额外工作，应按用户新指令执行。

## 如何使用

本机已有环境 `/root/miniconda3/envs/navsim/bin/python`。另建环境时可安装 `requirements.txt`；其中列出本机离线分析环境的版本，Torch 可按当地平台安装对应构建。

```bash
python -m pip install -r requirements.txt
python discover_local_assets.py --output paths.local.json
python run_analysis.py --list
```

生成配置后，确认 `analysis_root` 是独立输出目录。它不能位于本仓库或原输入树中。默认输出为 `/mnt/project/recogdrive-repair-analysis-output`。`native_root`、`code_root`、日志与图像路径为输入，`analysis_root` 为输出；其他模型路径仅供来源说明。

只重算完整评分/安全/候选统计：

```bash
python run_analysis.py --config paths.local.json --stages outcomes
```

独立重算实际修复向量或源差分几何：

```bash
python run_analysis.py --config paths.local.json --stages shifts mechanisms
```

用本地已有派生缓存重算全部统计、图表和报告：

```bash
python run_analysis.py --config paths.local.json --stages all \
  --reuse-derived /mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027/engineering/repair_patterns_20261103
```

`--reuse-derived` 只复制元数据、Train 已选轨迹及相应来源记录，不会通过符号链接回写旧目录。省略此参数则从原始日志/索引提取必要缓存，首次运行会读取大量数据。

阶段顺序：`outcomes → mechanisms → shifts → metadata → semantics → trajectories → gt → fit-planning → report`。`outcomes`、`mechanisms`、`shifts` 可单独执行；其余阶段会消费前面生成的文件。只选一个后续阶段时，可复用同一输出目录已有结果，或增加 `--with-dependencies` 自动补跑依赖。`--dry-run` 只显示计划，不运行脚本。

```bash
python run_analysis.py --config paths.local.json --stages report --with-dependencies --dry-run
```

产物写入 `analysis_root`：评分统计在 `outcomes/`，表征分析在 `representation/`，场景元数据在 `scene_metadata/`，轨迹分析在 `trajectories/`；主报告为 `REPAIR_PATTERN_REPORT.md`，浏览入口为 `EVIDENCE_BROWSER.html`。`logs/` 保存每步日志，`RUN.json` 记录本次执行的脚本、源码 SHA 与退出码。

`build_deliverable.py` 保留原论文证据报告的固定人口、教师身份断言及历史叙述，适合复现这次分析。更换模型、教师或场景人口后应同步修改报告文本与身份约束；不能将旧叙述直接作为新实验结论。原包中部分人工撰写的补充说明及 `PAPER_CLAIMS.csv` 不由这些脚本自动生成，继续从原 ZIP 读取。

## 本次交付

源码整理与路径工具已提供。按用户要求，不继续验证或重跑分析；本仓库不声明本次完成全流程复现。数据、模型、原分析结果与已停止的训练任务继续保留在原位置。
