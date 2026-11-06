# 使用本仓库的 Codex

先读 `README.md`。这是指定 `recogdrive_repair_pattern_analysis_full.zip` 的离线分析代码仓库。

- 用户希望 Codex 自己定位本地数据和模型。先运行 `discover_local_assets.py`，再按 README 的目录和资产文件名定向查找；不要一开始就要求用户逐项提供路径。
- 读取本地历史 `paths.local.json`、`BUNDLE.json`、`library.json`、manifest 与索引来确定身份。找到相似文件名时，继续核对来源和模型/教师 SHA。
- 输入资产保持原样，输出写独立的 `analysis_root`。本仓库的 Git 提交只包含源码、文档及小型配置/来源说明，不提交数据、权重、轨迹、图像、原始结果或 ZIP。
- 离线分析不会加载规划模型、启动 GPU 推理、重评分、重新拟合或恢复训练。若用户要求这些工作，再去原项目定位代码和当前资源约束。
- `source_evidence/` 是历史源码参考，不是默认执行入口。
- 仅按当前用户要求执行验证。此次交付用户明确表示“不需要再验证”，不要为了发布这些代码继续跑验证或分析。
