# Codex 任务说明：基于 DDAPO 论文在 ms-swift GRPO 中实现注意力奖励

## 一、角色与目标

你是实现 RLHF/GRPO 与多模态大模型训练的工程师。本任务**不是**把 trl_DAPO 的代码简单迁移到 ms-swift，而是**以 DDAPO 论文为唯一公式与变量名依据**，在 ms-swift 的 GRPO 框架下，构建一套**可复现、可扩展的实验代码**。

- **trl_DAPO**：仅作**流程与结构参考**（例如：何时取 attention、如何区分 text/vision token）。**禁止照搬其变量名与命名习惯**，请推翻/替换为论文中的符号与命名。
- **DDAPO.pdf**：**公式、指标定义、变量名**必须严格以该论文为准。若工作区中存在 `DDAPO.pdf` 或同名文档，请优先从中提取 Methods/公式与符号；若不存在，则依据论文标题与常见引用（如 Debias-DAPO 或 DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization 等）中的标准符号实现，并在日志中注明所参考的章节或公式编号。
- **当前阶段**：先**只兼容 ms-swift 的 GRPO**（即通过自定义 reward 函数接入，不改造 PPO 或其它算法）。

---

## 二、技术约束与接口

1. **框架**：ms-swift，入口为 `swift rlhf --rlhf_type grpo`，训练器为 `swift.rlhf_trainers.grpo_trainer.GRPOTrainer`。
2. **奖励接入方式**：采用 **GRPO 自定义 Reward 函数**（与 `examples/train/grpo/plugin/plugin.py` 一致）：
   - 实现一个继承 `swift.rewards.ORM` 的类，实现 `__call__(self, completions, **kwargs) -> List[float]`。
   - 在插件中注册到 `orms['<name>']`，训练时通过 `--reward_funcs <name>` 与 `--external_plugins <path>` 启用。
3. **注意力来源**：当前 GRPO 的 reward 阶段**只有** `completions` 等文本/列数据，**没有**现成的 attention。因此需要你在**该奖励函数内部**（或通过 kwargs 注入的 model/processor）对「prompt + completion」做**一次** Forward，并传入 `output_attentions=True`，用得到的 attention 按 **DDAPO 论文中的公式** 计算标量奖励。
4. **Rollout 引擎**：为保证能拿到 attention，本实验要求 **不使用 vLLM 做 rollout**（即 `use_vllm=False`），使用 Transformers 路径，以便后续若需要可在同模型上做带 attention 的 forward。
5. **多模态**：若支持 VLM（如 Qwen2-VL），需正确区分 **text tokens** 与 **vision tokens**（或论文中的等价概念），并按论文中的定义计算指标（例如对生成 token 对 prompt 中某类 token 的注意力聚合方式）。

---

## 三、实现清单（请你按顺序完成并写日志）

1. **从 DDAPO 论文提取规范**
   - 在仓库中查找 `DDAPO.pdf` 或同名文档；若存在，阅读 Methods/公式部分，列出：
     - 论文中使用的**主要变量名**（如对 attention 的记号、对 text/vision 的记号、对 reward/指标的记号）。
     - **核心公式**（如基于 attention 的指标如何从 attention map 得到标量、是否有 token 级权重等）。
   - 若不存在 PDF，则根据论文标题/引用采用该方向常用符号，并在日志中写明「未找到 DDAPO.pdf，采用 XXX 论文第 X 节公式」。

2. **新建 DDAPO 注意力奖励模块（严禁使用 trl_DAPO 的变量名）**
   - 在 `examples/train/grpo/plugin/` 下新建 `ddapo_attention_reward.py`（或你认为更合适的文件名）。
   - 实现内容：
     - 使用**论文中的变量名**实现：从 attention 计算论文中的指标（例如某种 balance/ratio，或 token 级权重再聚合）。
     - 实现一个 `ORM` 子类，在 `__call__` 中：
       - 从 `kwargs` 获取能访问 **policy model** 与 **processor** 的途径（若当前 GRPO 不传，需在下方第 3 步中说明如何在 ms-swift 中注入，例如通过 `trainer_state` 或全局/闭包）。
       - 对每个 sample 的 prompt+completion 做一次 forward（`output_attentions=True`）。
       - 根据 **DDAPO 论文公式** 从 attention 计算标量奖励，返回 `List[float]`。
     - 多模态：若为 VLM，按论文定义区分 text/vision（或等价的 token 类型），并只使用论文中出现的概念与命名。

3. **与 GRPO 的衔接**
   - 若 ms-swift 的 `_compute_rewards_per_func` 在调用 reward 时**未**传入 `model`/`processor`/`trainer`，请你在日志中明确写出：
     - 需要在哪一文件、哪一函数、以何种方式（例如在 `reward_kwargs` 中加入 `trainer` 或 `model`）注入，以便 reward 函数能拿到 policy model 做 forward。
     - 并给出**最小改动**的补丁（只写需要改动的片段与位置），便于后续在 ms-swift 主库中提交或做实验分支。
   - 在 `plugin.py` 中 import 并注册你写的 DDAPO 奖励类，例如 `orms['ddapo_attention'] = DDAPOAttentionORM`。

4. **可运行示例**
   - 在 `examples/train/grpo/` 下提供或更新一个可运行脚本（如 `run_grpo_with_ddapo.sh` 或 Python 脚本），要求：
     - `--rlhf_type grpo`
     - `--reward_funcs` 包含你注册的 DDAPO 奖励名（可与 accuracy、format 等组合）
     - `--use_vllm false`
     - 其他参数（如 dataset、model、batch size）可写死为小规模示例，便于单机双卡快速跑通。

5. **通讯日志（必须严格遵守）**
   - 所有上述实现与修改，必须写入仓库根目录下的 **`codex_dev.log`**（与本文件同级的上一级目录即 `examples/train/grpo/` 的父级下，即 ms-swift 根目录）。
   - 日志采用**追加**方式，每次完成一个「逻辑块」（例如一个文件的新建/修改、或一组相关改动）就写一段。格式如下，以便后续 Agent 解析与协作：

```text
## [TIMESTAMP] ISO8601 格式，如 2025-02-24T12:00:00Z
### File: `相对路径或绝对路径`
### Action: `Create | Modify`
### Description: 一两句话说明本段修改的目的和与 DDAPO 论文的对应关系（如：实现论文式 (3) 中的 XXX 指标，变量名与论文一致）。
### Code:
```<language>
<粘贴完整或关键代码片段，便于后续直接应用或审查>
```
---
```

   - 若某一步**没有代码**（例如仅结论或依赖说明），可省略 `### Code:` 块，但必须保留 `### Description:`。
   - 请在本次任务**结束时**在日志中追加一行总结块：
     - `## [TIMESTAMP] SUMMARY`
     - 列出：已实现文件、已修改文件、尚未实现或需要后续 Agent 完成的事项（如：DDAPO 的 token 级 loss 与 GRPO 的 loss 融合，留作后续）。

---

## 四、禁止与建议

- **禁止**：直接复制 trl_DAPO 中的变量名（如 `A_T`, `A_O`, `vgr`, `vision_density` 等）到新代码中；若含义与论文一致，请改为论文中的符号或命名。
- **建议**：新模块内注释中引用论文公式编号或小节（如 “Eq. (3) in DDAPO”），便于审阅与复现。
- **建议**：先实现「单样本、单次 forward + 论文中一个核心指标」跑通，再扩展 batch 与多模态。

请按上述顺序执行，并确保 `codex_dev.log` 在**仓库根目录**（即 ms-swift 根目录）下生成且格式符合规范。完成后可将 `codex_dev.log` 作为与后续 Agent 的通讯媒介。

---

## 五、日志解析约定（供后续 Agent / 脚本使用）

- **块分隔**：以 `## [TIMESTAMP]` 或 `## [TIMESTAMP] SUMMARY` 开头的行作为一条记录的起始。
- **字段提取**：每块内按行匹配 `### File:`, `### Action:`, `### Description:`, `### Code:`，其后的内容直至下一个 `###` 或 `---` 为该字段取值。
- **代码块**：`### Code:` 后若紧跟 ` ```<language>`，则代码内容到下一个 ` ``` ` 为止；解析时可按需 strip 首尾空行。
- **用途**：后续 Agent 可读取 `codex_dev.log`，根据 `File` + `Action` + `Code` 复现或合并修改，根据 `SUMMARY` 继续未完成项。
