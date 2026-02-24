# Codex 任务：DDAPO 注意力奖励为 0 —— 调试输出 + 改为 Eager 实现

## 一、背景与现象

- 在 ms-swift 中已实现 DDAPO 风格注意力奖励（`DDAPOAttentionORM`，在 `examples/train/grpo/plugin/plugin.py`），并通过临时关闭 gradient checkpointing 尝试拿到 `outputs.attentions`。
- **当前现象**：训练能跑，但 `rewards/DDAPOAttentionORM/mean` 始终为 0，日志中有 `DDAPOAttentionORM: model outputs did not include attentions. Returning 0.0 reward.`，说明 **forward 返回的 `outputs.attentions` 仍为 None**。

## 二、根因假设（与 trl_DAPO / HuggingFace 行为一致）

- **Flash Attention / SDPA** 等优化实现为节省显存与计算，**通常不计算也不返回** 完整 attention 权重矩阵；即使传入 `output_attentions=True`，很多 backend 也会忽略，导致 `outputs.attentions` 为 None。
- **只有 `eager` 实现**会真正计算并返回每层的 attention，适合 DAPO/DDAPO 这类需要 attention 做 reward 的场景。trl_DAPO 等实现中，**取 attention 的那次 forward 会使用 eager 模式**（或等价地，使用支持 output_attentions 的 backend）。
- 因此需要：**在 DDAPO 做 reward 的那次 forward 时，把模型的 attention 实现临时改为 `eager`**，forward 完成后再恢复，这样既能拿到 attentions，又不影响训练时继续用 flash/sdpa。

## 三、请你完成的两部分

### 3.1 添加调试输出（便于根据日志定位问题）

在 `examples/train/grpo/plugin/plugin.py` 的 `DDAPOAttentionORM.__call__` 中，在**第一次**进入「对单个 sample 做 forward 并检查 attentions」的逻辑时（即 `idx == 0` 且成功得到 `batch` 且即将调用 `model(**model_kwargs)` 时），增加**一次性的调试输出**（可用 `logger.info` 或 `print`，建议用 `logger.info` 并加统一前缀如 `[DDAPO_DEBUG]` 便于 grep）：

1. **model_kwargs**：打印其 **keys**，以及是否包含 `output_attentions` 及其值（应为 True）。
2. **inner_model 的 attention 配置**：对 `inner_model = getattr(model, 'module', model)`，打印其 `config` 上与 attention 实现相关的属性，例如：
   - `getattr(inner_model.config, '_attn_implementation', None)`
   - `getattr(inner_model.config, 'attn_implementation', None)`
   - `getattr(inner_model.config, 'llm_attn_implementation', None)`（Qwen2.5-VL 等可能用此 key）
   - 若存在 `inner_model.config.text_config`，同样打印其 `_attn_implementation` / `attn_implementation` / `llm_attn_implementation`。
3. **Gradient checkpointing 状态**：打印 `getattr(inner_model, 'is_gradient_checkpointing', None)`。
4. **Forward 之后**：打印 `type(outputs)`；`getattr(outputs, 'attentions', None)` 是否为 None；若不为 None，打印其类型及 `len(attentions)`（或 shape 的前几维）。

以上调试仅在**第一个 sample（idx==0）且仅一次**打印，避免刷屏；可用类成员如 `self._debug_done` 在第一次打印后置 True，后续不再打印。

### 3.2 在 DDAPO forward 前临时切换为 eager，forward 后恢复

在 `DDAPOAttentionORM.__call__` 中，在做「policy model 的 forward 以取 attention」的那段逻辑里：

1. **取得需要改 config 的对象**：在已有 `inner_model = getattr(model, 'module', model)` 的基础上，确定要修改的 config。对 **Qwen2.5-VL** 等，attention 实现可能在 `config.llm_attn_implementation` 或 `config.text_config.llm_attn_implementation`。请参考 `swift/model/utils.py` 中 `AttnImpl.attn_impl_keys`（如 `_attn_implementation`, `attn_implementation`, `llm_attn_implementation`），对 **inner_model.config** 以及若存在的 **inner_model.config.text_config** 都做统一处理。
2. **保存并设为 eager**：对上述每个 config 对象，遍历 `attn_impl_keys`（只处理该 config 上存在的 key），保存当前值，然后将该 key 设为 `'eager'`。
3. **在 with 块内执行 forward**：在现有的 `with torch.no_grad(), template.forward_context(...), disable_gradient_checkpointing(...)` 中执行 `outputs = model(**model_kwargs)`。**在进入该 with 之前**先完成「保存并设为 eager」；**在该 with 块结束后**立即恢复刚才保存的 attention 实现值。
4. **恢复**：将之前保存的每个 config 的每个 key 的值写回，确保训练时后续 forward 仍使用原来的 flash/sdpa 等实现。

注意：若模型被 DeepSpeed 等包装，`inner_model` 应为实际带 `.config` 的 HuggingFace 模型（例如 PeftModel 的 base_model.model 或 accelerator.unwrap_model 后的模型），确保修改的是**真正参与 forward 的 config**。

## 四、实现时的代码位置与顺序建议

- 文件：`examples/train/grpo/plugin/plugin.py`，类：`DDAPOAttentionORM`。
- 在 `for idx, completion in enumerate(completions):` 循环内，在得到 `batch` 并 `model_kwargs = self._build_model_kwargs(model, batch)` 之后、在 `with torch.no_grad(), ...` 之前：
  1. 若 `idx == 0` 且未做过调试输出，则执行 3.1 的调试打印（model_kwargs、config 的 attn 相关属性、gradient checkpointing 状态）。
  2. 对 inner_model（及其 config / text_config）执行「保存 attn 实现 → 设为 eager」。
  3. 进入现有 `with torch.no_grad(), template.forward_context(model, batch), disable_gradient_checkpointing(inner_model, gcp_kwargs):`，在其中 `outputs = model(**model_kwargs)`。
  4. 若 `idx == 0` 且未做过 forward 后调试，则打印 outputs 类型与 attentions 是否存在/长度，并标记调试已完成。
  5. 在 with 块**外**、同一轮循环内，立即「恢复」步骤 2 中保存的 attn 实现值。

这样既便于通过日志确认是否因非 eager 导致 attentions 为 None，又通过临时 eager 修复该问题。

## 五、日志与约束

- 调试输出请加统一前缀 `[DDAPO_DEBUG]`，便于搜索。
- 修改请写入仓库根目录的 `codex_dev.log`，格式可参考同目录下 `CODEX_PROMPT_DDAPO_GRPO.md` 中的「通讯日志」约定（File / Action / Description / Code）。
- 不要改动 DDAPO 的公式与变量名（仍以论文为准），仅增加调试与 attention 实现切换逻辑。

完成上述后，用户重新运行 `run_grpo_with_ddapo.sh`，应能看到 `[DDAPO_DEBUG]` 日志，并可根据输出判断问题；若根因确为非 eager，则 `rewards/DDAPOAttentionORM/mean` 应不再恒为 0。
