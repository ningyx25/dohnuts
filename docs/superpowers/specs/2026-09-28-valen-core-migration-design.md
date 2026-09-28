# Valen 核心能力迁移到 Dohnuts 的设计(双投影决策头 + 四档 stage + SFT/GRPO 训练目标)

日期:2026-09-28
状态:设计已确认,待实现
范围:在 Dohnuts 现有的批量打分执行栈之上,叠加 Valen 的**双投影决策头**、**四档 stage 参数开放**
(含 LoRA 枚举与四组独立学习率)、以及 **SFT 与 GRPO 两个新训练目标**(与现有 Laya 对齐的 RLCD
三者由 `method` 字段并存选择)。保持单卡、保持 prompt 模板字符串与已转换数据
(`data/processed/gui-v1`、`data/processed/ac-v1`)完全不变,`src/dohnuts/rlcd.py` 及其 golden
fixtures 一字不动。

## 1. 背景与目标

Dohnuts 与 Valen(`/home/ningyongxin/workplace/proj/Valen`)是**同源设计**:都是 Jev 风格的非生成式
决策模型 —— 输入 state + 若干问题,输出候选上的概率分布;问题类型同为 `choice`/`noul`/`score`;
backbone 同为 Qwen3.5 VLM 且丢弃词表输出层;绝不调用 `generate()`。但实现分歧很大:

| 维度 | Dohnuts 现状 | Valen |
| --- | --- | --- |
| 决策头 | 单个 `nn.Linear(H,1,bias=False)`,1024 参数,打分每个候选的 marker token 隐状态 | 双投影点积 `dot(W_c·h_cand, W_d·h_dec)/√256`,2·H·256 参数,候选数无关 |
| decision 位置 | 无(只有候选 marker 位置) | 专门的 `Decision:` token 位置 |
| LoRA | r=8/α=16,硬编码 12 个短模块名 | r=32/α=64,枚举 `language_model` 下全部 `nn.Linear` 路径 |
| 视觉 | 永久冻结(`train()` 每次重冻结) | 四档 stage 可解冻 merger 与顶 4 层 |
| 优化器 | 2 组(backbone/head) | 4 组(head/lora/merger/vision),各组独立 LR |
| RL 目标 | Gaussian 扰动 REINFORCE(Laya 对齐,golden 测试钉死数值) | 类别型 GRPO:多项式采样 + clip ratio + 固定参考 KL + Brier + num_iterations 复用 |
| SFT | 无独立阶段(soft CE 只是 RLCD 的辅助项) | 独立 method:分布 CE + RPS + Brier |
| 执行效率 | 批量多题一次 forward、共享前缀 KV(`PrefixCache`)、128MiB 图像特征 LRU | 每分支独立 full forward、`use_cache=False`、无共享 |

**执行效率一栏 Dohnuts 明显更优,本次迁移完整保留**;要迁移的是模型表达力(双投影头)、参数开放
粒度(stage)与训练目标(SFT/GRPO)。

目标:让 Dohnuts 在不破坏现有数据、现有 Laya 数值对齐、现有推理效率的前提下,获得 Valen 的这三项
能力,并且新旧路径可由 recipe 字段切换、可对照实验。

## 2. 用户已确认的决策

1. **迁移范围**:核心迁移 —— 双投影头 + 四档 stage + SFT/GRPO 目标。**不**引入 Valen 的
   `ArchitectureBackend`/`factory`/`interfaces` 抽象层、多卡数据并行(`distributed.py`)、
   token 预算整 state 打包、逐 rank checkpoint 状态、`config_version:2` 分区配置。
2. **候选打分方式**:保留 Dohnuts 的 marker 批量打分与共享前缀,只把打分函数换成 Valen 的双投影点积。
3. **decision 位置**:读**题干末尾** token(`Options:\n` 处),**不改 prompt 模板字符串** ⇒
   gui-v1/ac-v1 已转换数据与 token 预算继续有效,不重跑 74GB 数据管线。
4. **RL 目标**:三者并存,`method ∈ {rlcd(默认,Laya), sft, grpo}`;`rlcd.py` 与
   `tests/fixtures/laya-rlcd.json` 零改动。
5. **stage 迁移量**:四档全迁(warmup/text/joint/vision_top),LoRA rank 可配 + 枚举全部 Linear,
   head/lora/merger/vision 四组独立学习率。
6. **多卡**:暂不做,保持单卡(当前机器 4× A40 46GB,但现有代码路径全部单卡硬编码)。
7. **验证方式**:CPU 单元测试(小模型替身 + 数值闭式)+ 一个 GPU 冒烟脚本 + gui-v1 短预算训练冒烟;
   不跑完整对照实验。

## 3. 数据事实与代码事实(探索已验证)

- **backbone**:`Qwen/Qwen3.5-0.8B`(本地快照,`revision.txt` =
  `2fc06364715b967f1860aea9cf38778875588b17`)。text hidden 1024、24 层、`layer_types` 为
  3× `linear_attention` + 1× `full_attention` 交替(gated delta rule);vision depth 12、hidden 768、
  `spatial_merge_size` 2、`out_hidden_size` 1024。`AutoModel` 解析为 `Qwen3_5Model`。
  `Qwen/Qwen3.5-2B` 本地存在但**缺 `revision.txt`**,`model_revision()` 会报错,本次不用。
- **marker**:`<|fim_middle|>` → id 248062,单 token 编码成立;词表 248077。
- **tokenizer**:`padding_side="right"`、fast tokenizer(支持 `return_offsets_mapping`)。
- **decision position 实测**:stem 为 48 字符时,offset 定位得到 idx=12,该 token 解码为 `'\n'`
  (即 `Options:\n` 的换行),下一个 token 才是第一个候选的 `'-'` ⇒ **定位正确且不泄漏任何候选描述**。
- **视觉塔可微性**:`Qwen3_5VisionBlock` 用 `nn.LayerNorm` + `Qwen3_5VisionMLP`
  (`linear_fc1`/`act_fn`/`linear_fc2`),`Qwen3_5VisionPatchMerger` 用 `norm`/`linear_fc1`/`act_fn`/
  `linear_fc2`。`enable_fusion` 只按 `type(module) in {Qwen3_5RMSNorm, Qwen3_5RMSNormGated,
  Qwen3_5MLP}` 改写 forward ⇒ **视觉塔完全不被 fusion 触碰**,解冻后梯度可正常流动。
- **fusion detach 陷阱**:`fused_rms_norm` 使用 `module._dohnuts_norm_weight`,该 buffer 是
  `weight.detach().float() + 1` ⇒ 对语言塔 RMSNorm 权重**不可微**。当前所有 stage 都不解冻语言塔
  norm,因此安全;若未来要解冻语言塔 norm,必须先修此处(见 §14 已知限制)。
- **可训练参数量**:现状 LoRA r=8 共 5,412,352(含 head 1,024)。双投影头 P=256 时 head 变
  2·1024·256 = 524,288。
- **封闭 recipe**:`train.py:main()` 要求 `config == training_recipe(...)` **全等**,否则
  `ValueError`;`RLCDConfig` 用 `field(init=False)` 封住 5 个字段 ⇒ 现有唯一用户可调旋钮是
  `--steps`/`--sigma`/`--ce-weight`。
- **checkpoint 校验**:`model.load_adapter` 拒绝 `format_version != 1`、adapter 名不同、
  `base_revision` 不同、`(image_pixels, max_length, lora_rank) != (IMAGE_PIXELS, MAX_LENGTH, 8)`、
  sha256 不同,**或 safetensors 键集/形状与 `{n: p for n,p in named_parameters() if p.requires_grad}`
  不等**;`train.save_checkpoint`/`load_checkpoint` 同样断言 trainable 键集全等。
- **单次 forward 断言**:`scripts/run_laya_benchmark.py` 与 `scripts/evaluate_upstream.py` 对
  `predictor.model` 挂 forward hook 并要求每请求**恰好一次** forward 产出 `[B,K]` ⇒ 不得引入 Valen
  的逐分支循环。
- **现有调用方兼容性**:`tests/test_gui_data.py:987/1090` 与 `tests/test_android_control_data.py:2548`
  用 `collated, *_ = DecisionCollator(...)(rows)` 解包 ⇒ collator 返回元组扩展**天然兼容**;
  `scripts/audit_checkpoint.py` 只调 `train.evaluate()`,不触碰 batch 元组 ⇒ 无需改动;
  `scripts/prepare_data.py:936/1034` 与两个 `prepare_*_data.py` 的 `token_length` 都是 `[0]` 索引
  取 prompt ⇒ 无需改动;只有 `tests/test_gui_data.py:1097` 与 `tests/test_android_control_data.py:2528`
  用 `prompt, _ =` 二元解包,需改为三元。
- **质量门禁**:ruff line-length 100 / `select=["E4","E7","E9","F","I","UP","B","PT"]` / `ignore=["B905"]`、
  `ty check src`(scripts/ 不在门禁内)、prek、`sphinx-build -W`;测试 216 项全 CPU、无 conftest.py、
  缺权重时 `pytest.skip`。
- **仓库约定**:提交必须带显式路径(`git commit -m "..." -- <paths>`),因为其他分支上可能有用户自己的
  暂存改动;`data/processed/*`、`runs/`、`example-data/` 均不入 Git。

## 4. 双投影决策头(`src/dohnuts/model.py`)

新增 `DecisionHead(nn.Module)`,替换 `DecisionModel.head` 的 `nn.Linear(H,1)`:

```python
class DecisionHead(nn.Module):
    def __init__(self, hidden_size, projection_dim=256):
        super().__init__()
        self.decision  = nn.Linear(hidden_size, projection_dim, bias=False)   # fp32
        self.candidate = nn.Linear(hidden_size, projection_dim, bias=False)   # fp32
        self.scale = math.sqrt(projection_dim)      # Python float,不入 state_dict
```

`score_hidden` 改为批量双投影(`hidden` 是 `[B,L,H]`,`positions` 是 `[B,K]`,
`decision_positions` 是 `[B]`):

```python
def score_hidden(self, hidden, positions, decision_positions):
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    d = self.decision(hidden[rows, decision_positions].float())               # [B,P]
    c = self.candidate(hidden[rows[:, None], positions].float())              # [B,K,P]
    return (c * d[:, None, :]).sum(-1) / self.scale                           # [B,K]
```

`DecisionModel.forward(inputs, positions, decision_positions)` 增加第三个参数,两个位置张量同样做
`(x - offset).clamp_min(0)` 以适配共享前缀偏移。

**打分公式**:

```
logit[b,k] = ( W_candidate · h[b, positions[b,k]] ) · ( W_decision · h[b, decision_positions[b]] ) / √P
P = projection_dim = 256;读出与损失一律 fp32(backbone 仍 bf16)
```

- head 参数 `device="cuda", dtype=torch.float32`(沿用现状);初始化用 PyTorch `nn.Linear` 默认
  (不沿用现状的 `normal_(std=0.01)`,否则 256 维点积量级过小)。
- `self.head` 仍是名为 `head` 的子模块 ⇒ `train.py` 的 `n.startswith("head.")` 参数组分类与
  `experiment.py` 的 `head_parameters` 统计**自动适配**,无需改判定逻辑。
- 与 Valen 的差异:Valen 的 `DecisionHead.forward` 假设 batch=1(`hidden[0, ...]`,因为它逐分支
  forward);Dohnuts 是批量,因此保留 `rows` 索引。参数量与公式一致。

## 5. decision position 定位(`src/dohnuts/predictor.py` + `training_data.py`)

prompt 模板**字符串完全不变**,只让 `render_question` 额外返回题干字符长度:

```python
def render_question(state_text, question, *, has_image=False, adapter=None):
    ...
    stem = f"State: {state_text}\n{question['type']} question: {instructions}\nOptions:\n"
    if has_image:
        stem = adapter.image_prefix + stem
    content = stem + "".join(f"- {option}{marker}" for option in options)
    return content, labels, len(stem)                    # 新增第三个返回值
```

decision position 用 fast tokenizer 的 offset mapping 在 stem 字符范围内定位(collator 与
`predictor.prepare` 各逐行算一次):

```python
enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
# 取「起始字符 < stem_chars」的最后一个非空 token;右 padding 下批内索引与单行一致
decision_position = max(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < stem_chars)
```

- 实现处断言 `tokenizer.padding_side == "right"`(现状成立);若为 left 则需按 padding 量修正索引。
- collator 返回值从 5 元扩为 6 元:
  `(inputs, positions, decision_positions, mask, target, ordinal)`;`predictor.prepare` 同步返回。
- `token_length` 拿到的 prompt 字符串不变 ⇒ 已转换数据的 token 数不变,
  `test_token_length_matches_the_training_collator` 与 ac-v1 的 535 条 `parse:token_budget` 边界行
  全部继续成立,**不需要重跑任何数据管线**。
- **与 Valen 的语义差异**:Valen 的 decision 向量在 prompt 末尾(`Decision:` token),因果注意力下
  已看完全部候选;本设计读题干末尾,decision 向量**看不到任何候选描述**,是一个纯 query。这是为了
  不改模板/不重跑数据而做的取舍;好处是同一 state 的所有问题共享同一个 decision 读出位置语义,
  且与共享前缀机制天然兼容(cut 必须早于最早 marker,而 decision position 更早)。
- **不泄漏候选**是硬约束,必须有专门测试(§12)。

## 6. stage 参数开放与 LoRA 枚举(`src/dohnuts/adapters.py`)

`adapt_language` 扩为按 stage 应用,LoRA target 从硬编码短名改为枚举(对齐 Valen;自动覆盖
gated-delta-net 的 `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a`/`out_proj`,不依赖名字表):

```python
def apply_stage(self, backbone, stage, *, lora_rank, lora_alpha, training):
    backbone.requires_grad_(False)
    if stage != "warmup":
        targets = [n for n, m in backbone.language_model.named_modules() if isinstance(m, nn.Linear)]
        backbone.language_model = get_peft_model(backbone.language_model, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
            target_modules=targets, bias="none"))
        if training:
            backbone.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    if stage in {"joint", "vision_top"}:
        backbone.visual.merger.requires_grad_(True)
    if stage == "vision_top":
        for block in backbone.visual.blocks[-4:]:
            block.requires_grad_(True)
    self._vision_trainable = stage in {"joint", "vision_top"}
```

| Stage | 可训练部分 | 图像特征 LRU |
| --- | --- | --- |
| `warmup` | 只有双投影决策头(**不创建 LoRA**) | 可用 |
| `text` | head + `language_model` 全部 Linear 的 LoRA | 可用 |
| `joint` | `text` + `visual.merger` | **必须旁路**(§7) |
| `vision_top` | `joint` + `visual.blocks[-4:]` | **必须旁路**(§7) |

- `DecisionModel.enable_lora(...)` 改名/扩参为 `enable_stage(stage, *, lora_rank, lora_alpha,
  checkpointing=True)`;`predictor.from_checkpoint` 与 `train.final_evaluation` 从 checkpoint
  metadata 读 `stage`/`lora_rank`/`lora_alpha`/`projection_dim` 决定加载行为。
- `DecisionModel.train(mode)` **无需改动**:复核 `Qwen35Adapter.freeze_vision` 只调
  `backbone.visual.eval()`,并不动 `requires_grad` ⇒ 它不会清掉 joint/vision_top 的视觉梯度。
  视觉塔只有 LayerNorm + GELU(无 dropout/BN,已核 `Qwen3_5VisionBlock` 源码),`eval()` 与
  `train()` 行为等价,保留现状即可。冻结本身发生在 `load()` 的 `requires_grad_(False)` 与
  `apply_stage` 内,与此无关。实现时加一条测试钉住「`model.train()` 之后 merger 参数仍
  `requires_grad`」,防止日后有人把 `freeze_vision` 改成真的冻结。
- **vision_top 的显存**:现状只对 `language_model` 开 gradient checkpointing;解冻视觉顶 4 层后
  这部分激活会常驻。若冒烟时显存吃紧,再考虑对 `visual` 也开 checkpointing(本次不预设,记为
  实现期观察项)。
- **与 Valen 的 stage 语义差异**:Valen 的 `text` 档强制纯文本数据(`QwenBackend.validate_state`
  会拒绝含媒体的 state);Dohnuts 的 `text` 档**不**做此限制 —— 视觉特征仍参与前向,只是视觉权重
  冻结、图像 LRU 可用。这是刻意保留:Dohnuts 的 GUI 数据全部带截图,若照搬限制则 text 档无数据可用。
- 四档 stage 的解冻集合必须有精确测试(§12)。

## 7. 视觉解冻时旁路图像特征 LRU(`src/dohnuts/adapters.py`)

`_dohnuts_image_cache` 是 128MiB LRU,在 `torch.no_grad()` 下缓存**冻结**视觉特征。joint/vision_top
档视觉权重每步都在变,缓存会返回**过期特征**,造成静默训错(损失看起来正常,但梯度对应的是旧权重)。
`forward` 按 `_vision_trainable` 分流:

```python
if "pixel_values" in inputs:
    if self._vision_trainable:
        features = backbone.get_image_features(          # 带梯度,不缓存
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True).pooler_output
    else:
        features = self.image_features(backbone, inputs)  # 现有 128MiB LRU,no_grad
    ...
```

- `shared_image_inputs` 路径下的 `features.repeat(B,1)` 在带梯度路径同样可微,无需分支。
- `enable_linear_patch_embedding` 用 `F.linear(patches, module.proj.weight.flatten(1), bias)`
  ⇒ patch embed 若被解冻同样可微(vision_top 不解冻 patch_embed,仅记录事实)。
- 代价:joint/vision_top 失去图像特征复用,同一 batch 内重复截图会重复编码。这是正确性换效率,
  必须在文档与 recipe 注释中写明。

## 8. 训练目标三选一(新增 `src/dohnuts/objectives.py`,`rlcd.py` 不动)

统一接口,与 `train.py` 现有 `stats` 累积方式(dict of detached scalar tensors)兼容:

```python
loss(logits[B,K], targets[B,K], mask[B,K], ordinal[B], *, config, reference_logits=None)
    -> (loss, metrics: dict[str, Tensor(detached)])
```

### 8.1 SFTObjective(批量、masked,对齐 Valen `training/sft.py::question_loss`)

```
logp  = logits.float().masked_fill(~mask, -inf).log_softmax(-1)
p     = logp.exp()
ce    = -(targets * logp).masked_fill(~mask, 0).sum(-1).mean()
rps   = score 行: ((p.cumsum(-1) - targets.cumsum(-1))^2 取前 K-1 项).mean() ,否则 0
brier = ((p - targets)^2 * mask).sum(-1).mean()
L     = ce + rps_weight * rps + brier_weight * brier      # 默认 rps_weight=0, brier_weight=0
```

- `ordinal[B]` 标记 score 行,RPS 只对这些行计算(与现有 `rlcd_loss` 的 ordinal 语义一致)。
- 默认权重 0 ⇒ 纯分布交叉熵,与 Valen 默认一致。
- metrics:`ce`、`rps`、(`brier` 当权重非零)。

### 8.2 GRPOObjective(对齐 Valen `training/rlcd.py`)

```python
GRPOConfig: group_size=16, num_iterations=1, correctness_weight=1.0, confidence_weight=1.0,
            brier_weight=1.0, beta=0.02, clip_epsilon=0.2, advantage_epsilon=1e-6
# 严格校验:未知键 → ValueError;类型/范围校验对齐 Valen validate_options
# (group_size int≥2、num_iterations int≥1、权重有限非负非 bool、0<clip_epsilon<1、
#  advantage_epsilon>0、correctness 与 confidence 权重不可同时为 0)
```

**prepare(`no_grad`,每 microbatch 一次)**:

```
p        = softmax(logits.float().masked_fill(~mask, -inf))
actions  = multinomial(p, group_size, replacement=True)                 # [G,B] 候选索引
corr     = targets.gather(actions);  conf = p.gather(actions)
err      = corr*(1-conf)^2 + (1-corr)*conf^2
reward   = w_c*corr - w_e*err                                          # [G,B],detach
advantages = (reward - mean_G) / (std_G + eps) ;  整组 reward 全等 → 置 0(精确零)
old_logp = logp.gather(actions)
ref_logp = reference_logits.float().log_softmax(-1)                     # 完整分布,[B,K]
```

**update(带梯度,重复 `num_iterations` 次,梯度累积到同一组)**:

```
ratio  = exp(new_logp.gather(actions) - old_logp)
policy = -min(ratio*A, clip(ratio, 1±δ)*A).mean()
kl     = (new_p * (new_logp - ref_logp)).sum(-1).mean()                 # exact categorical KL
brier  = ((new_p - targets)^2 * mask).sum(-1).mean()
L      = policy + β*kl + w_brier*brier
```

metrics:`policy_loss`、`kl`、`brier`、`reward_mean`、`reward_std`、`clip_fraction`、
`zero_advantage_group`、`sample_correctness`、`sample_confidence`、`confidence_error`。

- **奖励语义**:`corr` 取软标签在被采样候选上的概率质量(不是 argmax 命中),因此 ac-v1
  `screenshot_choice` 的 30.37% 多命中软标签行天然被正确处理(与 Valen 软标签语义一致)。
  硬标签时 `err` 退化为 `(p_a - y_a)^2`。
- `num_iterations=1` 时 ratio ≡ 1、clip 不生效(退化为 PG + KL + Brier);要让 clip 真正生效需 ≥2,
  实现为同一 microbatch 内先 `no_grad` 采样固定 rollout,再重复带梯度 forward。默认取 1(省一次
  forward),recipe 可配。此取舍需在 `docs/rlcd.md` 写明。
- **Brier 项的作用**(Valen 测试证明):整组采样到同一候选时 advantages 全零、纯 GRPO 梯度精确为 0,
  Brier 项仍提供学习信号。必须保留对应的单元测试。
- **reference 模型**:`beta > 0` 时在训练起点 `deepcopy(model)` → `requires_grad_(False).eval()`,
  并**清空其 `_dohnuts_image_cache`**(否则参考模型会读到策略模型后续写入的缓存条目);`beta == 0`
  不建参考模型、不算 KL。resume 时从 checkpoint 的 `training_state.reference_weights` 恢复**原始**
  快照(键集必须与当前 trainable 集合匹配),绝不从当前权重重新导出 —— 这是 Valen 的语义,必须保留。

### 8.3 与现有 Laya RLCD 的关系

`rlcd.py`(`RLCDConfig`/`distribution_rewards`/`rlcd_loss`)与
`tests/fixtures/laya-rlcd.json`、`tests/test_rlcd.py` **零改动**。三个目标由 `config["method"]`
分派,互不影响。`docs/rlcd.md` 保留 Laya 对齐段落,新增 SFT/GRPO 段落与选择说明。

## 9. recipe 字段扩展(`src/dohnuts/recipe.py`)

`training_recipe(...)` 新增字段(保持封闭 schema 与 `main()` 全等校验):

```python
"method": "rlcd",                 # rlcd | sft | grpo
"stage": "text",                  # warmup | text | joint | vision_top(text = 现状默认)
"projection_dim": 256,
"lora_rank": 8,                   # 现状字段,改为可配(默认仍 8)
"lora_alpha": 16,                 # 新增(现状硬编码 16)
"backbone_lr": 1e-4,              # 现有,即 lora 组 LR
"head_lr": 5e-4,                  # 现有
"merger_lr": 1e-5,                # 新增,仅 joint/vision_top 生效
"vision_lr": 2e-6,                # 新增,仅 vision_top 生效
"sft":  {"rps_weight": 0.0, "brier_weight": 0.0},                         # 仅 method==sft 写入
"grpo": {group_size, num_iterations, correctness_weight, confidence_weight,
         brier_weight, beta, clip_epsilon, advantage_epsilon},             # 仅 method==grpo 写入
```

`rlcd` 字段(`sigma`/`ce_weight`)仅在 `method == "rlcd"` 且非默认时写入,沿用现状写法。

## 10. 训练循环分派(`src/dohnuts/train.py`)

- `policy` 构造按 `config["method"]` 分派:`rlcd` → `RLCDConfig(**config.get("rlcd", {}))` +
  `rlcd_loss`;`sft` → `SFTObjective`;`grpo` → `GRPOObjective`。
- 参数组从 2 组扩为最多 4 组,按参数名分类,**无法归类直接报错**(对齐 Valen `qwen_optimizer_groups`):

```python
head  ← n.startswith("head.")            @ head_lr
lora  ← "lora_" in n                      @ backbone_lr
merger← ".visual.merger." in n            @ merger_lr
vision← ".visual.blocks." in n            @ vision_lr
```

  只发出非空组;LR schedule 的 `factor` 对所有组统一乘各自 `initial_lr`(现状机制不变)。
- `model.enable_lora()` → `model.enable_stage(config["stage"], lora_rank=…, lora_alpha=…)`。
- microbatch 内:`grpo` 走 prepare(`no_grad`) + `num_iterations` 次带梯度 forward/backward;
  `sft`/`rlcd` 单次 forward/backward。batch 元组扩为 6 元,`evaluate()`(L109 解包)与训练循环
  (L292 解包)同步更新,`decision_positions` 传入 `model(...)`;`to_gpu`/`prefetch_batches` 按
  元组长度自适应,无需改。
- `frozen` config 字典纳入全部新字段 ⇒ resume 全等校验自动覆盖。**迁移前创建的旧 run 因 config
  不等不能 resume**,需用 `--initialize-from` 重起(在 `docs/run-experiment.md` 注明)。
- **checkpoint `format_version` 1 → 2**:`save_checkpoint` 增 `training_state`(GRPO reference 权重);
  `export_checkpoint`/`load_adapter` 的 metadata 增 `stage`、`projection_dim`、`method`、
  `lora_alpha`,`lora_rank` 校验改为读 metadata 实际值(不再硬编码 8);键集相等校验保留(自然适配
  双投影头与 stage 相关的 trainable 集合)。**旧 `format_version: 1` checkpoint 一律拒绝加载**
  (决策头架构已变,参数键集不同),错误信息需说明原因。

## 11. 推理路径(`src/dohnuts/predictor.py`)

`prepare` 返回 `decision_positions`,`predict` 调 `model(inputs, positions, decision_positions)`。
输出仍是**一次 forward 出 `[B,K]`** ⇒ `run_laya_benchmark.py`/`evaluate_upstream.py` 的单次 forward
hook 断言不受影响。温度校准(`fit_temperatures`)、答案组装、confidence 公式全部不变。

## 12. 测试(CPU,小模型替身 + 数值闭式)

| 测试文件 | 覆盖 |
| --- | --- |
| `tests/test_decision_head.py`(新) | 双投影头输出形状 `[B,K]`、fp32、与手算点积一致、对 decision/candidate 位置都有梯度、`scale` 不入 `state_dict`、参数量 = 2·H·P |
| `tests/test_objectives.py`(新) | SFT 损失对 `F.cross_entropy` 闭式(含 RPS/Brier 项与权重);GRPO 的 reward/advantage/clip/KL/Brier 对手算用例;常数奖励组 advantage 精确为 0 而 Brier 仍有梯度;未知 GRPO 键报错;软标签 correctness 取概率质量;`beta=0` 时不建参考模型 |
| `tests/test_stages.py`(新) | 四档 stage 的 trainable 参数集合精确匹配;LoRA 枚举覆盖 `in_proj_*`/`out_proj`;`warmup` 不创建 LoRA;joint/vision_top 下 `_vision_trainable` 为真且图像特征带梯度、LRU 被旁路;`train()` 不再重冻结已解冻视觉 |
| `tests/test_decision_position.py`(新) | offset 定位落在 stem 内、**不泄漏任何候选描述**、右 padding 下批内索引与单行一致、含图与纯文本两种 stem |
| `tests/test_metrics.py`(扩) | checkpoint round-trip:`format_version: 2`、`stage`/`projection_dim` 校验、旧 v1 拒绝加载、GRPO reference resume 等价 |
| `tests/test_rlcd.py` + fixtures | **零改动**(Laya 路径隔离的回归保证) |
| `tests/test_gui_data.py:1097`、`tests/test_android_control_data.py:2528` | `prompt, _ =` → `prompt, _, _ =`(仅这两处) |

测试哲学沿用 `docs/design.md`:断言可观察行为或已知失败用例,不断言内部分配、日志键、文件数与
错误措辞。

## 13. GPU 冒烟与文档

**`scripts/smoke_valen_migration.py`(新)**,真实 `Qwen/Qwen3.5-0.8B` 单卡:四档 stage 解冻集合 +
各组非零梯度;双投影头前向/反向数值;joint 档图像特征带梯度且 LRU 确被旁路;SFT 2 步 loss 下降;
GRPO 2 步(`beta>0`)reference 固定、指标有限;checkpoint save→load→resume 等价;merge 后
`Predictor` 输出合法分布。

**文档更新**:`docs/design.md`(双投影头公式、stage 表、四组 LR)、`docs/rlcd.md`(新增 SFT/GRPO
目标与 method 选择,保留 Laya 段落与 `num_iterations` 取舍说明)、`docs/run-experiment.md`
(新 recipe 字段、旧 run 不能 resume)、`docs/inference.md`(decision 位置语义)。
`MODEL_CARD.md` 暂不改(未重训发布模型)。

## 14. 硬约束与已知限制

1. **封闭 recipe**:`main()` 全等校验 ⇒ 所有新字段必须同时进 `training_recipe()` 与 `frozen` 字典;
   漏一处就无法启动训练。
2. **golden 隔离**:`rlcd.py` 一字不改,SFT/GRPO 全部放新模块 `objectives.py`。
3. **checkpoint 不向后兼容(已定)**:`format_version: 1` 一律拒绝加载,错误信息说明「决策头已从单
   Linear 升级为双投影,参数键集不同;请用 `--initialize-from` 从 v1 导出重训,或重新训练导出」。
   **不保留 v1 双路径** —— 理由:`load_adapter` 的既有设计哲学就是严格校验并拒绝任何偏离 pinned
   recipe 的权重(base revision、sha256、键集全等),引入两套头结构会让 `model.py` 长期背负分支代码,
   违反单一职责。影响面:本地 `runs/notebook/{seed-42,gui-v1,ac-v1}`、`runs/smoke` 下的 v1 产物
   (均为 gitignored 的探索性冒烟结果)不再可用 `Predictor` 加载;已发布的
   `PsiACE/Dohnuts-0.1.0-0.8B` 不在本地缓存,本次不重训不发布,其加载路径随 v1 一并废弃 —— 若日后
   需要复现发布模型的评测,应在迁移前的 commit 上做。
4. **decision 位置看不到候选**:相比 Valen 的 prompt 末尾 `Decision:` 是表达力取舍(§5)。若实验
   显示这限制了质量,后续可加模板变体 + 重跑数据管线(独立需求)。
5. **图像 LRU 与视觉解冻互斥**:joint/vision_top 必须旁路缓存并带梯度算特征,否则静默训错;
   代价是失去同 batch 重复截图的特征复用。
6. **fusion detach buffer**:语言塔 RMSNorm 用 detach 权重 ⇒ 不可微。当前所有 stage 都不解冻语言塔
   norm,安全;**解冻语言塔 norm 之前必须先修 `fused_rms_norm`**。
7. **单次 forward 断言**:不得引入 Valen 的逐分支循环,否则 `run_laya_benchmark.py`/
   `evaluate_upstream.py` 会失败。
8. **GRPO reference 显存**:`deepcopy` 整模型约 +1.7GB(0.8B bf16);joint/vision_top + GRPO 会再叠加
   视觉梯度显存。建议 GRPO 配 warmup/text。A40 46GB 充足。
9. **Score 不做等级隔离**:保持 Dohnuts 的单分支 + `ordinal` RPS 惩罚,不迁 Valen 的 K 个独立分支
   (那会让 Score 计算量 ×K 并破坏单次 forward 断言)。
10. **单卡**:不迁 `distributed.py`;`set_per_process_memory_fraction(0.8)`、`device="cuda"`、
    `Sampler(Path("/sys/class/drm/card1/device"))`(AMD sysfs,本机只记录 `rss_bytes`)等硬编码保持现状。

## 15. 实现落点

| 文件 | 改动 |
| --- | --- |
| `src/dohnuts/model.py` | `DecisionHead` 新增;`score_hidden`/`forward` 加 `decision_positions`;`enable_lora` → `enable_stage`;`train()` 条件冻结视觉;`load_adapter` 校验 v2 |
| `src/dohnuts/predictor.py` | `render_question` 返回 stem 长度;`prepare` 算并返回 `decision_positions`;`predict` 传入;`from_checkpoint` 读 stage/projection_dim |
| `src/dohnuts/training_data.py` | `DecisionCollator` 算并返回 `decision_positions`(6 元组) |
| `src/dohnuts/adapters.py` | `apply_stage`(LoRA 枚举 + merger/vision 解冻);`forward` 按 `_vision_trainable` 旁路 LRU |
| `src/dohnuts/objectives.py`(新) | `SFTObjective`、`GRPOConfig`/`GRPOObjective`(prepare/update/state_dict) |
| `src/dohnuts/recipe.py` | 新字段(method/stage/projection_dim/lora_alpha/merger_lr/vision_lr/sft/grpo) |
| `src/dohnuts/train.py` | method 分派、4 组参数、`enable_stage`、6 元组解包、checkpoint v2 + `training_state` |
| `src/dohnuts/experiment.py` | 无逻辑改动(`head_parameters` 自动适配);确认 environment 记录含新字段 |
| `scripts/prepare_data.py`、`scripts/prepare_*_data.py` | **无需改动**(`[0]` 索引取 prompt) |
| `scripts/audit_checkpoint.py` | **无需改动**(只调 `evaluate()`) |
| `scripts/smoke_valen_migration.py`(新) | GPU 冒烟 |
| `tests/*` | §12 表格 |
| `docs/*` | §13 |
| `src/dohnuts/rlcd.py`、`tests/test_rlcd.py`、`tests/fixtures/laya-rlcd.json` | **零改动** |

## 16. 验证

```bash
git checkout -b valen-core-migration nvidia      # 已完成
# 1. 静态门禁
pdm run check                                    # ruff + ty check src
# 2. CPU 测试(216 现有 + 新增,全绿;golden 不变)
make test
# 3. GPU 冒烟(真实 0.8B,单卡)
pdm run python scripts/smoke_valen_migration.py
# 4. 短预算训练冒烟(data/processed/gui-v1,确认三条 method 路径都跑通并产出可加载 checkpoint)
#    method=sft   stage=warmup  steps=2
#    method=grpo  stage=text    steps=2   (--initialize-from 上一步导出)
#    method=rlcd  stage=text    steps=2   (回归:Laya 路径数值不变)
```

成功标准:`check`/`test` 全绿;冒烟脚本确认四档 stage 解冻集合与梯度正确、双投影头数值与手算一致、
GRPO reference 固定且指标有限、checkpoint resume 等价;三条 method 各跑 2 步 loss 有限,且 checkpoint
可被 `Predictor.from_checkpoint` 加载并输出合法概率分布。

## 17. 明确不做

- 不引入 Valen 的 `ArchitectureBackend`/`factory`/`interfaces` 抽象层。
- 不引入多卡数据并行、token 预算整 state 打包、逐 rank checkpoint 状态、`epoch_shard`。
- 不改 prompt 模板字符串、不加 `Decision:` token、不重跑 gui-v1/ac-v1 数据管线。
- 不把 Score 改成 Valen 的 K 个独立分支。
- 不引入 `config_version:2` 分区配置(保持扁平 recipe)。
- 不动 `rlcd.py` 及其 golden fixtures。
- 不重训、不发布新模型,不改 `MODEL_CARD.md`。

## 18. 修订记录

- 2026-09-28:初版。基于对 Valen(`valen/modeling/qwen/*`、`valen/training/*`、`valen/data/*`)与
  Dohnuts(`src/dohnuts/*`、`scripts/*`、`tests/*`、`docs/*`)的完整探索,以及用户确认的 7 项决策。
  实现前已实测验证:tokenizer `padding_side="right"` 且为 fast;offset 定位得到的 decision token
  不泄漏候选;视觉塔不被 `enable_fusion` 改写;`fused_rms_norm` 的 detach buffer 仅影响语言塔 norm;
  现有 collator 调用方用 `*_` 解包因而兼容元组扩展。
