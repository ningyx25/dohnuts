# Android Control 转换对齐 mobile-jev prompt 的设计(state + questions 直出决策行)

日期:2026-09-30
状态:已实现(转换管线、测试与 node01 全量运行)
范围:把 `scripts/prepare_android_control_data.py` + `src/dohnuts/android_control_data.py`
的输出从 gui-v1 的 5 个问题族(action/complete/element/swipe_dir/button)换成
**mobile-jev 的 state + questions 形状**:每一行的 prompt 必须逐字等于 android_world
`ClientMobileJev` 对同一屏、同一 goal 会发给 Jev 的请求。gui-v1 管线与其产物不受影响。

参考:`docs/superpowers/specs/2026-09-30-mobile-jev-prompt-construction-design.md`
(端口事实、字段与上限),以及 android_world 提交 `8e39c3b` 的
`android_world/agents/mobile_jev.py` 与上游 `mobile-jev/scripts/mobile-agent/policy.mjs`。

## 1. 目标与成功标准

1. **Parity**:`src/dohnuts/mobile_jev_prompt.py` 构造的 `{'state', 'questions'}` 与
   android_world 端口对同一棵 a11y 树逐字段相等;训练形状
   (`jev_training_prompt`)与端口只差三处,且每一处都被单独断言(见 §7)。
   测试用 stub harness 直接导入端口,**同一个 a11y JSON** 分别喂给两边比较;
   设 `ANDROID_WORLD_ROOT` 时运行,否则跳过。
2. **确定性**:同一输入 serial 与 parallel 产出的行**逐字节相同**(image 路径按文件名归一
   后比较),manifest 除 `sha256` 外完全相同。
3. **可审计**:每个未产出的族/步都有原因,写进 `excluded.jsonl`(`stage: parse|family|isolate`)
   与 manifest 的覆盖率表。
4. **token 不设门禁**:转换阶段不因长度排除任何行;长度由独立脚本在全量产物上测量,
   供之后重新定义预算。
5. gui-v1 的脚本、数据与文档结论不动;旧 `data/processed/ac-v1` 保留。

## 2. 已确认的决策

| # | 决策 |
| --- | --- |
| 1 | 行族**完全替换**为 `jev_operation` / `jev_tap_target` / `jev_scroll_target`(保留 id,不产出行)/ `jev_text_value` / `jev_app_target` |
| 2 | **保留图像**:非 target 行用原图,`jev_tap_target` 行用标号图(编号 = criteria 键) |
| 3 | state **严格 9 键**(+ 可选 `focusedField`),去掉 `task_progress`,历史只走 `recentActions` |
| 4 | app 清单**两遍扫描**:第一遍收集全语料 `open_app` 显示名当"已安装清单",再按 `_named_apps` 语义整词收窄 |
| 5 | 跳过 `jev_scroll_target`(AC 的 scroll 只有方向、没有坐标) |
| 6 | `text_value` **严格**:候选恒为 goal 的 1..8 元 n-gram + `NONE`,GT 不命中就不产出行 |
| 7 | token 预算按全量统计定为 `MAX_LENGTH = 8192`,并在转换时作为门禁(`--model` 启用) |
| 8 | 行候选上限提到 **255**,与端口一致 |
| 9 | **四个 `SCROLL_*` 合并为一个 `SCROLL`**,方向由新增的 `jev_scroll_direct` 族选择 |
| 10 | **不产出 `jev_text_value`**;`TYPE_TEXT` 仅作为 operation 的一个选项保留 |
| 11 | `jev_app_target` 候选:优先用 goal 整词命中的 app;命中不足 2 个或 GT 不在其中时,从语料词表**确定性随机**抽 15–30 个 + GT |
| 8 | 统计以**完整语料**为准(`node01:/home/ningyongxin/workplace/proj/dohnuts/example-data/android_control_parsered`) |

## 3. 代码落点

| 文件 | 职责 |
| --- | --- |
| `src/dohnuts/jev_training_prompt.py`(新) | 训练形状的三处偏离:`SCROLL` 合并与 `scroll_direct` 问题、app 候选采样、`text_value` 不出行;端口本身不动 |
| `src/dohnuts/mobile_jev_prompt.py`(新) | 端口的逐字移植:`RULES`、operation 说明、target 模板、`text_candidates`、`named_apps`、`candidates_for`、`describe_action`/`element_label`、`build_questions`、`build_state`、`build_request`;`Element`/`Phone`/`Observation`/`Action`/`Candidate`/`HistoryEntry`/`QuestionSpace`/`Request` 数据结构 |
| `src/dohnuts/android_control_data.py`(重写) | AC → `Observation` 的观测层(`iter_nodes`、`clamped_bounds`、`foreground_package`、`summarize_observation`)、候选与命中(`element_bounds`/`element_hits`/`element_target_weights`)、`step_operation`、`parse_episode`、`resolve_target`、`rows_for_ac_step`、`app_vocabulary` |
| `src/dohnuts/android_control_mark.py` | `mark_screenshot(image, [(label, element), …])`:标签由调用方给,即 criteria 键 |
| `scripts/prepare_android_control_data.py` | 两遍转换、内容寻址存图、`isolate` → `validate_rows` → 四 split + `excluded.jsonl` + `manifest.json`;**token 门禁移除** |
| `src/dohnuts/token_stats.py` + `scripts/report_token_lengths.py`(新) | 全量 token 测量:`token_lengths.jsonl`(逐行)+ `token_stats.json`(每 dataset×split 的 min/p50/p90/p99/max 与 over_max_length) |
| `src/dohnuts/predictor.py` | `render_question` 用 `render()` 渲染 instructions,`{goal, rules}` 以 JSON 形态进 prompt(字符串 instructions 输出不变) |
| `src/dohnuts/metrics.py` | 四个 `jev_*` 数据集进入 macro-F1 抑制集 |

## 4. 行 schema

行信封不变:`{id, dataset, group, aliases, split, state, image, question, target, reference}`;
`id = android_control_{episode}_step{n}:{family}`、`group = task:android_control_{episode}`;
`reference` 增加 `question_id`(逐行),保留 `thought/action/tool_call/ac_action/element_positions`
(element_positions 现在是"在 TAP 候选里的命中位置")。

| 族(dataset) | 何时产出 | question id | criteria | target |
| --- | --- | --- | --- | --- |
| `jev_operation` | 每个解析成功的步 | `operation` | 该屏在场的 operation,固定键序(四个 `SCROLL_*` 合并为一个 `SCROLL`) | GT operation 的 one-hot |
| `jev_tap_target` | click/long_press 且命中 TAP 候选、2..255 候选 | `tap_target` | `[i] label` | 命中候选上的逆面积分布 |
| `jev_scroll_direct` | scroll 步 | `scroll_direct` | `DOWN/UP/LEFT/RIGHT`(值沿用端口四句说明) | 记录方向的 one-hot |
| `jev_app_target` | open_app 且 GT ∈ 候选、2..255 候选 | `app_target` | goal 命中的 app,否则 15–30 个采样 + GT | 该 app 的 one-hot |
| (`jev_text_value`) | 不产出 | — | — | — |
| (`jev_scroll_target`) | 不产出 | — | — | — |

每步 1–2 行(operation + 至多一个 target 族),与端口"只校验被选中分支"一致。
`reference.element_positions` 只对 `jev_tap_target` 有意义(其余族为 `[]`)。

## 5. state 与观测层

- `state` 键序与端口一致:`goal, app, isEditable, textSource, textEntryAvailableAfterFocus,
  visibleText, elements, availableApps, recentActions`,可选 `focusedField` 追加在最后。
- `app` = `foreground_package(document)`:排除 IME 后,`TYPE_APPLICATION` 中**可见面积最大**的
  窗口的多数 `packageName`;无该类型窗口时退化为"排除 IME 后面积最大的窗口"。文件顺序破平局。
  实测与"节点最多的非 IME/systemui 窗口"规则在 1,000/1,000 步上一致。
- 观测过滤链与端口 `summarize_state` 相同:丢弃不可见、clamp 后空框、以及
  `text`/`label`/`clickable`/`editable`/`scrollable` 全空的节点;**元素 id 是节点在扁平列表里的
  原始下标**(被丢弃的节点照样占号),`elements`/`criteria` 的 index 从 1 开始、TAP 先编号、
  再是 scroll-only 区域。
- `recentActions`:前序步最后 8 条 `{operation,label,screenChanged[,text]}`;`label` 用端口
  `describe_action` 渲染;`screenChanged` 由相邻两步的 a11y 指纹(端口同构的规范 JSON sha256)
  比较得出。**已知近似**:SCROLL 的 `region_id` 取该屏下标最小的滚动候选(无候选时空缺)。
- 顶层 `scroll_target` 问题仍由构造器生成(state 与端口同形),只是不产出对应行。

### 5.1 SCROLL 合并与 `scroll_direct`

- 四个 `SCROLL_*` 在 operation criteria 里折叠成一个 `SCROLL`(位置 = 原先第一个方向的位置,
  即 `TYPE_TEXT` 之后、`BACK` 之前),说明文案:
  `Scroll the screen to reveal more content in one direction; another question chooses that direction.`
- `state.elements[i].operations` 去重成单个 `SCROLL`(与 TAP 并存时为 `['TAP','SCROLL']`)。
- 新增问题 `scroll_direct`,criteria 固定为 `DOWN/UP/LEFT/RIGHT`,值逐字复用端口的四句
  `SCROLL_*` 说明;instructions 为新增文案,并按端口惯例包成 `{goal, rules}`。
- GT = AC `scroll` 的 `direction`(恒等映射);方向非法时 `step_operation` 仍返回 `None`
  (整步 `unknown_action`)。
- history 的 `operation` 记为 `SCROLL`,label 继续用端口的 `describe_action`(含方向、近似区域
  与 gesture JSON)。

### 5.2 app 候选采样

- `named = named_apps(词表, goal)`(端口整词规则)。
- `len(named) >= 2` 且(该步没有打开 app,或 GT ∈ named)→ 候选 = `named`(词表顺序)。
- 否则 → `rng = Random(f'{APP_SAMPLE_SEED}:{step_id}:{gt_app}')`,
  `k = rng.randint(15, 30)`,`干扰项 = rng.sample(词表 - {GT}, min(k, 其余数量))`,
  候选 = `{GT} ∪ 干扰项` 按词表顺序;GT 不在词表里时追加在末尾。
- 种子只依赖 (step_id, GT) 与输入词表 → serial/parallel、重跑都抽到同一组名字。
- `state.availableApps` 与 `questions.app_target.criteria` 同步替换(端口本来同源)。
- 词表太小(候选 < 2)时按 `too_few_candidates` 丢该族。

## 6. 上限与排除

| 情形 | 处理 |
| --- | --- |
| 结构损坏 / 图片不可读 / a11y 不可读 / 未知动作 | 整步排除 `unparsable_metadata` / `missing_image` / `missing_a11y` / `unknown_action` |
| 请求超过 150 KB(端口的 `MAX_PAYLOAD_BYTES`) | 整步排除 `payload_too_large` |
| GT operation 不在该屏 criteria | 整步排除 `operation_not_offered` |
| 点击没命中任何候选 | 丢 `tap_target`,`no_target_element` |
| 候选 <2 或 >255(每个 choice 族都会检查) | 丢该族,`too_few_candidates` / `too_many_candidates` |
| 任一行超过 `recipe.MAX_LENGTH`(8192,`--model` 启用) | 整步排除 `token_budget`,`detail` 记最长行 |
| `scroll_direct` 里没有记录的方向 | 丢 `scroll_direct`,`no_target_element`(方向非法时已在整步排除) |
| GT app 不在候选 | 丢 `app_target`,`app_not_offered` |


族级丢弃写 `excluded.jsonl` 的 `stage: "family"`(`id = {step}:{family}`),并在 manifest 的
`family_coverage[family].dropped` 里归因到**提出该问题的族**(计数来自 worker,不靠 reason 反推)。

## 7. 对齐测试(parity)

`tests/test_mobile_jev_prompt.py` 把同一份 a11y JSON 分别交给

- android_world 端口:序列化成 `UIElement` 列表 → `summarize_state` → `MobileJevPolicy.decide`
  (假 `JevWrapper` 捕获请求);
- 本仓库:`android_control_data.summarize_observation` → `jev_training_prompt.build_training_request`;

然后按三类断言:

1. **无偏离时逐字节相等**(无滚动区 + goal 命中 ≥2 个 app 且含 GT)—— 覆盖 RULES、state 键序、
   元素编号、tap/scroll_target 问题等绝大部分 prompt 面;
2. **只有文档化的 scroll 重构不同**:把 `operation.criteria`、`elements[*].operations`、
   `scroll_direct` 从两侧剔除后逐字节相等,并单独断言合并后的键序、方向问题的四项与
   `scroll_target` 仍在;
3. **只有文档化的 app 采样不同**:剔除 `app_target` 与 `availableApps` 后逐字节相等,并断言
   候选含 GT、数量在 16..31。

真实语料用例(12 个 episode × 2 步)用第 2、3 类的"剔除后比较"。
`ANDROID_WORLD_ROOT` 未设置时这些用例跳过,仓库不新增依赖。

## 8. 实测(node01 全量 15,283 episode)

转换命令见 §11(`--model` 打开 token 门禁、`--workers 32`),墙钟约 45 分钟(收尾的
isolate/validate 要逐个走 146k 个 PNG 的容器校验),产物 74GB。

| 指标 | 值 |
| --- | --- |
| episode / metadata 哈希 | 15,283 / 15,283 |
| 步 | 99,131 步中 97,503 步产出请求;步级排除 `operation_not_offered` 1,612、`token_budget` 86、`missing_a11y` 10、`payload_too_large` 6 |
| 行(写出) | **163,551**:`jev_operation` 97,138、`jev_tap_target` 50,128、`jev_scroll_direct` 10,618、`jev_app_target` 5,667 |
| split | train 114,392、calibration 15,482、dev 17,058、test 16,619 |
| 覆盖率(pre-isolation) | operation 97,417/97,503 = 0.999;tap_target 50,237/51,947 = 0.967;**scroll_direct 10,637/10,640 = 0.9997**;**app_target 5,697/5,697 = 1.000** |
| 族级丢弃 | 只有 tap_target:`no_target_element` 1,280、`too_few_candidates` 361 |
| 隔离期丢弃 | 437(`cross_split_group` + `duplicate_input`) |
| 图像 / app 词表 | 146,462 个内容寻址 PNG / 758 个显示名 |
| 确定性 | 60-episode 冒烟上 serial 与 `--workers 4` 的 667 行逐字节一致(含采样后的 app 候选);manifest 仅 `sha256` 不同 |

**token 分布**(`report_token_lengths.py`,`Qwen/Qwen3.5-0.8B` 的 processor,
`IMAGE_PIXELS=512²`,`MAX_LENGTH = 8192`):

| dataset | rows | p50 | p90 | p99 | max | >8192 |
| --- | --- | --- | --- | --- | --- | --- |
| `jev_operation` | 97,138 | 2,041 | 3,393 | 5,035 | 8,175 | 0 |
| `jev_tap_target` | 50,128 | 1,962 | 3,776 | 6,323 | 8,185 | 0 |
| `jev_scroll_direct` | 10,618 | 1,908 | 3,027 | 4,751 | 7,956 | 0 |
| `jev_app_target` | 5,667 | 1,693 | 2,668 | 4,352 | 6,899 | 0 |
| **全部** | **163,551** | **1,995** | **3,487** | **5,305** | **8,185** | **0** |

预算的来历:首轮用旧的 2048 测量时,p50 就已经是 2,128、p99 6,653,78,546/150,300 行(52.3%)
超预算——旧预算是按 gui-v1 的 `{user_query, task_progress}` 定的。按 p99 取整定为 **8192**,
同时把 `Qwen35Adapter.max_input_tokens` 从 4096 提到 8192。本次门禁只切掉 86 步(0.09%),
`over_max_length` 为 0 即门禁生效。

**迭代历史**(同一管线,三次全量):

| 版本 | 行数 | 变化 |
| --- | --- | --- |
| 首轮:128 候选上限、2048 预算、旧族 | 150,300 | `text_value` 30.9%、`app_target` 23.2%;`token_budget` 未设门禁 |
| 第二轮:255 候选上限、8192 门禁 | 151,499 | text_value +1,396、app_target +191;门禁切 227 步 |
| 本轮:SCROLL 合并 + `scroll_direct` + 去 text_value + app 采样 | **163,551** | 新增 `scroll_direct` 10,618;`app_target` 1,507 → **5,667**(覆盖 100%);text_value 归零;门禁只剩 86 步;p90 从 4,793 降到 3,487 |

## 9. manifest

`schema_version: 2`,新增/替换:

- `family_stats`:每 split 每 dataset 的行数、候选 min/mean/max、`soft_targets`、`max_positive_weights`;
- `family_coverage`:每族 `steps_asking`(**按 GT operation 归类**)、`rows`(pre-isolation)、
  `rate`、`dropped`(按族归因),带 `basis` 说明;
- `family_drops`:所有 `family:<reason>` 计数;
- `app_inventory`:`{apps, offered_cap, source}`;
- `vocabularies`:新增 `rules`(771 字符原文)、`operation_descriptions`、`target_question_template`、
  `text_value_none`、`text_value_instructions`、`state_keys`、`question_ids`、`limits`、
  `prompt_rule`、`target_rule`、`marked_images`、`deviations`(§10 的清单);
- `token_stats`:指向 `scripts/report_token_lengths.py` 的产物(转换本身不设门禁);
- 移除 `token_check`;`element_stats`/`element_resolution`/`action_classes` 由 `family_stats`/
  `family_coverage`/`operation_classes` 取代。

## 10. 已记录的偏差(manifest `vocabularies.deviations`)

1. 行带截图(端口纯文本):非 target 行原图,tap_target 行标号图。
2. **一个 `SCROLL` 取代端口的四个 `SCROLL_*`,方向改由新增的 `scroll_direct` 问题选择** ——
   端口没有这个问题。
3. **app 候选 = goal 命中(≥2 且 GT 在其中)否则确定性采样 15–30 个 + GT**,而不是端口的
   安装清单(≤200)。
4. **不产出 `jev_text_value` 行**(端口的问题仍在 request 里)。
5. `jev_scroll_target` 不产出(AC 无 scroll 坐标)。
6. history 里 SCROLL 的区域取最小下标滚动候选。
7. app 词表来自语料自身的 `open_app` 名。
8. 末步视为成功终止(parsed 语料丢了 `goal_status`)。
9. `focusedField` 的键位置以 android_world 端口为准(与上游 JS 不同)。

**服务侧后果**:第 2、3 条使训练数据的请求不再与 `ClientMobileJev` 逐字相同 —— 要用这个模型
替换 Jev,需要在服务侧加一层桥接(`SCROLL` + 方向映射回 `SCROLL_*`;app 侧要么用同一采样规则,
要么接受设备给出的任意清单)。

## 11. 运行

```bash
# 本地:转换(不加载模型,不需要 token 检查)
python scripts/prepare_android_control_data.py \
  --input example-data/android_control_parsered/parsered \
  --output data/processed/ac-jev-v1-subset --workers 8

# 全量(node01,完整语料 15,283 episode;.venv 为 3.12 + Pillow 12.3.0 + transformers 5.17)
# --model 打开 token 门禁(MAX_LENGTH = 8192),--no-token-check 可关掉
ssh node01
cd /home/ningyongxin/workplace/proj/dohnuts && git pull ningyx25 <branch>
.venv/bin/python scripts/prepare_android_control_data.py \
  --input example-data/android_control_parsered/parsered \
  --output data/processed/ac-jev-v1 --model Qwen/Qwen3.5-0.8B --workers 32

# 长度测量(不排除任何行,只产出分布)
.venv/bin/python scripts/report_token_lengths.py \
  --input data/processed/ac-jev-v1 --model Qwen/Qwen3.5-4B-Base --workers 32

# 对齐测试(可选,需要 android_world 检出)
ANDROID_WORLD_ROOT=/path/to/android_world pytest tests/test_mobile_jev_prompt.py
```

**产物**(node01 `data/processed/ac-jev-v1/`,74GB):
`train.jsonl`(877MB)/`dev.jsonl`/`calibration.jsonl`/`test.jsonl`、
`excluded.jsonl`(1.3MB)、`manifest.json`、`images/`(146,613 个 PNG)、
`token_lengths.jsonl`(逐行长度)、`token_stats.json`(分布)。
本地子集冒烟与 node01 全量使用的都是同一份代码,`manifest.source.metadata_files_hashed`
可确认输入完整。
