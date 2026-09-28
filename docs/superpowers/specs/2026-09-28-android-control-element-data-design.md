# Android Control 元素决策数据的构建设计(episode 目录 → Dohnuts direct decisions)

日期:2026-09-28
状态:已实现(转换管线与文档);端到端冒烟与真实数据转换待执行
范围:一个确定性转换管线,把 Android Control 的 episode 目录
(`metadata_{episode_id}.json` + 每步截图 + `step_NNN_a11y.json`)转成 Dohnuts
决策行:5 个问题族、行级 split、排除审计、元素候选与标号截图,可直接喂给现有
`DecisionCollator`/`train.py`/`metrics.py`。

## 1. 背景与目标

已有管线(`src/dohnuts/gui_data.py` + `scripts/prepare_gui_data.py`)把 mobile_use
逐步轨迹转成 4 族决策行,产出 `data/processed/gui-v1`。新增的
`example-data/android_control_parsered/parsered`(符号链接 → 公共 AC 数据)是
15,283 个 episode、99,131 步、83,848 个非空动作,每步带纯 JSON 的 a11y 树
(无 protobuf),因此**可以提取可交互元素**。

目标:在 gui-v1 既有 4 族之上新增"元素选择"决策行(criteria 放可点击元素条目、
截图放标号图),格式对齐 `example-data/train.jsonl` 末条(`dataset="screenshot_choice"`,
键 `r<idx>`,值 `"UI element N: {text/content_description}"`),且整条管线保持确定性。

## 2. 用户已确认的决策

1. **行族范围**:产出全部 5 族 —— action/complete/swipe_dir/button + 新增 element 行。
2. **候选形式**:标号截图 + **仅 clickable 元素**进 criteria(跟随参照行)。
3. **open_app**:**扩展词表** —— action 行用 9 类词表(8 类 + `open_app` 追加在末尾),
   前 8 个下标与 gui-v1 完全对齐;gui-v1 的 8 类词表冻结不动。

## 3. 数据事实(探索已验证)

- 每 episode:`metadata_{id}.json = {episode_id, goal, steps[]}`;step =
  `{step_id, screenshot, accessibility_tree, action, step_instruction}`;末步 `action=null`。
- 动作分布:click 51,918 / scroll 11,130 / input_text 6,091 / wait 5,782 /
  open_app 5,697 / navigate_back 3,034 / long_press 167 / navigate_home 29。
- **无 GT 元素索引**:click/long_press 只有像素坐标,需要对 a11y 节点 bbox 做命中测试反推。
- a11y JSON 是 `windows[].tree.nodes[]` 扁平节点表,`boundsInScreen` 为绝对像素,
  proto3 省略默认值(布尔字段 absent = false);屏幕 96.4% 为 1080×2400。
- 参照行(train.jsonl 末条)的元素编号与 task_progress 叙述是人工/模型产物,不可复现
  → 本项目自定义确定性规则。
- 全语料元素行 token 探针:49,924 行中 520 行(1.04%)超过 `MAX_LENGTH=2048`,
  最大 38,642 tokens(某 a11y 节点的 `text` 是一整篇 PDF),中位数 624、p99 2,070。

## 4. 行 schema(每步 2–3 行,共享 state/group/aliases/reference)

| 行 | dataset | 条件 | target |
| --- | --- | --- | --- |
| `action` | `gui_action` | 每步(含末步) | 9 类 one-hot |
| `complete` | `gui_complete` | 每步(含末步) | noul;末步 → `true` |
| `element` | `screenshot_choice` | click/long_press 步 | 候选元素 one-hot |
| `swipe_dir` | `gui_swipe` | scroll 步 | 4 方向 one-hot |
| `button` | `gui_button` | navigate_back/home 步 | 4 键 one-hot |

- 行 id:`android_control_{episode_id}_step{n}:{family}`,`n` 为步在 episode 内的
  0 起始下标;group:`task:android_control_{episode_id}`。
- 每行的 `reference` 仅作溯源:`{thought:"", action:<当前步 step_instruction>,
  tool_call:{name:"mobile_use", arguments:<映射后参数>}, ac_action:<原始 action 或 null>,
  element_position:<GT 候选位置或 null>}`;训练不读 `reference`。
- 行内不出现 thought、不出现坐标以外的生成式载荷:坐标与 typed 文本是编排层的输入。

## 5. 动作映射(AC → mobile_use 词表)

| AC action_type | 映射 | 派生行 |
| --- | --- | --- |
| click(x,y) | click(像素点在 `reference`) | + element |
| long_press(x,y) | long_press(像素点在 `reference`) | + element |
| scroll(direction) | swipe,**方向取反** | + swipe_dir |
| input_text(text) | type | — |
| wait | wait | — |
| open_app(app_name) | open_app(第 9 类,下标 8) | — |
| navigate_back / navigate_home | system_button(Back/Home) | + button |
| (末步 null) | terminate | complete 目标为 `true` |

- `AC_ACTIONS = {**ACTIONS, "open_app": "open an app by name"}`,前 8 项与 gui-v1
  同序同义;`ACTIONS` 本身不改。
- **scroll 取反**的理由:AC 记录的是内容移动方向(`scroll down` = 查看下方内容 =
  手指上滑),gui-v1 的 `swipe_dir` 来自手指位移;取反后两数据集的 `up`/`down`
  都表示同一手指动作。常量只有一处(`SCROLL_TO_SWIPE_DIRECTION`)。
- 末步 terminate 假设 episode 均为成功演示(parsed 数据已丢 `goal_status`),计入文档与
  manifest(已知限制);结构损坏的 step 一律排除而不是当作末步(避免铸出假 terminate)。

## 6. 元素提取、编号与 GT(纯 stdlib + PIL,`utils/representation_utils.py` 的 JSON 移植)

1. **提取**:按文件顺序遍历 windows、按列表顺序遍历 nodes,保留
   `isClickable` 且 `validate_element` 通过的节点。有效性沿用数据集规则:
   `isVisibleToUser` 非 true 丢弃;bbox 退化或完全出屏丢弃;`boundsInScreen` 缺键按 0。
   不做 `forest_to_ui_elements` 的叶节点/有描述预过滤(会丢 clickable 容器),
   也不 dedup(同框由命中测试平局规则处理)。
2. **编号**:对过滤后的候选列表按顺序 enumerate,**idx = 0..N-1 连续编号、无空洞**
   (非 clickable 或无效节点不占编号)。criteria 键与标号图标签都用这个序号;
   元素 dict 只带 `index`/`text`/`content_description`/`bounds`,节点标志位不外泄。
3. **GT 挂接**:`hit_test` 取包含 (x,y) 的候选中**面积最小**者,面积相同取**位置靠前**者
   (bbox 两端闭区间);无包含 → 整步排除 `no_target_element`(不猜测)。
4. **候选数**:<2 → `too_few_candidates`;>128 → `too_many_candidates`;数量检查先于命中测试。
5. **criteria**:键 `r{idx}` 按 idx 升序,值 `"UI element {idx}: {payload}"`,
   payload = `json.dumps` **仅含非空 `text`/`content_description`**(此键序,
   `ensure_ascii=False`),都没有 → `"{}"`;**不含 index / is_\* / class_name**。
6. instructions 用参照行原文
   `"Which action should be taken next to complete the user's task?"`。

## 7. 标号截图(set-of-mark)

- 只为产 element 行的步渲染:复制原图(**清空 `Image.info`**,丢弃 ICC/DPI/时间戳),
  为每个候选画绿色 (0,255,0) 矩形(线宽 2)+ 左上角白底黑字序号 chip;其余元素不画,
  GT 不高亮,标号集合与 criteria 键一一对应。
- 字号随图高缩放(`max(12, height // 86)`;Pillow ≥ 10.1 的 `load_default(size=…)`)。
- 图按内容哈希存 `<output>/images/<sha>.png`,字节确定性**只在固定 Pillow 版本下成立**
  (默认字体由 FreeType 光栅化、PNG 由 Pillow 编码),因此 manifest 记录
  `environment.python`/`environment.pillow`。
- **隔离 alias**:element 步的**所有行** aliases =
  `[image-bytes:<原图 sha>, image-bytes:<标号图 sha>]`;element 行的 `image` 指向标号图,
  其余行指向原图。非 element 步只有原图 alias。

## 8. state 模板

`state.user_query` = goal 逐字。`state.task_progress` = 确定性模板:
`"(You have done the following operation on the current device): "` +
`" ".join(f"Step {k}: {instruction};")` + `" ."`,instruction 取**已完成步**的
`step_instruction` 逐字(缺失/非字符串的条目跳过并重新连续编号);第 0 步渲染为
前缀 + `"."`。**绝不包含当前步的 instruction**(防答案泄漏)。与参照行的人工叙述不同是有意为之。

## 9. token 预算

- 传 `--model` 且未传 `--no-token-check` 时启用(manifest `token_check` 记 `enabled`/`skipped`);
  模型加载失败不静默降级,`SystemExit` 中止。
- `token_length` 与训练侧一致:`render_question(render(state), question, has_image=True)`
  的分词长度 + `smart_resize` 后图像占位符数(`IMAGE_PIXELS`),比较 `MAX_LENGTH=2048`;
  按**整步**排除(任一行的最大长度进 `detail`),排除发生在写任何文件之前。
- 该检查是必需的:`DecisionCollator` 遇到超预算 batch 直接抛 `ValueError`,从不截断。
- 成本:全语料 99,131 步探针约 6.0 分钟(约为一次完整转换的 1.3%)。

## 10. 排除与审计

- 解析期(整步/整个 episode):`unparsable_metadata`(envelope、step 结构、或目录名与
  `episode_id` 不一致)、`missing_image`、`missing_a11y`、`unknown_action`、
  `too_few_candidates`、`too_many_candidates`、`no_target_element`、`token_budget`、`unexpected`。
  单个 episode 损坏不中止整批(15,283 个文件规模)。
- 隔离期(行级):`cross_split_group`、`duplicate_input`(沿用 gui_v1 规则)。
- 全部写入 `excluded.jsonl`(`{id, reason, detail, stage}`)与 manifest 的 `exclusions`
  计数(`parse:<原因>` / `<dataset>:<split>:<原因>`);解析期 id 用 step id 或 episode 目录名,
  隔离期用带族后缀的行 id。
- **一步不产生部分行**:任一必需问题无法派生即整步排除。

## 11. split、隔离与采样权重

- 分桶完全复用 gui_v1:`SPLIT_SEED="doh-gui-split-2026"`,
  `bucket = int(sha256(seed+":"+group)[:8],16) % 100`,calibration < 10、dev < 20、
  test < 30、其余 train。group 由 episode 决定,故任务天然不跨 split。
- 截图字节作为 alias 逐记录消歧(`train < calibration < dev < test` 优先级最高者胜),
  低优先级 split 中携带同一 alias 的行按 `cross_split_group` 丢弃,任务其余行不受影响;
  element 步的两个 alias(原图/标号图)都参与,且每行都携带两个 alias,所以两种副本
  都不会跨 split。
- dataset 命名:`gui_action`/`gui_complete`/`screenshot_choice`/`gui_swipe`/`gui_button`。
  `TrainingBatches` 按 dataset 名均匀采样,5 个名字等权,因此 button/swipe 行相对加权;
  这是有意选择,写入 manifest `dataset_weighting`。
- 输出目录独立(`data/processed/ac-v1`)而不是与 gui-v1 合并:AC 的 `gui_action` 是
  9 类而 gui-v1 是 8 类,同名混装会让按 dataset 的聚合与 batch 宽度填充混入不同语义。

## 12. 输出物与 manifest

`--output <dir>` 下:`train/dev/calibration/test.jsonl`(即使为空也创建)、
`excluded.jsonl`、`images/<sha256>.png`(原图与标号图,内容寻址)、`manifest.json`:

- `schema_version=1`、`split_seed`、`split_limits`、`path_convention`、`sha256`(四文件)。
- `source`:输入根、episode 数、`metadata_files_hashed`、
  `metadata_sha256`(只覆盖可读 metadata 文件的拼接字节,读不到的文件不计入)。
- `counts`(dataset × split)、`action_classes`(每 split 的动作类别分布,只列出现过的类别)。
- `element_stats`(**post_isolation**,`basis` 键):每 split 的
  rows / candidates min·mean·max / `empty_target_payloads` / `empty_target_payload_rate`;
  没有 element 行的 split 不出现。
- `element_resolution`(**pre_isolation**,`basis` 键):`element_rows`、
  三种拒绝计数、`hit_rate`。
- `exclusions`、`images`(本次运行写入的 PNG,写文件只发生在行已铸出之后,故磁盘与列表一致)。
- `vocabularies`:`ac_actions`(9 类)、`buttons`、`swipe_directions`、
  `instructions`(含 element)、`complete_criteria`,以及白话规则
  `element_rule`/`marked_images`。
- `environment`:`python`/`pillow` 版本(标号图字节与后者绑定)。
- `token_check`、`dataset_weighting`。

## 13. metrics 影响

`src/dohnuts/metrics.py` 的 macro-F1 抑制集合加入 `"screenshot_choice"`:
候选标签逐行不同(每行的 `r0..r{N-1}` 来自各自屏幕),跨行 label 下标做 macro-F1 无意义。
逐行 accuracy、`macro_accuracy` 与 dev 选点仍包含该 dataset。

## 14. 实现落点

| 文件 | 职责 |
| --- | --- |
| `src/dohnuts/android_control_data.py` | `AC_ACTIONS`、`map_action`、`validate_element`、`extract_elements`、`element_bounds`、`hit_test`、`resolve_element_choice`、`element_description`、`parse_metadata`、`parse_step`、`task_progress`、`rows_for_ac_step` |
| `src/dohnuts/android_control_mark.py` | `mark_screenshot`(纯 PIL,set-of-mark 渲染) |
| `scripts/prepare_android_control_data.py` | CLI:episode 发现、metadata 读取与哈希、原图/标号图存储、token 预算、`isolate` → `validate_rows` → 四 split + `excluded.jsonl` + `manifest.json` |
| `tests/test_android_control_data.py`、`tests/test_android_control_mark.py` | 合成 episode fixture(迷你 metadata + PIL 小图 + 手写 a11y JSON)驱动的单元与 CLI 集成测试 |
| `docs/data-and-evaluation.md` | "Android Control conversion" 小节 |

复用 `gui_data` 的 `split_for`/`one_hot`/`image_digest`/`isolate`/`validate_rows`/
`ACTIONS`/`BUTTONS`/`SWIPE_DIRECTIONS`/`COMPLETE_CRITERIA`;`gui_data.validate_rows`
在本次工作中只做了两处等价收紧(自检按期去重解码、`resolve()` 失败也报可读性错误)。

## 15. 验证

1. `pdm run test` 全绿(不依赖 GPU/真实数据):本文档写作时 193 passed,其中 AC 两个文件
   共 105 个用例。
2. `pdm run check`(lint + format-check + typecheck)对仓库内被跟踪文件通过;
   `train.ipynb`(未跟踪)自带的 F541 与本次改动无关。
3. 确定性:同一 `--input`/`--output` 重复转换,manifest 与四文件 sha256 不变;
   标号图有字节级确定性测试与像素级存在性测试。
4. token 一致性:stub 测试钉住算术形状,另有真实 `Qwen/Qwen3.5-0.8B` 处理器与
   `DecisionCollator` 的逐行一致性用例(本地快照存在时运行)。
5. 真实数据探针:元素行 1.04% 超预算(见 §3);23 个真实 episode 的小样转换
   产出 384 行、206 张图、2 条排除(1 条超预算、1 条候选过少),GT 命中率 0.98。
6. 待执行:端到端冒烟(train → calibrate → evaluate → predict)与全语料转换
   `data/processed/ac-v1` 及其检查清单(见计划 Task 7–8)。

## 16. 已知限制

- 末步 terminate 假设 episode 均为成功完成(parsed 数据已丢 `goal_status`)。
- `task_progress` 是模板拼接,非自然语言叙述,与参照行措辞不同。
- GT 元素由命中测试反推(面积最小规则),不是标注给出;命中不了即整步排除。
- 大量 GT 元素的 payload 是 `{}`(真实小样上测得 44–49%):很多可点击容器没有文案,
  信号由标号截图承担;`element_stats.empty_target_payload_rate` 记录比例。
- 标号图字节依赖 Pillow 版本,跨版本重跑会改变文件名与四文件 sha256。
- 标号图与 gui-v1 原图视觉上不同域,element 行的图像分布与 gui-v1 各族不同。
- 标号编号与 `example-data/train.jsonl` 手工 smoke 行不同(该行用带空洞的数据集下标,
  本管线用连续的可点击元素编号)。
- 输入根内的符号链接仍可解析到根外(拒绝绝对路径与 `..`,未做 `resolve()` 校验)。
