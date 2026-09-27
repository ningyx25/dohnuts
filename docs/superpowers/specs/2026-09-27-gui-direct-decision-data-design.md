# GUI 逐步决策数据的构建设计(raw_data → Dohnuts direct decisions)

日期:2026-09-27
状态:待实现
范围:一个确定性转换管线,把 GUI Agent 的逐步 SFT 轨迹(`raw_data.json`)转成
Dohnuts 决策行,并在 <1k 步的规模下给出可复现的 split、审计与验证。

## 1. 背景与目标

`example-data/raw_data.json` 是轨迹式 SFT 数据:每条记录一个 step,含 user 消息
(用户指令 + 已完成步骤历史 + `<image>`)与 assistant 消息(Thought + Action +
`<tool_call>`);`images` 指向该步截图;`bbox` 为 null(批量数据同样只有正确动作,
没有元素清单)。

`example-data/{train,dev,calibration,test}.jsonl` 里已有一条手工转换的 smoke 行:
`state={user_query, task_progress}` + 截图 + `question=choice`(5 个含坐标的候选动作),
target 指向正确动作,`reference` 保留 thought/action/tool_call 以便溯源。该行证明了
Dohnuts 决策 schema 能承载 GUI 步骤,但候选是手工写的,无法扩展。

目标:把"每步的下一步动作"转成**分解式多问题**决策数据——不依赖候选清单、全部由
GT tool_call 自动派生、与现有 `DecisionCollator`/`Predictor`/`train.py`/`metrics.py`
完全兼容,可直接喂给 `train.ipynb` 的 C 流程。

## 2. 范围

### 2.1 v1 覆盖

- 输入:每步一条记录,一条记录恰好一张截图。
- 每步产出 2–4 行决策:`action`(8 类)、`button`(4 类,仅 system_button 步)、
  `complete`(noul)、`swipe_dir`(4 类,仅 swipe 步)。
- 任务级 group 隔离、确定性 split、排除计数、清单与哈希。

### 2.2 v1 不覆盖(明确不做)

- **区域选择(点哪个像素)**:候选取自 UI 检测器,属推理侧职责;后续版本可用
  离线检测器产出 ScreenQA 式区域 choice + 候选级 noul。
- **生成式载荷**:`type`/`answer` 的文本、`wait`/`long_press` 的时长,由编排层提供。
- **多图步、跨步状态压缩、在线反馈信号**。

## 3. 术语与既有约定

- 决策行 = `{id, dataset, group, aliases, split, state, image, question, target, reference}`,
  与 `scripts/prepare_data.py` 产出的行同构(额外键如 `reference` 允许存在,训练与
  评估不读取)。
- `question.type ∈ {choice, noul, score}`;`choice` 候选 2–128 个;`target` 归一分布。
- 渲染与训练共用 `predictor.render_question`;marker 为 `<|fim_suffix|>`。
- 规则来源:`docs/data-and-evaluation.md`(固定词表、不注入答案、审计计数、按组隔离)。

## 4. 数据行 schema 与示例

`645_BrowserMaze_step3` 产出 3 行(swipe 步才有第 4 行):

```json
{"id": "645_BrowserMaze_step3:action",
 "dataset": "gui_action",
 "group": "task:645_BrowserMaze",
 "aliases": ["image-bytes:<sha256>"],
 "split": "train",
 "state": {"user_query": "In the Files app, ...", "task_progress": "(You have done the following operation on the current device): Step 1: ...; ."},
 "image": "data/processed/gui-v1/images/<sha256>.png",
 "question": {"type": "choice",
   "instructions": "What is the next action the agent should take?",
   "criteria": {"click": "tap a single point on the screen", "long_press": "press and hold a point for some time", "swipe": "drag from one point to another", "type": "enter text into the active input field", "answer": "output the answer to the user", "system_button": "press a system button", "wait": "wait for the screen to change", "terminate": "finish the task and report the result"}},
 "target": [0, 0, 0, 0, 0, 1, 0, 0],
 "reference": {"thought": "...", "action": "...", "tool_call": {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Back"}}}}

{"id": "645_BrowserMaze_step3:button",
 "dataset": "gui_button",
 "question": {"type": "choice",
   "instructions": "Which system button should be pressed?",
   "criteria": {"Back": "return to the previous screen", "Home": "go to the home screen", "Menu": "open the application menu or recents", "Enter": "press the enter key"}},
 "target": [1, 0, 0, 0]}

{"id": "645_BrowserMaze_step3:complete",
 "dataset": "gui_complete",
 "question": {"type": "noul",
   "instructions": "Should the agent terminate the task now?",
   "criteria": {"false": "no, more UI actions are needed", "true": "yes, the agent should terminate now"}},
 "target": [1.0, 0.0]}
```

要点:

- `state` 只含 `user_query` 与 `task_progress`(逐字保留源文本,不清理标点),**绝不含
  thought**(项目规则:rationale 不进模型输入)。`reference` 仅作溯源,训练不读。
- 输出目录约定放在 `.gitignore` 覆盖的 `data/processed/` 下(与现有 `data/processed/v1`
  一致),原始数据与产物不进 Git。
- 候选顺序即词表顺序;`target` 下标与之对齐;训练期 `DecisionCollator` 的候选置换
  会同步置换 `criteria` 与 `target`,无需数据侧处理。
- `swipe_dir` 行:`{"type": "choice", "instructions": "In which direction should the screen be swiped?",
  "criteria": {"up": "swipe towards the top edge", "down": "swipe towards the bottom edge",
  "left": "swipe towards the left edge", "right": "swipe towards the right edge"}}`。

## 5. 派生规则

检查顺序固定为:**结构**(记录是对象、`messages` 是列表、条目都是对象)→ **轮次**
(恰好 1 user + 1 assistant)→ **state 模板** → **tool_call** → **记录 id** →
**action** → **button** → **swipe** → **图片**;先命中的原因码胜出。

解析一条原始记录:

1. `state`:取 user 消息,只去掉**尾部的** `<image>` 标记(`re.sub(r"<image>\s*$", "", ...)`,
   文本其余部分逐字保留);用一条锚定模板的正则解析(允许 query 与 history 自身含换行):
   `^The user query:\s*(?P<q>.*?)\nTask progress\s*(?P<p>.*?)\s*$`(DOTALL)。
   `user_query` 为 `The user query:` 之后的文本;`task_progress` 为 `Task progress`
   标签之后的全部文本——含源文本里的 `(You have done ... device):` 引导语与结尾的
   `; .`,不做标点清理或括号剥离(与现有 smoke 行仅差源文本自带的括号)。不匹配 →
   排除 `unparsable_state`。
   记录必须**恰好含一条 user 与一条 assistant 消息**;条数不为 1(多轮)或结构非法
   (记录不是对象、`messages` 不是列表、条目不是对象)→ 排除 `multi_turn` /
   `unparsable_state`。条目是对象但 `content` 非字符串时不计入轮次统计:一条合法轮次
   都没有 → `unparsable_state`;有合法轮次但 user/assistant 各自不为 1 → `multi_turn`。
   本版一记录一步,不猜测该取哪一轮。
2. `tool_call`:取 assistant 消息中**第一个** `<tool_call>...</tool_call>` 块解析 JSON。
   缺失或不是合法 JSON → `missing_tool_call`;`name != "mobile_use"` → `unknown_tool`。
   记录自身必须带非空字符串 `id` → 否则 `missing_id`。
3. `action` = `arguments.action`;必须是非字符串以外都判非法——即**仅当它是 8 类词表中的
   字符串**(`isinstance(action, str) and action in ACTIONS`)才接受,否则 `unknown_action`。
   非字符串值(list/dict 等)同样落到 `unknown_action`,不抛异常。
4. `button` 行:仅当 `action == "system_button"`;`arguments.button` 必须是 4 类词表中的
   字符串,否则 `invalid_button`(同样先做 `isinstance` 检查)。
5. `complete` 行:每步都产出;target = `[0.0, 1.0]` 当且仅当 `action == "terminate"`,
   否则 `[1.0, 0.0]`。正例 ≈ 任务数,类不平衡如实报告,不做标签感知采样。
6. `swipe_dir` 行:仅当 `action == "swipe"`;取 `arguments.coordinate` 与
   `arguments.coordinate2`(各两个数值)。`dx = x2 - x`,`dy = y2 - y`:
   `|dx| > |dy|` → 按 `dx` 符号取 left/right;`|dy| > |dx|` → 按 `dy` 符号取 up/down;
   `|dx| == |dy|` 或坐标缺失/非数值/长度不对 → `invalid_swipe`(平局排除,不猜测)。
7. `image`:`images` 必须是长度恰为 1、且元素为非空字符串的列表 → 否则 `multi_image`;
   该路径必须是**相对路径**且不含 `..` 上跳(绝对路径或越界一律拒绝,保证输出只由输入
   内容决定);相对输入目录解析后必须存在且能以 RGB 打开 → 否则 `missing_image`。
   图片复制到 `<out>/images/<sha256>.png`(按内容去重),行内存**相对仓库根目录**的路径
   (仓库根 = `git rev-parse --show-toplevel`,转换与训练都必须从仓库根运行,与现有
   smoke 行一致);manifest 记录 `path_convention`。若基准目录不同则拒绝运行。

## 6. 排除与审计

排除原因(写入 `excluded.jsonl` 与 manifest 计数):
`unparsable_state`、`multi_turn`、`missing_tool_call`、`unknown_tool`、`missing_id`、
`unknown_action`、`invalid_button`、`invalid_swipe`、`multi_image`、`missing_image`、
`token_budget`、`duplicate_input`、`cross_split_group`、`unexpected`。

`unexpected` 是 CLI 记录级兜底:转换器在单条记录上除 `parse_step` 之外仍可能遇到
文件系统或解码层面的异常(超长文件名、解压炸弹、深层嵌套 JSON 等);该记录按
`unexpected` 排除并在 `detail` 记录异常类型与消息,转换继续,不中断整批。

**键名与落盘(统一方案)**:manifest 的 `exclusions` 是扁平计数表,键统一为
`<来源>:<原因>` —— 解析期与 token 预算为 `parse:<原因>`,隔离期为
`<dataset>:<split>:<原因>`(沿用 `scripts/prepare_data.py` 的命名习惯)。
`excluded.jsonl` 每行 `{id, reason, detail, stage}`:`stage` 取 `parse` 或 `isolate`,
**解析期与隔离期的丢弃都要写入**(隔离期丢弃时 `id` 用该行的 `id`,`reason` 为
`duplicate_input` 或 `cross_split_group`)。`duplicate_input` 覆盖两种情形:同一
`(dataset, group, state, question)` 的重复输入,以及两条记录铸出同一行 `id`(后者在
`detail` 记 `row id already seen`,排除后出现的那条,避免自检直接中止整批)。内容重复
时保留输入顺序中**先出现**的那条;转换器对输入文件排序,因此给定相同输入与相同输出
目录,结果可复现。

- 一条原始记录若任一必需问题无法派生,整条记录排除(不产出部分行)——避免"某步只训
  一半问题"造成的分布偏斜;`button`/`swipe_dir` 本就是条件行,其缺失不算失败。
- `token_budget`:用与训练完全相同的 `render_question(render(state), question,
  has_image=True)` 加处理器分词,并按 `smart_resize` 公式加图像 token,超过
  `MAX_LENGTH=2048` 的行排除(做法同 `prepare_data.filter_data`)。
- `duplicate_input`:`digest(json([dataset, group, state, question], sort_keys=True))`
  去重,保留首个。

## 7. split、group 隔离与采样权重

- `group = "task:" + 任务id`,`任务id` 由记录 id 去掉 `_step\d+$` 后缀得到
  (无法匹配时整条 id 作任务 id)。同一任务的所有 step 共用一个 group。
- 分桶(常量:`SPLIT_SEED = "doh-gui-split-2026"`):
  `bucket = int(digest(SPLIT_SEED + ":" + group)[:8], 16) % 100`
  → `< 10` calibration,`< 20` dev,`< 30` test,其余 train(≈70/10/10/10)。
- 图片内容 `sha256` 作为 alias 参与 group 并查(union-find),防止同一张截图出现在两个
  split;并查后的组按优先级 `train < calibration < dev < test` 保留最高优先级分区,
  其余行排除并计数 `cross_split_group`。合并后的组名取该组**任务成员**的字典序最小值
  (alias 只参与并查、不参与命名,否则合并组会被改名为图片哈希),与输入分片顺序无关。
  合并组名与最终保留哪个分区无关(取最小值,不取幸存者)。**但输出哈希的完全可复现性
  以「相同输入文件、相同记录顺序、相同 `--output`」为前提**:行的 `image` 字段内嵌输出
  目录下的图片路径,内容重复时保留首个,二者都依赖调用方式。
- dataset 命名:`gui_action` / `gui_button` / `gui_complete` / `gui_swipe`,按问题类型
  分开。理由:`metrics.py` 只对固定候选词表计算 macro-F1,混在一个 dataset 里会失真。
  代价:`TrainingBatches` 按 dataset 名均匀采样,4 个问题族各得约 1/4 更新,少量
  button/swipe 行被相对加权。这是有意选择,写入 manifest(`dataset_weighting`),不改
  采样器代码。
- 已知局限:任务级隔离不阻止"不同任务共用同一 app 同屏"的近邻泄漏;在 manifest 记录
  该限制,不声称语义去重。

## 8. 输出物

`--output <dir>` 下:

- `train.jsonl` / `dev.jsonl` / `calibration.jsonl` / `test.jsonl`(每行一个 JSON 对象,
  无空行;行数与 dataset 计数进 manifest;四个文件即使为空也创建)。
- `manifest.json`:`schema_version=1`、`split_seed`、输入来源(目录与每个输入文件的
  sha256)、`counts`(dataset × split)、每 split 的动作类别分布、`exclusions` 计数、
  候选词表与 instructions 原文、`dataset_weighting`、`path_convention`、
  四个输出文件的 sha256。
- `excluded.jsonl`:每行 `{id, reason, detail, stage}`(解析期 `stage=parse`,隔离期
`stage=isolate`)。
- `images/<sha256>.png`:按内容去重的截图副本。

## 9. 验证

1. **转换器内置自检(默认开启)**:id 全局唯一;`split` 字段与所在文件一致(未知 split
   也判失败);target 归一且长度等于候选数;候选数 2–128;图片可 RGB 打开;group 不跨
   split。自检失败一律抛**带行 id** 的 `ValueError`(含未知 split 与图片不可读),转换器
   把它转成一行 `SystemExit` 信息并非零退出,而不是裸回溯。
2. **token 预算检查**:`--model` 给出本地模型路径时执行(见 §6);模型缺失时跳过并在
   manifest 记录 `token_check: "skipped"`。
3. **单元测试** `tests/test_gui_data.py`:
   - fixture = 测试内构造的合成记录(JSON 字典 + PIL 生成的临时截图),**不依赖**被
     `.gitignore` 忽略的 `example-data/`,保证 CI 可跑;
   - 断言产出 3 行(action/button/complete)、schema 正确、target 与词表对齐、id 与
     group 符合规则;
   - 确定性:同一输入在同一 `--output` 目录重复转换,输出文件 sha256 相同;
   - 排除用例:未知 action、缺图、无 `<tool_call>`、swipe 平局、多图、state 不可解析、
     缺 id;
   - 另加一条集成断言:`example-data/raw_data.json` 存在时(本地)转换该样例,断言
     3 行、target 指向 `system_button`/`Back`/非终止;文件缺失则 `pytest.skip`;
   - 不依赖 GPU/模型(token 检查关闭)。
4. **端到端冒烟**:构造 ~100 个合成任务(改写样例记录的 id,并覆盖 8 类 action、
   swipe 方向与 terminate 步)转换,再喂给 `train.ipynb`(小 STEPS)跑通
   train → calibrate → evaluate → predict;四个 split 均非空。
5. 真实数据转换后:报告每 split/dataset/动作类别计数与排除计数,确认校准集
   `choice` 与 `noul` 行各 ≥10(否则该类型温度保持 1.0)。

## 10. 实现落点

- `src/dohnuts/gui_data.py`:纯函数核心(解析、派生、split、并查隔离、去重),
  只依赖标准库与 PIL;可被测试直接导入。
- `scripts/prepare_gui_data.py`:CLI 外壳(`--input`、`--output`、`--model`、
  `--no-token-check`),输入发现(`*.json`,文件名排序保证确定性)、写文件与 manifest。
- `tests/test_gui_data.py`:见 §9.3。
- `docs/data-and-evaluation.md`:新增小节 "GUI step conversion",记录词表、规则、
  排除策略、split seed 与限制。

## 11. 服务端契约(v1)

一次前向给同一 state 上的多个问题;编排层按答案组装 tool_call:

| 模型答案 | 编排层动作 |
| --- | --- |
| `action = system_button` + `button` | 发系统按键 |
| `action = swipe` + `swipe_dir` | 映射为标准滚动手势(固定起终点模板),或由检测器给坐标 |
| `action ∈ {click, long_press}` | 坐标由 UI 检测器/编排层提供(v1 不含) |
| `action ∈ {type, answer}` | 文本由编排层提供(v1 不含载荷) |
| `action = terminate` 或 `complete = true` | 结束任务 |

明确:本版模型是**决策模块**(做什么 / 按哪个键 / 往哪滑 / 是否结束),不是端到端
坐标生成器;文档与模型卡不得宣称坐标级 grounding 能力。

## 12. 复现性与版本

已知限制:输入的 `images` 若是指向输入根之外的**符号链接**,仍会被成功读取(只拒绝
绝对路径与 `..` 上跳);如需要可后续版本用 `Path.resolve()` 再校验。

- 转换是确定性的:输入文件哈希 + 固定词表/规则 + `SPLIT_SEED` 决定输出;任何规则、
  词表或比例变更都会改变输出 sha256,必须重新训练(`train.py` 会比对四文件 SHA)。
  可复现性以**相同输入文件、相同记录顺序、相同 `--output`** 为前提(见 §7:行内嵌
  输出目录下的图片路径,内容重复时保留首个)。
- `schema_version=1`;未来新增问题类型(区域、时长、三分类 terminate)时递增,并在
  manifest 记录。

## 13. 未来工作

- 区域选择:离线跑 UI 检测器,产出区域 choice 与候选级 noul(负例采样规则与类别
  平衡入 manifest)。
- `terminate` 三分类(continue / terminate_success / terminate_failure)。
- `wait`/`long_press` 时长的分箱 score 问题;`type`/`answer` 载荷的 noul 校验。

## 14. 验收标准

1. `tests/test_gui_data.py` 全绿且不依赖 GPU。
2. `pdm run python scripts/prepare_gui_data.py --input example-data --output <tmp>` 产出
   3 行且自检通过;对 ~100 合成任务的端到端冒烟跑通 `train.ipynb`。
3. 真实数据转换后 manifest 含全部计数与排除项,校准集满足 §9.5。
