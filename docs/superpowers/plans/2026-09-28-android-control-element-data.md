# Android Control Element Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Android Control 的 episode 目录确定性地转成 Dohnuts 决策行(action 9 类 / complete noul / element 候选 / swipe_dir 4 类 / button 4 类),产出四个 split、标号截图与审计清单,可直接喂给 `train.ipynb` 的 C 流程。

**Architecture:** 纯函数核心放 `src/dohnuts/android_control_data.py`(metadata/step 解析、动作映射、元素提取、命中测试、行派生),标号渲染放 `src/dohnuts/android_control_mark.py`,CLI 外壳放 `scripts/prepare_android_control_data.py`(episode 发现、图像存储、token 预算、隔离、manifest)。一步产 2–3 行,共享 state/group/aliases,一行一问题,复用现有 `DecisionCollator`/`Predictor`/`train.py`/`metrics.py`。规格:`docs/superpowers/specs/2026-09-28-android-control-element-data-design.md`。

**Tech Stack:** Python 3.12、PIL、transformers(AutoProcessor/smart_resize)、pytest、ruff、ty、pdm。

**状态(2026-09-28):** Task 1–6 已完成并提交;Task 7(端到端冒烟)与 Task 8(真实数据转换)尚未执行。

**约定(每个任务都适用):**

- 所有命令从仓库根目录运行(转换器会强制校验 `Path.cwd() == git rev-parse --show-toplevel`)。
- 本仓库工作区**已有他人暂存的改动**。每次提交必须显式给出路径:
  `git add <文件> && git commit -m "<msg>" -- <同一批路径>`。**不要**用不带路径的
  `git commit`,那会连同暂存区里无关的改动一起提交。
- 代码风格:ruff 行宽 100;提交前跑 `pdm run format && pdm run lint`;`src/` 内代码还要过
  `pdm run typecheck`。
- 测试:`pdm run test`(即 pytest,从仓库根运行,不加载模型权重)。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `src/dohnuts/android_control_data.py`(新建) | 词表与规则常量、metadata 校验(`parse_metadata`)、动作映射(`map_action`)、元素提取与有效性(`extract_elements`/`validate_element`/`element_bounds`)、命中测试(`hit_test`/`resolve_element_choice`)、prompt 条目(`element_description`)、step 解析(`parse_step`)、state 模板(`task_progress`)、行派生(`rows_for_ac_step`) |
| `src/dohnuts/android_control_mark.py`(新建) | `mark_screenshot`:纯 PIL 的 set-of-mark 渲染(绿框 + 白底序号 chip,丢源图 metadata) |
| `scripts/prepare_android_control_data.py`(新建) | CLI:episode 发现、metadata 读取与哈希、原图/标号图内容寻址存储、token 预算、`isolate` → `validate_rows` → 四 split + `excluded.jsonl` + `manifest.json` |
| `tests/test_android_control_data.py`(新建) | 解析、映射、元素规则、5 族行、CLI 集成测试(合成 episode fixture,不依赖被 gitignore 的 `example-data/`) |
| `tests/test_android_control_mark.py`(新建) | 标号渲染的像素级与字节级确定性测试 |
| `src/dohnuts/metrics.py`(改) | macro-F1 抑制集合加入 `"screenshot_choice"` |
| `src/dohnuts/gui_data.py`(改) | 自检按期去重解码;`validate_rows` 的 `resolve()` 失败也报可读性错误(等价收紧) |
| `docs/data-and-evaluation.md`(修改) | 追加 "Android Control conversion" 小节 |
| `docs/superpowers/specs/2026-09-28-android-control-element-data-design.md`、`docs/superpowers/plans/2026-09-28-android-control-element-data.md`(新建) | 本规格与本计划 |

提交映射(12 个实现提交,每个任务后附对应 SHA):Task 1 `f203cf1` `21c1c59`;
Task 2 `d82faed` `6b79f5d` `014de0b`;Task 3 `90b11a1` `2715bc3` `6fc9312`;
Task 4 `5accff3` `778c39d`;Task 5 `c528e5f` `307f9e1`;Task 6 文档提交(见下)。

---

## Task 1: 解析与元素规则

- [x] **Status:** 完成 —— `f203cf1`(解析与元素规则)、`21c1c59`(命中测试返回位置、never-raise 收紧)

**Files:** `src/dohnuts/android_control_data.py`(新建)、`tests/test_android_control_data.py`(新建)

**内容:** `AC_ACTIONS`(9 类,前 8 项与 `gui_data.ACTIONS` 同序同义)、`SCROLL_TO_SWIPE_DIRECTION` 取反常量、`map_action`(click/long_press/scroll/input_text/wait/open_app/navigate_back/navigate_home;类型不符返回 None)、`read_bounds`/`integer_field`/`read_text`(proto3 省略默认值)、`validate_element`、`extract_elements`(仅 clickable + 有效,连续编号)、`element_bounds`、`hit_test`(面积最小、平局取靠前、闭区间)、`resolve_element_choice`(2–128 与三种拒绝原因)、`element_description`(仅 text/content_description)。

**Verification:** 单元测试覆盖动作映射(含取反)、元素提取与编号(非 clickable 不占编号)、有效性、命中测试(平局、无命中、非有限坐标、非 list)、候选数边界、payload 字段白名单与非 ASCII;`pdm run test` 通过。

---

## Task 2: 五族行派生与 state 模板

- [x] **Status:** 完成 —— `d82faed`(五族行)、`6b79f5d`(step 结构校验、延后图像解码)、`014de0b`(图像校验限制与 step schema 文档)

**Files:** `src/dohnuts/android_control_data.py`、`tests/test_android_control_data.py`

**内容:** `validate_step`/`parse_metadata`(结构检查;`action` 键缺失不得被误读为末步)、`read_screenshot`(只校验容器头,不做全解码)、`read_a11y`、`task_progress`(已完成步模板,跳过缺失条目并重编号)、`parse_step`(整步解析与排除原因)、`rows_for_ac_step`(action/complete + element 或 button 或 swipe_dir;双 alias;element 行指向标号图)、`reference`(thought 空、ac_action、element_position)。

**Verification:** 5 族 row 形状与 id/group/split/target 对齐词表;末步产 terminate + complete=true;state 模板第 0 步为空历史、只含已完成步;`validate_rows` 对 AC 行通过;整步排除用例(缺图、截断图、缺 a11y、无命中、候选过多、未知动作、坏 instruction)逐条可解释。

---

## Task 3: 标号截图

- [x] **Status:** 完成 —— `90b11a1`(set-of-mark 渲染)、`2715bc3`(丢源图 metadata、拒绝畸形 bounds)、`6fc9312`(极大整数 bounds 加固,收紧 Task 1 的 `element_bounds`)

**Files:** `src/dohnuts/android_control_mark.py`(新建)、`src/dohnuts/android_control_data.py`、`tests/test_android_control_mark.py`(新建)、`tests/test_android_control_data.py`

**内容:** `mark_screenshot` 返回新 RGB 副本,`Image.info` 清空,按候选位置画绿框与白底黑字 chip,字号 `max(12, height // 86)`;`element_bounds` 统一拒绝缺失/畸形/非有限/反转/超大整数 bbox,标号与命中测试共用同一判定。

**Verification:** 标号渲染的字节确定性、像素存在性(绿框与 chip 颜色字面值)、三位数标签、越界裁剪、空元素列表、畸形 bounds 跳过但保留序号;`mark_screenshot` 不修改入参图像。`tests/test_android_control_mark.py` 最终 12 个用例。

---

## Task 4: CLI 转换器与 metrics 抑制

- [x] **Status:** 完成 —— `5accff3`(CLI + metrics 抑制)、`778c39d`(自检解码去重、manifest 收紧)

**Files:** `scripts/prepare_android_control_data.py`(新建)、`src/dohnuts/metrics.py`、`src/dohnuts/gui_data.py`、`tests/test_android_control_data.py`、`tests/test_metrics.py`

**内容:** episode 目录按数值序发现;`metadata_{name}.json` 读取(读不出 → `parse:unparsable_metadata`,并记录目录名与 `episode_id` 不一致);原图与标号图按内容寻址写入(写文件只发生在行已铸出之后);`isolate` → `validate_rows` → 四 split + `excluded.jsonl` + `manifest.json`;`element_stats`(post_isolation)/`element_resolution`(pre_isolation)/`environment`/`vocabularies.element_rule`/`marked_images`;`metrics.py` 的 macro-F1 抑制集合加 `screenshot_choice`。

**Verification:** manifest 全字段与行内容一致;同 `--output` 重跑 sha256 相同;磁盘镜像文件与 `manifest["images"]` 一致;输入目录下没有 episode 目录时 `SystemExit`;非仓库根运行 `SystemExit`;`validate_rows` 改按期去重后,150 episode 的真实小样自检耗时从 267 s 降到约一半(130 s 的重复解码被消除)。

---

## Task 5: token 预算接线

- [x] **Status:** 完成 —— `c528e5f`(预算检查)、`307f9e1`(存储镜像不变量、可读性失败契约)

**Files:** `scripts/prepare_android_control_data.py`、`src/dohnuts/gui_data.py`、`tests/test_android_control_data.py`、`tests/test_gui_data.py`

**内容:** `--model`/`--no-token-check`;`token_length` 与训练 collator 同式(渲染文本 + `smart_resize` 图像占位符,`IMAGE_PIXELS`,`MAX_LENGTH=2048`);超预算整步排除(`detail` = 该步最长行);预算探针在写任何文件之前运行,且用与最终行同形的 row。

**Verification:** stub 处理器钉住算术形状;真实 `Qwen/Qwen3.5-0.8B` 处理器与 `DecisionCollator` 逐行一致(本地快照存在时);全语料探针 49,924 元素行中 520 行(1.04%)超预算,最大 38,642 tokens,中位数 624、p99 2,070,探针开销约 6.0 分钟;60 个真实 episode 的开/关预算 A/B 产出行数一致(1048/1048,该样本无超预算步),说明预算不改变其余行的行集。

---

## Task 6: 文档小节与仓库内规格/计划

- [x] **Status:** 完成 —— 本次提交(文档提交)

**Files:** `docs/data-and-evaluation.md`、`docs/superpowers/specs/2026-09-28-android-control-element-data-design.md`、`docs/superpowers/plans/2026-09-28-android-control-element-data.md`

**内容:** `docs/data-and-evaluation.md` 追加 "Android Control conversion"(5 族表、动作映射表、元素规则、标号图、state、token 预算、排除、manifest、metrics、独立输出目录、已知限制、结尾运行示例);补齐仓库内缺失的设计规格与执行计划(本文件)。

**Verification:** `pdm run docs`(sphinx `-W`)退出码 0。

---

## Task 7: 端到端冒烟(需要 GPU 与本地 Qwen3.5-0.8B)

- [ ] **Status:** 待执行

**计划内容:** 合成 ~100 个 episode(每个 episode 一个目录,metadata + 每步截图 + a11y JSON,截图字节各不相同),转换后喂给 `train.ipynb` 的 CLI 流程跑通 train → calibrate → evaluate → predict;四个 split 均非空,`screenshot_choice` 在 calibration/dev/test 各有行。

**已完成的等价款:** 对 23 个**真实** episode 的实跑(仅转换,未训练):384 行、206 张图(143 原图 + 63 标号图)、`token_check: enabled`、排除 2 条(`parse:token_budget` 1、`parse:too_few_candidates` 1)、GT 命中率 0.984(63/64)。产物在 `/tmp`,未进仓库。

**待做:** 训练/校准/评估/推理四步冒烟,以及把 `train.ipynb` 的 `DATA_DIR` 指向新数据后的端到端确认。

---

## Task 8: 真实数据转换与交付检查

- [ ] **Status:** 待执行

**计划内容:**

```bash
pdm run python scripts/prepare_android_control_data.py \
  --input example-data/android_control_parsered/parsered \
  --output data/processed/ac-v1 --model Qwen/Qwen3.5-0.8B
```

**检查清单(执行后逐项确认):**

1. 四 split 非空,`screenshot_choice` 在各 split 有行;calibration 的 choice 与 noul 各 ≥10(否则温度保持 1.0)。
2. `no_target_element`/`too_few_candidates` 排除率若 >10% 需复查命中规则。
3. `action_classes` 各 split 分布相近,`open_app` 出现在 9 类分布里。
4. `element_stats` 候选数均值与探索一致(约 12 起,真实小样 21.8);`empty_target_payload_rate` 与探针的 44–49% 相符。
5. `token_check: enabled`,`parse:token_budget` 逐条可解释(全语料预计约 520 条,与探针一致)。
6. 磁盘:输出预计 ~40–50 GB(99k 原图 + 52k 标号图),先 `df -h` 确认。
7. 转换后不要再改 `data/processed/ac-v1` 下的 jsonl(`train.py` 会比对 SHA-256)。

---

## 验收状态

| 条目 | 状态 | 证据 |
| --- | --- | --- |
| `pdm run test` 全绿、不依赖 GPU | 通过 | 193 passed(其中 AC 两个文件 105 个用例) |
| `pdm run check` | 通过(被跟踪文件) | `pdm run format-check`、`pdm run typecheck` 退出 0;`pdm run lint` 仅报未跟踪的 `train.ipynb` 自带 F541 |
| 转换确定性 | 通过 | 同 `--input`/`--output` 重跑 manifest 与四文件 sha256 不变;标号图字节级确定性测试 |
| token 估算与训练 collator 一致 | 通过 | 真实处理器逐行一致性用例(本地快照存在时运行) |
| 真实小样转换 | 通过 | 23 episode → 384 行 / 206 图 / 2 排除 / 命中率 0.984 |
| 端到端冒烟(Task 7) | 待执行 | 计划位于本文件 Task 7 |
| 全语料转换与检查单(Task 8) | 待执行 | 计划位于本文件 Task 8 |
