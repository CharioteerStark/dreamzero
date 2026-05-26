# 两阶段微调数据采集需求(Stage A 具身适配 / Stage B 未见任务)

> 本文档说明 **遥操作数据采集团队** 在两个微调阶段分别需要采集**什么数据**、**采多少**、**怎样标注**。两阶段共同遵循的 LeRobot v2 交付格式参见 [`DATA_COLLECTION_REQUIREMENTS_zh.md`](DATA_COLLECTION_REQUIREMENTS_zh.md);本文件只描述**内容层面**的差异。
>
> 训练侧总体流程见 [`STAGE_A_PLAN.md`](STAGE_A_PLAN.md)。

---

## 1. 概览:两个阶段为什么需要不同的数据

| 维度 | Stage A:具身适配 | Stage B:未见任务专化 |
|---|---|---|
| 训练目的 | 让模型学会**这台机器人怎么动**、语言怎么映射到本体动作空间 | 让模型学会**这台机器人怎么完成具体的新任务** |
| 论文对应实验 | AgiBot G1 → YAM 少样本具身迁移(§5 Q5, Fig 12) | AgiBot 上的三项下游任务后训练(§4.2, Fig 10) |
| 总数据量 | **30–60 分钟**(论文 30 分钟,55 条轨迹,11 个任务) | **每个任务 10–40 小时**(论文:折衬衫 33 h、水果打包 12 h、清桌 40 h) |
| 任务种类 | **10–12 种短动作原语**,广而浅 | **1–N 个目标任务**,深而专 |
| 每条轨迹长度 | 短(≈ 20–40 s) | 中-长,允许多阶段 |
| 物体与场景多样性 | 中等多样,重点是动作原语覆盖 | **极高**多样性,每条 episode 都要在物体种类、组合、位姿、布局上变化 |
| 语言标注 | 每个任务 ≥ 2 条同义改写,以丰富语言条件 | 每条 episode 一句清晰任务描述;多阶段任务可分段标注 |
| 失败轨迹 | 仅采集成功轨迹 | 仅采集成功轨迹 |
| 成功定义 | 不强求"任务完成",动作可识别即可 | 必须达成任务目标(可分级:`shirt_stages_done / 5`、`fruits_packed / 10`) |

**核心区别:** Stage A 用很少的数据让模型把"语言 → 本体动作"打通;Stage B 用大量、多样化的数据让模型把"特定任务"做扎实。**两个阶段使用同一台机器人、同一组相机、同一份关节布局,中间不允许任何硬件改动**。

---

## 2. Stage A 数据规范 — 具身适配(约 30–60 分钟)

### 2.1 整体要求

| 项 | 目标值 |
|---|---|
| 总时长(去除 idle 帧后) | ≥ **30 分钟**;建议采 **45–60 分钟** 留余量 |
| episode 总数 | 约 **50–80 条** |
| 任务种类 | **10–12 种**短动作原语 |
| 每种任务轨迹数 | ≥ **4–6 条**(论文均值 ≈ 5) |
| 单条 episode 时长 | **20–40 秒**,严禁长时间静止或 idle 段 |
| 帧率 | 20–30 Hz(以机器人原生发布率为准,所有 episode 必须一致) |
| 状态/动作维度 | **14-D**(双臂 + 双夹爪);具体切分见 §4 |
| 相机 | **3 路**:`top`(俯视)、`left_wrist`(左腕)、`right_wrist`(右腕);外参固定 |
| 语言 | 英文,每个任务 **2–3 条同义改写**,见 §2.4 |

> ⚠️ **idle 帧处理:** 论文预训练明确"filter out idle actions"。采集端可以原样保留,但**必须在数据卡中说明 idle 段的范围**(如:每条 episode 开头 1 s 静止),便于训练侧过滤。理想情况下,采集者在开始动作前才按"开始录制"。

### 2.2 任务设计指南

Stage A 的任务要覆盖以下**动作原语类别**,尽量每类至少 1 个:

| 原语类别 | 描述 | 论文 YAM 中的对应任务 |
|---|---|---|
| 单臂抓取-放置(小件) | 抓小物 → 放到目标点 | Cube to pad |
| 单臂抓取-放置(长件) | 抓长条物 → 放到目标位置(姿态有约束) | Bar to pad |
| 单臂放入容器/支架 | 抓 → 放入有几何约束的目标(架、孔、槽) | Bar to rack |
| 精确插入 | 抓 → 对位 → 插入 | Pipe insertion |
| 双臂交接 | 一只手抓 → 移到中间 → 另一只手接 → 放 | Hand off cube |
| 双臂协作折叠/操作软体 | 双臂协同操控柔性物 | Towel fold / T-shirt fold |
| 多步堆叠 | 按顺序逐个抓-放,有顺序依赖 | Stack bowls |
| 多实例处理 | 容器内多个物体逐个搬运 | Dish to dish rack |
| 语义匹配(条件放置) | 根据物体特征选择正确目标 | Mail to mailbox |
| 多阶段组合 | 抓-放-关闭-再放,含状态切换 | Candy to box |

**建议:** 如果你的机器人形态接近 YAM(双臂、平行夹爪、3 相机),直接**复刻**论文这 11 个任务最稳。任务定义见 §2.3。

### 2.3 论文 YAM 11 个参考任务(可直接复刻)

下表来自论文官方数据可视化页 `https://dreamzero0.github.io/yam_gallery/`,**语言指令逐字引用**。如果直接复刻,采集团队请优先使用下列英文原句作为 `annotation.task` 文本。

| # | 任务名 | 英文指令(逐字) | 中文含义(仅供采集者理解,**不要写入标注**) | 涉及双臂? | 物体 / 道具 |
|---|---|---|---|---|---|
| 1 | Pipe insertion | `Pick up the pipe and insert it into the slot.` | 抓起管子并插入插槽 | 单臂 | 管 + 插槽板 |
| 2 | Cube to pad | `Pick up the black cube and place it on the red halo pad.` | 抓起黑色立方块,放到红色环形垫上 | 单臂 | 黑色立方块 + 红色环形垫 |
| 3 | Bar to pad | `Pick up the black bar and place it on the red horizontal pad.` | 抓起黑色长条,放到红色横向垫上 | 单臂 | 黑色长条 + 红色横向垫 |
| 4 | Bar to rack | `Move the green bar to rack` | 把绿色长条移到架子上 | 单臂 | 绿色长条 + 支架 |
| 5 | Hand off cube | `Hand off the cube from left to right hand and place it on the red pad.` | 立方块从左手交到右手并放到红色垫上 | **双臂协作** | 立方块 + 红色垫 |
| 6 | Stack bowls | `Stack the bowls in consecutive sizes on top of the largest bowl.` | 按尺寸依次将碗叠到最大的碗上 | 单臂(可双臂) | 多个不同尺寸的碗 |
| 7 | Towel fold | `With both grippers fold the towel from bottom to top, then with the left gripper fold the towel from left to right.` | 双夹爪先把毛巾从下向上折,再用左夹爪从左向右折 | **双臂协作** | 毛巾 |
| 8 | T-shirt fold | `Fold the t-shirt.` | 把 T 恤折好 | **双臂协作** | T 恤 |
| 9 | Dish to dish rack | `Take the dishes out of the basket and place them in the dish rack.` | 把篮子里的盘子逐一取出并放入碗碟架 | 单臂 | 篮子 + 多个盘子 + 碗碟架 |
| 10 | Mail to mailbox | `Put the mail into its corresponding mailbox.` | 把信件放入对应的邮箱(按标识/颜色匹配) | 单臂 | 多封信 + 多个有标识的邮箱 |
| 11 | Candy to box | `Place the candy into the box and close it, then place the box into the tray matching the candy's color.` | 把糖放入盒中并合盖,再把盒子放入与糖颜色一致的托盘 | 单臂(可双臂) | 糖果 + 带盖盒子 + 多个彩色托盘 |

**轨迹分配建议(总计 ≈ 55 条,与论文一致):**

```
Pipe insertion        5 条
Cube to pad           5 条
Bar to pad            5 条
Bar to rack           5 条
Hand off cube         5 条
Stack bowls           5 条
Towel fold            5 条
T-shirt fold          5 条
Dish to dish rack     5 条
Mail to mailbox       5 条
Candy to box          5 条
────────────────────────
合计                  55 条
```

每个任务内部要保持物体位姿、初始角度、抓取点的合理变化,**不要 55 条全部从同一个起始位姿开始**。

### 2.4 语言标注示例(英文,每任务 2–3 条同义改写)

**论文明确指出**(footnote 12, p.17):YAM 实验仅使用每任务一条全局标注,这是该实验的一个限制。**我们要主动改进:每个任务准备 2–3 条同义改写**,在 5 条轨迹里随机分配。

| 任务 | 主指令(论文原句) | 同义改写示例 |
|---|---|---|
| Pipe insertion | `Pick up the pipe and insert it into the slot.` | `Grab the pipe and slot it into the hole.` / `Insert the pipe into the slot.` |
| Cube to pad | `Pick up the black cube and place it on the red halo pad.` | `Move the black cube onto the red ring pad.` / `Put the black block on the red halo mat.` |
| Hand off cube | `Hand off the cube from left to right hand and place it on the red pad.` | `Pass the cube from the left arm to the right arm and set it on the red pad.` |
| Towel fold | `With both grippers fold the towel from bottom to top, then with the left gripper fold the towel from left to right.` | `Fold the towel up first, then fold it across to the right.` |
| ... | ... | ... |

**严格要求:**
- 所有标注 **必须为英文**,禁止任何中文/拼音/混写(详见 [`DATA_COLLECTION_REQUIREMENTS_zh.md`](DATA_COLLECTION_REQUIREMENTS_zh.md) §6)。
- 同一条 episode 内 `annotation.task` 文本保持不变(整段统一)。
- 不要使用 `task1`、`do the thing`、空字符串等占位符。

#### 2.4.1 论文 AgiBot 的"同任务 + 不同物体"写法参考

论文 Table 5(AgiBot Seen Tasks Evaluation Setup, p.26)对每个任务给了 **4 条不同物体/位置的真实指令**,**这是最值得仿照的范式** —— 不是单纯改用词,而是**保持句式 + 替换具体物体/颜色/目标容器**,让模型学到"动作模板 + 物体绑定"。

**示例 A — "拿水果放容器" 类任务的 4 种写法(逐字引用):**

| # | 指令 |
|---|---|
| 1 | `The left arm picks up the banana on the table and places it on the Blue Plate.` |
| 2 | `The left arm picks up the lime on the table and places it on the light green plate.` |
| 3 | `The left arm picks up the peach from the plastic bag and places it on the baking pan.` |
| 4 | `The left arm picks up the green pear on the table and places it in the blue checkered bowl.` |

**示例 B — "擦桌" 类任务的 4 种写法(逐字引用,展示工具替换):**

| # | 指令 |
|---|---|
| 1 | `The left arm uses a sponge to wipe the coffee spill off the table.` |
| 2 | `The left arm used a black cloth to wipe the white powder off the table.` |
| 3 | `The left arm uses a napkin to wipe the orange spill off the table.` |
| 4 | `The left arm uses a paper towel to wipe the water off the table.` |

**示例 C — "把笔放进笔筒" 类任务的 4 种写法(逐字引用,展示笔的颜色/材质替换):**

| # | 指令 |
|---|---|
| 1 | `The left arm picks up the Red Marker pen from the table and placed it into the pen holder.` |
| 2 | `The left arm picked up the black marker from the table and placed it into the pen holder.` |
| 3 | `The left arm picked up the white marker from the table and placed it into the pen holder.` |
| 4 | `The left arm picked up the mechanical pencil from the table and placed it into the pen holder.` |

**关键观察:**
1. **句式高度统一**:`The left/right arm + 动作 + 物体描述 + and + 放置动作 + 目标` —— 不要乱换语序。
2. **替换的是物体描述**:颜色(red / black / green pear)、材质(sponge / cloth / napkin / paper towel)、容器(Blue Plate / wooden basket / baking pan)。
3. **冠词、命名一致**:每个具体物体在多条 episode 里反复出现时,英文名保持一致(`pen holder` 不要混写成 `pencil holder`)。
4. **首字母大小写不严格**:论文里 `Blue Plate`、`blue checkered bowl`、`Wooden Basket` 都出现过 —— 不必死磕统一,但同一物体的写法尽量稳定。

> 💡 **对你 Stage A 11 个任务的建议:** 复刻 YAM 的 11 个任务时,每个任务**至少**准备 **4 条** 形如示例 A/B/C 的指令(论文 AgiBot 是 4 条,YAM 只 1 条 —— 我们至少要做到 AgiBot 的密度)。然后把 5 条轨迹里的指令在这 4 条之间随机分配。

> 📌 **关于 subtask 标注:不需要做。** 按论文做法,每条 episode **一句 global 英文 instruction 就够了**,**不需要 per-frame / per-subtask 切分标注**:
> - YAM 30 min finetune 数据明确只用了 "11 short, global language annotations unique for each task"(论文 p.17 footnote 12),即 11 个任务共 11 条 string,每条 episode 内 instruction 不变。
> - 论文里唯一有 per-frame subtask 的是 **AgiBot 预训练数据**(42 subtasks / episode),且这些 subtask 段来自 AgiBot 自家流水线的 `label_info.action_config` 字段,不是 teleoperator 手工逐段标注的。
> - 仓库里 fine-tune 用的 converter(`scripts/data/convert_lerobot_to_gear.py`)只接受 **一个** `--task-key` 字段,**不支持** 多列 subtask 输入。
> 
> 想超过论文?可以让同一任务多准备几条**整段同义改写**(2–3 条),在不同 episode 间随机分配(本节 §2.4 已要求)。**不要**尝试在 episode 内部切 subtask —— 这会同时增加采集成本和数据团队/训练侧的兼容性风险。

### 2.5 Stage A 交付检查项

- [ ] 总时长(去除 idle 后)≥ 30 min
- [ ] 任务数 ≥ 10
- [ ] 每个任务 ≥ 4 条轨迹
- [ ] 每个任务有 ≥ 2 条不同语言改写
- [ ] 物体识别与目标位置在英文指令中都明确出现
- [ ] 3 路相机视频与 parquet 帧数严格对齐
- [ ] 14-D 状态/动作切分文档随数据交付
- [ ] 无 NaN / Inf,纯 ASCII 标注

---

## 3. Stage B 数据规范 — 未见任务专化(每任务 10–40 小时)

### 3.1 整体要求

| 项 | 目标值 |
|---|---|
| 每个任务总时长 | 论文:**折衬衫 33 h、水果打包 12 h、清桌 40 h**。新动作类(折叠、插入、多阶段)按 **10–40 h** 规划;若仅是 Stage A 原语的物体/位姿变体,可降到 **1–3 h** |
| 每个任务 episode 数 | 视任务长度而定(论文 episode 平均 4.4 min;短任务可 1–2 min,长任务 5–10 min) |
| 任务数 | 1 个或多个,**每个任务独立交付一份数据集** |
| 帧率、相机、状态/动作维度 | **与 Stage A 完全一致**(不允许任何变动) |
| 语言 | 每条 episode **一句**清晰英文指令(即使是多阶段任务,也写成一段完整 paragraph,不在 stage 边界换文本) |

> 📌 **关于多阶段任务的语言写法 — 重要,容易踩坑。** 即使任务有明确的 stage 切分(折衬衫 5 步、糖果盒"放糖 → 合盖 → 按色匹配"),论文做法是 **把整个流程压成一段完整 paragraph 作为单条 instruction**,**不**在 stage 边界换 string。
> 
> 参考论文 Table 5 row 8 折衬衫(原文):
> > *"Both arms grip the bottom of the light grey shirt sleeves and fold it toward the middle. They then pull the short sleeves across the table to the edge. Next, both arms grasp the top of the shirt and fold it downwards to the middle. Finally, the right arm grasps the collar and folds it down to complete the task."*
> 
> 以及 Table 6 row 5 Cube Stacking:把"抓绿块 → 叠 → 抓蓝块 → 叠 → 抓橙块 → 叠"4 步过程也合成一段 paragraph。
> 
> **采集团队要做的:**
> - 每条 episode 的 `annotation.task` 是 **一整段** 描述全过程的英文 paragraph,整段都填同一字符串,逐帧不变。
> - 同一任务可准备 2–3 条 paragraph 同义改写(整段不同写法,而不是切片),在不同 episode 间随机使用。
> - **不要**:在 episode 中间换 string;**不要**:把多阶段拆成多列 `annotation.stage_1` / `annotation.stage_2`;**不要**:每帧写不同的 action_text。

> 🔒 **关键约束:** Stage B 的硬件配置(相机外参、相机顺序、关节布局、控制频率)必须与 Stage A **完全一致**,Stage A 结束后**锁定工作单元**,直到 Stage B 全部采完。任何漂移都会让 Stage A 训出的具身嵌入失效。

### 3.2 多样性要求(比 Stage A 严格得多)

论文后训练的核心做法是**每条 episode 都随机化**:

| 任务 | 论文随机化策略 |
|---|---|
| Shirt folding (33 h) | 2 种衬衫,衬衫初始位置随机 |
| Fruit packing (12 h) | 水果组合随机、水果与袋子位置随机 |
| Table bussing (40 h) | 5 种垃圾 + 5 种餐具(碟、碗、叉、勺),物体种类、组合、位置随机 |

对你交付的每个 Stage B 任务,**至少**要做到:

| 维度 | 要求 |
|---|---|
| 物体种类 | ≥ 3 种(同类不同实例算多种,如不同颜色/形状的同类物) |
| 物体数量 | 若任务涉及多物体,数量也要变(如 3–5 件) |
| 初始位姿 | 在桌面合理范围内**每条 episode 都不同**,不要近似复制 |
| 场景布局 | 周边干扰物、桌面遮挡、目标容器位置都要变化 |
| 光照 | 若条件允许,在不同时段或不同灯光下采集若干批 |
| 抓取/接触点 | 同一物体可以用不同抓取点完成 |

**反面例子(请避免):** 100 条 episode 全部用同一个红色立方块从同一个起点抓到同一个目标点。这种重复数据论文 §5 Q1 已经证明会让 generalization 大幅退化(33% → 50% 当切换到多样数据后)。

### 3.3 论文多阶段任务 paragraph instruction 实例(摘自 Table 5/6,逐字引用)

下列指令都是论文里**真实使用过的** —— 注意每条都是一整段 paragraph,描述从开始到完成的全部步骤,**整条 episode 内逐帧不变**。

**示例 D — Folding Shirts(Table 5 row 8,seen tasks):同一任务 4 种颜色衣物的 paragraph:**

| # | Paragraph instruction |
|---|---|
| 1 | `Both arms grip the bottom of the light grey shirt sleeves and fold it toward the middle. They then pull the short sleeves across the table to the edge. Next, both arms grasp the top of the shirt and fold it downwards to the middle. Finally, the right arm grasps the collar and folds it down to complete the task.` |
| 2 | `Both arms hold the bottom of the gray short sleeves to the middle. They then pull the short sleeves of the gray shirt to the edge of the table. Next, both arms grip the top of the gray shirt sleeve and fold it down to complete the task.` |
| 3 | `Both arms hold the bottom of the gray shirt sleeves to the middle. Then they pull the short sleeves of the checkered shirt, move it over the bowl, and ... ` *(论文中长 paragraph,省略)* |
| 4 | `Both arms hold the bottom of the white shirt sleeves to the middle. They then pull the short sleeves to the edge of the table. Next, the right arm grasps the top of the shirt sleeve and folds it down to complete the task.` |

**示例 E — Cube Stacking(Table 6 row 5,unseen tasks):3 步堆叠的 paragraph:**

> `The robot reaches its right arm to pick up the green cube, moves it over the red cube, and releases it to stack. It then reaches its left arm to pick up the blue cube, moves it over the stack, and releases it onto the stack to finish the task. It then reaches the left arm to pick up the orange cube, moves it over the stack, and releases it onto the stack to finish the three-tier structure.`

**示例 F — Stacking Clothes(Table 5 row 10):简短的"放叠"动作,仍然是单条 paragraph:**

> `Both arms pick up the black shirt and place it on the stack of clothes.`

**示例 G — Pulling Cart(Table 6 row 10):动作简单时,paragraph 也可以很短:**

> `The robot reaches its left arm to grab the cart and pulls it.`

**示例 H — Painting(Table 6 row 6):双臂协作的两步动作,合写一句:**

> `The left arm grabs the brush. Then right arm paints with the brush on the notebook.`

**写作规律(从论文真实样本中归纳):**

1. **多步任务用"逻辑连接词"串起来**:`Then` / `Next` / `Finally` / `It then` / `and then` / `before`。**严禁**用句号 + 单独编号(❌ `1. ... 2. ...`)。
2. **每个步骤一句话**:动作 + 物体 + 目标,不要堆形容词。
3. **始终从机器人视角第三人称写**:`The robot ...` / `The left arm ...` / `Both arms ...`。
4. **结尾用 "to complete the task" / "to finish the task" / "to finish the three-tier structure" 收束** —— 给模型一个明确的"任务结束"信号。
5. **同一任务的 N 个 paragraph 写法之间,保持 step 数量与顺序一致**,只替换物体/颜色/容器(对照示例 D 的 4 条版本)。

### 3.4 任务定义模板(交付时随数据填写)

每个 Stage B 任务交付时**必须**提供一个任务卡(`<task_name>/TASK_CARD.md`):

```markdown
# Task: <task_name>

## 任务描述
<一段中文 + 英文双语描述:机器人需要做什么、什么算成功>

## 成功判据(用于评估)
- <可量化的成功指标,如 "5 个水果中 4 个进袋">
- <部分完成定义,如 "0–1 之间的部分完成度">

## 物体清单
| 物体 | 数量范围 | 备注 |
|---|---|---|
| ... | ... | ... |

## 随机化维度
- [ ] 物体身份(列出可选物体)
- [ ] 物体位姿(列出可变范围,如 "桌面 40×30 cm 区域内")
- [ ] 容器/目标位姿
- [ ] 其他(光照、干扰物、初始关节角)

## 语言指令(英文)
- 主指令(一整段 paragraph,包含全部步骤):<英文原句>
- 同义改写(2–3 条,整段不同写法):
  - <改写 1>
  - <改写 2>

## 阶段说明(仅供采集者理解 / 评估打分用,**不进入** annotation 标注)
> 多阶段任务在数据集中是**单条 paragraph instruction**,不在 stage 边界切换文本(见 §3.1)。
> 此处的阶段切分只用于:① 采集者理解流程;② 评估时按阶段打部分完成度。

| 阶段 | 描述 | 评估打分点 |
|---|---|---|
| 1 | ... | ... |
```

### 3.5 Stage B 交付检查项

- [ ] 每个任务有独立的目录与 `TASK_CARD.md`
- [ ] 每个任务总时长达到承诺值
- [ ] 物体、位姿、布局的随机化在 episode 间清晰可见(抽 10 条对比即可看出)
- [ ] 所有 episode 都为**成功轨迹**(失败的不要混入)
- [ ] 多阶段任务:`annotation.task` 是**单条整段 paragraph**(同一 episode 内逐帧不变),阶段切分**仅**写在 `TASK_CARD.md` 里供评估使用
- [ ] 相机外参、关节布局、控制频率与 Stage A 完全一致(在数据卡中签字确认)
- [ ] 所有标注英文 ASCII,无空字符串

---

## 4. 两阶段共享格式(参考)

LeRobot v2 目录结构、parquet 列要求、`meta/info.json` 字段、`annotation.task` 命名约定等所有**格式层面**要求,见已发布的 [`DATA_COLLECTION_REQUIREMENTS_zh.md`](DATA_COLLECTION_REQUIREMENTS_zh.md)。本文件 **不** 重复这些细节;采集团队应将本文件视为对该规范在两个阶段下的**内容补充**。

简要提醒:

- 目录:`data/chunk-XXX/episode_NNNNNN.parquet` + `videos/chunk-XXX/observation.images.<cam>/episode_NNNNNN.mp4` + `meta/info.json`。
- 必需列:`observation.state`、`action`、`timestamp`、`frame_index`、`episode_index`、`index`、`annotation.task`。
- `observation.state` / `action` 为 14-D `float32`,无 NaN/Inf。
- 视频与 parquet 行数严格相等,FPS 与 `info.json` 一致。
- 标注必须为英文,采用 `动词 + 对象 + 目标` 句式。

---

## 5. 交付目录建议

```
deliveries/
├── stage_a/
│   └── myrobot_stage_a_lerobot/              # ≈ 55 条 episode,10–12 个任务
│       ├── data/chunk-000/episode_*.parquet
│       ├── videos/chunk-000/observation.images.{cam_top,cam_left,cam_right}/episode_*.mp4
│       ├── meta/info.json
│       └── DATA_CARD.md                       # 列出 11 个任务及其语言改写
└── stage_b/
    ├── <task_1>/
    │   ├── data/...
    │   ├── videos/...
    │   ├── meta/info.json
    │   ├── TASK_CARD.md
    │   └── DATA_CARD.md
    ├── <task_2>/
    │   └── ...
    └── ...
```

每份数据交付时,同步提供一份 `DATA_CARD.md`,记录:采集者、采集日期、机器人型号、控制频率(Hz)、坐标系/单位、动作是否绝对、idle 帧位置、相机外参摘要、是否包含失败轨迹。
