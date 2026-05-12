# 数据采集格式需求文档

> 本文档定义了用于训练 DreamZero 策略的机器人数据采集**交付格式**。数据团队只需按本文档要求把数据组织为 **LeRobot v2** 格式并交付;后续的格式转换、训练侧元数据生成、训练脚本接入,均由项目维护方统一完成,采集端无需处理。

---

## 1. 总体要求

- **数据集格式**:必须遵循 **LeRobot v2** 数据集规范。
- **存储后端**:Parquet(状态/动作)+ MP4(视频)+ JSON/JSONL(元数据)。
- **同步性**:同一时间步(timestep)内,所有相机帧、本体状态、动作指令必须严格对齐。
- **完整性**:每条 episode 必须包含完整的(观测、状态、动作、语言任务标注)信息,不允许缺帧或缺值。
- **一致性**:同一具身(embodiment)下,所有 episode 的状态/动作维度、相机数量、相机名称、FPS 必须保持一致。

---

## 2. 目录结构

数据采集完成后应组织为以下目录结构:

```
<dataset_root>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.<cam_name_0>/
│       │   ├── episode_000000.mp4
│       │   ├── episode_000001.mp4
│       │   └── ...
│       ├── observation.images.<cam_name_1>/
│       │   └── ...
│       └── observation.images.<cam_name_2>/
│           └── ...
└── meta/
    └── info.json
```

### 约束说明

| 项 | 要求 |
|---|---|
| episode 文件命名 | `episode_{6位零填充编号}.parquet` / `.mp4`,编号从 `000000` 开始连续递增 |
| episode 编号对应关系 | 同一 episode 的 parquet 与各相机 mp4 必须使用相同编号 |
| chunk 划分 | 每个 chunk 默认包含若干 episode,目录名为 `chunk-XXX` |
| 相机目录命名 | 必须以 `observation.images.` 为前缀,后接相机标识(例如 `cam0`、`top`、`left_wrist`) |

---

## 3. 视频(Video)需求

每个相机一个独立目录,每条 episode 一个 MP4 文件。

| 项 | 需求 |
|---|---|
| 容器格式 | `.mp4` |
| 编码 | H.264 (建议) |
| 帧率(FPS) | 与 parquet 数据的采样频率严格一致,常用值为 `15` / `20` / `30` |
| 分辨率 | 建议原始采集分辨率 ≥ `320 × 176`(训练时会按 `image_resolution_width/height` 缩放) |
| 长度对齐 | MP4 帧数 = parquet 行数(同一 episode 内严格相等) |
| 色彩空间 | RGB(避免 BGR,避免单通道灰度) |
| 相机数量 | 3 路(例如 `top` / `left_wrist` / `right_wrist`),与 `num_views` 训练参数对应 |
| 相机命名 | 每路相机使用稳定、可读的标识(例如 `cam0`、`top`、`wrist`),并在 episode 间保持一致 |
| 时间戳同步 | 所有相机帧与状态/动作记录通过同一时间基对齐,误差小于 1 帧 |

---

## 4. Parquet 数据列需求

每个 `episode_XXXXXX.parquet` 文件按时间步(行)存储,需包含以下列:

### 4.1 必需列

| 列名 | 类型 | 说明 |
|---|---|---|
| `observation.state` | `list[float32]` | 机器人本体状态向量(关节角、夹爪开度等) |
| `action` | `list[float32]` | 控制器下发或目标动作向量 |
| `timestamp` | `float32` | 当前帧的时间戳(秒),从 0 起算 |
| `frame_index` | `int64` | 当前 episode 内的帧索引,从 0 起算 |
| `episode_index` | `int64` | episode 编号,与文件名一致 |
| `index` | `int64` | 整个数据集的全局帧索引,从 0 起算 |

### 4.1.1 可选列(采集端可不提供)

| 列名 | 类型 | 说明 |
|---|---|---|
| `task_index` | `int64` | 任务索引;采集端可省略,由维护方在后处理阶段根据语言标注列自动生成 |

> 只需保证第 4.2 节的语言标注列存在且非空,`task_index` 是否填写不影响后续训练。

### 4.2 语言/任务标注列

至少需要一个以 `annotation.` 为前缀的字符串列,用于该帧对应的自然语言任务描述(可在一条 episode 内整段保持一致,也可分段切换)。

| 列名 | 类型 | 说明 |
|---|---|---|
| `annotation.task` | `string` | **推荐使用的默认列名**,无需额外沟通 |

如果采集栈已经在用其他列名(例如 `annotation.human.task_description`、`annotation.language.language_instruction`),可以保留原名,但**必须在交付时书面告知维护方实际使用的列名**。

> 重要:整条 episode 内不允许出现空字符串或占位文本,详见第 6 节。

### 4.3 状态/动作向量切分约定

`observation.state` 与 `action` 通常是 **打包的浮点向量**,后续训练侧需要把它切分为命名子键(例如 `joint_pos`、`gripper_pos`)。**采集端只需保证向量内部顺序在所有 episode 内一致**,并在数据卡中书面记录:

1. 每个子键对应的索引区间(start, end);
2. 子键名称与物理含义。

**示例(单臂 6 自由度 + 1 维夹爪):**

| 子键 | 维度区间 | 含义 |
|---|---|---|
| `joint_pos` | `[0, 6]` | 6 个关节位置(弧度) |
| `gripper_pos` | `[6, 7]` | 夹爪开度(归一化或物理单位) |

**示例(双臂):**

| 子键 | 维度区间 | 含义 |
|---|---|---|
| `left_arm_joint_pos` | `[0, 7]` | 左臂 7 关节 |
| `left_gripper_pos` | `[7, 8]` | 左夹爪 |
| `right_arm_joint_pos` | `[8, 15]` | 右臂 7 关节 |
| `right_gripper_pos` | `[15, 16]` | 右夹爪 |

### 4.4 数值要求

- **dtype**:`float32`(状态/动作);`int64`(索引);`string`(任务文本)。
- **单位**:同一字段在所有 episode 内必须使用同一物理单位(关节角统一弧度或度,夹爪统一同一标度)。
- **NaN / Inf**:禁止出现 `NaN` 与 `Inf`。
- **范围**:不强制归一化,但需稳定;后续训练会按 `q01/q99` 做归一化。
- **action 含义**:必须明确是 **绝对目标** 还是 **相对增量**,并在整套数据中保持一致。推荐采集**绝对动作**(absolute target),相对量的统计由维护方在后处理时计算。

---

## 5. `meta/info.json` 需求

`meta/info.json` 必须至少包含:

```json
{
  "fps": 30,
  "total_episodes": 1234,
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [7]
    },
    "action": {
      "dtype": "float32",
      "shape": [7]
    },
    "observation.images.cam0": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "info": {
        "video.fps": 30.0,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p"
      }
    },
    "annotation.task": {
      "dtype": "string"
    },
    "timestamp": {"dtype": "float32", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]}
  }
}
```

### 字段约束

| 字段 | 要求 |
|---|---|
| `fps` | 整数,数据集真实采样率;与所有 MP4 的实际 FPS 一致 |
| `total_episodes` | 与 `data/` 下 episode 数量一致 |
| `features.<key>.dtype` | 真实 dtype:`float32` / `int64` / `string` / `video` |
| `features.<key>.shape` | 状态/动作向量长度;视频为 `[H, W, C]` |
| 相机 feature 命名 | `observation.images.<cam_name>`,与 `videos/chunk-XXX/` 下目录名严格一致 |

---

## 6. 任务标注(Language)需求

- 每条 episode 至少有一条自然语言任务描述。
- **必须且只能使用英文(English-only,强制要求)**。DreamZero 的预训练 backbone(`DreamZero-AgiBot`、Wan2.1)与文本编码器(`google/umt5-xxl`)均以英文语料为主进行训练;DROID / AgiBot / YAM 等上游数据集的任务标注也全部为英文。**禁止**使用中文、中英混写、拼音、表情符号或任何非英文字符 —— 这类标注会显著降低 zero-shot / few-shot 表现,并在自检阶段被直接退回。
- 描述应清晰、具体、可复现,避免空字符串、占位符(如 `"task"`、`"todo"`)。
- 同一任务的描述在数据集内应保持文本统一(或在合理的同义改写范围内),便于后续聚合。
- 句式建议采用祈使句(动词开头),包含**动作 + 对象 + 目标位置/状态**三要素。

**推荐写法:**

| ✅ 推荐(英文,动作+对象+目标) | ❌ 避免 |
|---|---|
| `pick up the red block and place it in the blue bowl` | `task1` |
| `open the top drawer of the cabinet` | `do the thing` |
| `pour water from the bottle into the cup` | (空字符串) |
| `fold the towel in half` | `把毛巾叠起来`(中文) |
| `hand over the screwdriver to the left arm` | `Pick block.`(过于简略) |

> 标注流程要求:从采集第一帧开始就直接用英文输入;**不允许**先记录中文再翻译。交付前由一名熟悉英文表达的同事复核语法与清晰度。
>
> 自动检查建议:在交付脚本里加一行 `assert df['annotation.task'].str.match(r'^[\x00-\x7F]+$').all()`,出现任何非 ASCII 字符直接报错。

---

## 7. Episode 长度与采集策略

| 项 | 建议 |
|---|---|
| 单条 episode 长度 | ≥ `len(video delta_indices) + action_horizon`,通常 ≥ 50 帧;若使用默认 25 视频帧 + 24 动作步,则 ≥ 49 帧 |
| 数据集规模 | 单具身建议 ≥ 数百条 episode;场景/物体多样性优先于单纯数量堆积 |
| 任务覆盖 | 每个语言任务至少包含若干条成功 episode,确保模型学到该任务 |
| 失败轨迹 | 默认仅采集成功轨迹;若包含失败轨迹,应在 episode 元信息中标注 |
| 演示来源 | 遥操作 / 脚本策略 / 真人示教均可,但应在数据卡中记录来源 |

---

## 8. 具身一致性(Embodiment Consistency)

同一 `<EMBODIMENT>` 标签下的所有数据必须满足:

- 相同的关节/夹爪结构与维度;
- 相同的相机数量与命名;
- 相同的状态/动作子键切分约定;
- 相同的 FPS;
- 同一物理单位与坐标约定。

> 如更换了关节配置、相机数量,或动作语义(绝对/相对)改变,**必须**重新启用一个独立的数据集目录交付,不要把不同硬件配置的 episode 混入同一批数据。具体的具身标签由维护方决定,采集端无需关心。

---

## 9. 交付前自检清单

提交数据前请逐项确认:

- [ ] 目录结构符合第 2 节示例
- [ ] 所有 episode 的 parquet 与 mp4 编号一一对应,无缺失
- [ ] 每个 MP4 帧数 = 对应 parquet 行数
- [ ] 所有 MP4 的 FPS 与 `info.json` 中 `fps` 一致
- [ ] `observation.state` / `action` 在所有 episode 内维度一致,dtype 为 `float32`
- [ ] 状态/动作向量无 `NaN` / `Inf`
- [ ] 已明确状态/动作子键的索引切分,并以书面形式随数据交付
- [ ] `annotation.task`(或指定的 `--task-key` 列)存在,且每条 episode 都有非空、清晰的**英文**任务描述(不允许中文或中英混写)
- [ ] `meta/info.json` 中 `features` 字段完整且与实际数据匹配
- [ ] 所有相机目录命名为 `observation.images.<cam_name>` 且在 episode 间保持一致
- [ ] 数据集 README/数据卡注明:采集者、采集日期、机器人型号、控制频率、坐标系、单位、是否绝对动作
