# java-pico Agent 项目面试介绍文档

## 一、项目概述

本项目是一个**从零手写的 ReAct Agent 框架**，基于阿里云 DashScope（通义千问）构建的 Java 面试 AI 助手。核心特点是不依赖 LangChain4j 等 Agent 框架，自己实现了 Agent 循环、模型路由、多层记忆管理。

**技术栈：**
- Spring Boot 4.0.6 + Java 21
- 阿里云 DashScope：通义千问（qwen3-8b / 3.7-plus / 3.7-max）
- PostgreSQL + pgvector（向量数据库）
- 纯手写 Agent 框架（PicoAgent），不依赖 LangChain4j AiServices

---

## 二、核心模块

### 2.1 自定义 ReAct Agent 循环（PicoAgent）

**职责**：实现"思考-行动-观察"的 Agent 主循环

**核心流程**：
```
用户消息 → buildPrompt(拼接完整 prompt) → LLM 调用 → parse(解析标签)
  ├─ <tool> → 执行工具 → 结果写回 history → 继续循环
  ├─ <final> → 返回最终答案
  ├─ <error> → 返回错误
  └─ 无标签 → retry（让 LLM 重新生成）
```

**关键实现**：
- Prompt 即协议：用文本标签 (`<tool>` / `<final>` / `<reroute>`) 让 LLM 选择下一步，零依赖 function calling
- 工具路由：7 个工具（list_files / read_file / write_file / patch_file / search / search_knowledge / web_search）
- 路径安全检查：所有文件操作经过 `resolve()` 校验，防止 LLM 越权访问工作区外文件

---

### 2.2 三级模型路由（Router→Medium→Max）

**职责**：根据问题复杂度分配合适的模型，避免浪费

```
用户 → qwen3-8b (Router) 分类
  ├─ direct → 8b 直接回答（闲聊、简单问答）✅ 省成本
  ├─ knowledge → qwen3.7-plus 查知识库
  ├─ web_search → qwen3.7-plus 联网搜索
  └─ code → qwen3.7-max 编码/文件操作
```

**关键设计**：
- 首次消息走完整路由，后续复用上次路由结果（同一轮对话中用户需求不会频繁切换）
- 任意模型可主动输出 `<reroute>` 标签要求切换模型，框架截获后自动切换
- 每次 LLM 调用在终端显示当前模型名：`--- LLM call #1 (model: qwen3.7-plus) ---`

---

### 2.3 结构化多层记忆（SessionMemory）

**职责**：跨轮对话的记忆管理，替代传统滑动窗口

**四层结构**：

| 层级 | 内容 | 特点 |
|------|------|------|
| task_summary | 当前任务摘要（用户消息第一行） | 历史裁剪后仍保留的对话锚点 |
| recent_files | 最近访问的文件路径（去重，上限 5 个） | 自动跟踪文件操作 |
| file_summaries | 文件内容摘要 + 新鲜度 | 带 MD5 哈希校验，文件修改后旧摘要自动失效 |
| episodic_notes | 跨轮笔记（标签+关键词召回） | 零 embedding，纯文本匹配召回 top 3 |

**设计理念**：记忆不由 LLM 生成（不额外消耗 token），所有字段由代码规则自动提炼。

---

### 2.4 持久化架构（Session / Run）

**两条独立持久化路径**：

```
.pico/
├── session.json        ← 跨轮会话（history + memory），退出自动保存，启动自动恢复
└── runs/
    └── run_yyyymmdd-HHMMss/
        ├── trace.jsonl ← 事件流水线，每次 LLM 调用/工具执行都有记录
        └── report.json ← 最终报告（状态、步数、耗时、结果摘要）
```

- **Session**：恢复现场用，覆写保存
- **Run**：审计复盘用，追加不可篡改

---

## 三、技术难点与解决方案

### 难点1：LLM 输出格式不稳定

**问题**：模型输出的 XML 标签可能残缺、格式错误、或者不按协议输出

**解决方案**：
1. 正则非贪婪匹配 + DOTALL 模式，应对多行 JSON
2. 5 层解析优先级：`<error>` → `<tool>` → `<final>` → 长文本兜底 → retry
3. 连续 retry 时自动降级到 8b 模型直接回答

---

### 难点2：文件摘要新鲜度追踪

**问题**：Agent 读过的文件摘要可能因后续修改而过期

**解决方案**：
- 每条文件摘要附带当前文件内容的 MD5 哈希作为 freshness 标记
- 每次渲染 prompt 时重新计算哈希，对不上就不展示
- `write_file` / `patch_file` 后主动失效关联摘要

---

### 难点3：路径安全与跨文件系统

**问题**：WSL 环境下 `Path.startsWith()` / `relativize()` 因文件系统不匹配抛异常

**解决方案**：
- 用 `toAbsolutePath().normalize()` 替代 `toRealPath()`（后者会解符号链，可能进入不同文件系统）
- `startsWith` / `relativize` 抛 `ProviderMismatchException` 时防御性降级

---

### 难点4：模型路由的卡死问题

**问题**：路由跳过后，当前模型无法处理但又不输出 `<reroute>` 标签，导致死循环

**解决方案**：
- 所有模型统一用同一套 prompt（工具列表 + 记忆 + 历史），8b 也走相同的 agentLoop
- `<reroute>` 标签让模型能主动要求升级/降级
- 连续 retry 达上限时输出明确错误信息

---

## 四、性能优化实践

### 优化1：路由跳过多余 LLM 调用

**优化前**：每次用户消息都调 8b 分类

**优化后**：首次路由后复用 `lastRoute`，后续跳过 router 调用

**效果**：连续闲聊场景减少 1 次 LLM 调用/轮

---

### 优化2：read_file 结果不走历史

**优化前**：read_file 返回的完整文件内容塞进 `renderHistory()`，被截断后 LLM 反复重读

**优化后**：read_file 的结果经过 `summarizeContent()` 压成摘要（前 30 行 + 末 5 行），存到 SessionMemory 的 `file_summaries`，历史只保留摘要

**效果**：解决了 LLM 反复重读同一个文件的死循环

---

### 优化3：文件摘要替代全文

**优化前**：`summarizeContent()` 只返回文件首行 + 统计信息，LLM 无法工作

**优化后**：≤1000 字符返回全文，＞1000 字符返回前 30 行 + "省略" + 末 5 行

**效果**：模型能看到文件的结构（import / 类声明 / 方法签名）和结尾

---

## 五、设计决策与权衡

### 决策1：为什么自己写 Agent 循环而不是用 LangChain4j AiServices？

**选择 A（当前）**：手写 PicoAgent
- 优点：完全控制 prompt 结构、解析逻辑透明、不依赖框架版本兼容性、可加自定义标签（`<reroute>`）
- 缺点：需要自行处理所有边界情况

**选择 B**：LangChain4j AiServices
- 优点：开箱即用、function calling 原生支持
- 缺点：多轮工具调用行为不稳定、调试困难、prompt 结构不透明

**结论**：对于需要深度定制的 Agent 场景，手写循环的灵活性和可控性超过框架带来的便利。

---

### 决策2：为什么用文本标签而不是 Function Calling？

| 对比 | 文本标签 | Function Calling |
|------|---------|-----------------|
| 模型兼容性 | 任何文本模型 | 需 API 级支持 |
| 调试 | 直接看文本，直观 | 看 JSON，略绕 |
| 容错 | 截断也能兜底 | 结构错误即失败 |
| 灵活性 | 可混写推理+工具调用 | 严格分离 |

**结论**：文本标签方案更适合"小模型也能跑"的场景，且调试体验更好。

---

### 决策3：为什么分 Session 和 Run 两条存储路径？

**Session**（`session.json`）：恢复现场用，启动时自动加载历史+记忆，退出时覆写保存

**Run**（`runs/<run_id>/`）：审计复盘用，trace 追加写不可篡改，report 汇总结果

合起来就是：Session 回答"我还记得什么"，Run 回答"我是怎么到这的"。

---

## 六、面试官可能的追问

1. **Agent 循环**：
   - "为什么用 `<tool>` `<final>` 文本标签而不是 JSON？如果模型不按格式输出怎么办？"
   - "maxSteps 和 maxAttempts 为什么是两个不同的计数器？"
   - "history 只在渲染时取最近 12 条，但列表本身不压缩，这合理吗？"

2. **模型路由**：
   - "路由策略是首次路由后续复用，如果用户突然切换话题怎么办？"
   - "`<reroute>` 标签让模型主动换模型，小模型真的能正确输出这个标签吗？"
   - "3 个模型的调用成本比例大概多少？有数据支撑路由策略的价值吗？"

3. **记忆管理**：
   - "file_summaries 用 MD5 做 freshness，MD5 碰撞怎么办？（面试考点：MD5 已知碰撞攻击）"
   - "episodic_notes 的纯文本召回，为什么不加 embedding？"
   - "为什么不用 LLM 来做摘要？如果用 LLM 做摘要会有什么优劣？"

4. **Session/Run 架构**：
   - "session.json 里的 history 只保留 50 条，超过怎么办？"
   - "run 的 trace.jsonl 不断追加，文件会无限增长吗？"
   - "如果程序突然崩溃，task_state.json 的覆写机制能保证多少数据不丢？"

5. **工具安全**：
   - "CodeTool.resolve() 的路径检查能防止 `../../etc/passwd` 吗？"
   - "如果 LLM 被注入攻击，让 agent 执行恶意文件操作怎么办？"

---

## 七、项目亮点总结（口头版）

> "我手写了一个 ReAct Agent 框架，不依赖任何 Agent 框架（LangChain4j 只用来做 RAG 的 embedding 和向量存储），核心亮点有三个：
>
> 1. **三层模型路由**：用 qwen3-8b 做分类器，qwen3.7-plus 处理常规任务，qwen3.7-max 只做最难的任务。模型可以主动输出 `<reroute>` 标签要求切换，不需要每轮都调路由。简单闲聊 8b 直接答，完全不浪费大模型。
>
> 2. **结构化四层记忆**：不是简单的滑动窗口，而是分成任务摘要、最近文件、文件摘要（带文件内容 MD5 新鲜度校验）、情景笔记。文件改了旧摘要自动失效，不需要记忆这种笔记不会过期。
>
> 3. **Session/Run 两级持久化**：Session 保存历史+记忆，下次启动自动恢复现场；Run 记录每次调用的完整事件流水线（prompt、LLM 响应、工具执行耗时），复盘调试直接看 trace。
>
> 最大的挑战：文本标签解析的健壮性——LLM 输出不可控，5 层解析 + 兜底逻辑 + 重试机制组合才让循环稳定跑通。"

*文档版本：2026-06-13*
