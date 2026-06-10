package org.ai.aihelper.pico;

import org.ai.aihelper.ai.tools.CodeTool;
import org.ai.aihelper.ai.tools.WebSearchTool;

import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 【Agent 架构核心】java-pico 的 Java 版 Agent 循环主控。
 *
 * <h2>1. 它在整个 Agent 体系里的位置</h2>
 * <pre>
 *   PicoCLI (REPL 入口)
 *        │
 *        ▼
 *   PicoAgent  ◄─── 你正在看的这个类，是整个 Agent 的"大脑"
 *   ├── DeepSeekClient  （大模型 HTTP 客户端）
 *   ├── CodeTool        （本地文件操作工具集）
 *   ├── WebSearchTool   （网页搜索工具）
 *   └── RagRetriever    （RAG 知识库检索工具）
 * </pre>
 *
 * <h2>2. 它和 LangChain4j 的关系</h2>
 * 完全自己实现 ReAct 风格的 Agent 循环（拼 Prompt → 调 LLM → 解析标签 → 执行工具 → 续跑），
 * <b>不依赖</b> LangChain4j 的 AiServices / AgentExecutor。LangChain4j 在本项目里只承担
 * "Embedding 嵌入 + 向量存储 + Rerank 精排" 这一段 RAG 能力。
 *
 * <h2>3. Agent 循环的伪代码</h2>
 * <pre>{@code
 *   ask(userMessage):
 *     history.append(userMessage)
 *     for step in 0..maxSteps:
 *         prompt      = buildPrompt(history, workspace, tools)   // 拼系统提示+历史+工具表
 *         rawResponse = llm.complete(prompt)                    // 调 LLM
 *         parsed      = parse(rawResponse)                      // 解析 <tool> / <final> / <error>
 *         switch parsed.kind:
 *             case "error" : return error
 *             case "retry" : continue                            // 让 LLM 看到自己的输出再试一次
 *             case "tool"  : history.append(toolResult); continue
 *             case "final" : return final answer
 *     return "达到步数上限"
 * }</pre>
 *
 * <h2>4. 协议约定：LLM 输出格式</h2>
 * LLM 被要求 EXACTLY 输出以下三种 XML 标签之一（由 {@link #buildPrompt} 中的 system prompt 约束）：
 * <ul>
 *   <li>{@code <tool>{"name":"list_files","args":{"path":"."}}</tool>} — 调工具，参数是 JSON</li>
 *   <li>{@code <final>最终答案</final>} — 结束循环并返回</li>
 *   <li>{@code <error>出错原因</error>} — 主动报错（实际很少触发，HTTP 错误时由客户端写入）</li>
 * </ul>
 *
 * <h2>5. 工具集</h2>
 * 当前注册的工具（共 7 个）：
 * <ul>
 *   <li>list_files / read_file / write_file / patch_file / search — 本地文件操作（CodeTool）</li>
 *   <li>search_knowledge — RAG 知识库检索（RagRetriever）</li>
 *   <li>web_search — 联网搜索（WebSearchTool）</li>
 * </ul>
 * 注意：get_current_time（CurrentTimeTool）虽然存在，但 PicoAgent 当前并未在 executeTool 里分发它，
 * 保留以备未来扩展。
 */
public class PicoAgent {

    // ===================================================================
    //                          LLM 输出协议正则
    // ===================================================================
    // 三个标签对应 LLM 在 system prompt 中被强制要求的输出格式（见 buildPrompt）。
    // 使用非贪婪匹配 (.*?) + DOTALL，保证多行 JSON 也能正确切分。
    private static final Pattern TOOL_TAG = Pattern.compile("<tool>(.*?)</tool>", Pattern.DOTALL);
    private static final Pattern FINAL_TAG = Pattern.compile("<final>(.*?)</final>", Pattern.DOTALL);
    private static final Pattern ERROR_TAG = Pattern.compile("<error>(.*?)</error>", Pattern.DOTALL);

    // ANSI colors
    private static final String C_RESET = "[0m";
    private static final String C_DIM = "[2m";
    private static final String C_CYAN = "[36m";
    private static final String C_YELLOW = "[33m";
    private static final String C_BLUE = "[34m";
    private static final String C_GREEN = "[32m";
    private static final String C_RED = "[31m";
    private static final String C_BOLD = "[1m";
    private static final String C_BRIGHT_GREEN = "[92m";

    // ===================================================================
    //                           依赖注入的成员
    // ===================================================================
    /** 大模型客户端：负责把 prompt 发到 DeepSeek 并取回文本 */
    private final DeepSeekClient llm;
    /** 本地文件操作工具集：list_files / read_file / write_file / patch_file / search */
    private final CodeTool codeTool;
    /** 网页搜索工具：处理"实时新闻/最新资讯"类请求 */
    private final WebSearchTool webSearchTool;
    /** RAG 检索器：把 query 投到向量库 + Rerank，返回带分数的知识片段 */
    private final RagRetriever ragRetriever;
    /** Agent 工作区根目录（所有文件操作都被限制在此目录下，防止越权访问） */
    private final Path workspaceRoot;
    /** Agent 循环最大步数：超过此步数仍未拿到 final，就强制终止（防死循环 / 防 LLM 失控） */
    private final int maxSteps;
    /** 每次调 LLM 允许生成的最大 token 数（控制单次响应长度和成本） */
    private final int maxNewTokens;

    /**
     * 多轮会话历史。
     * <p>
     * 每条记录的 role 取值：{@code user} / {@code assistant} / {@code tool}。
     * 这里的"tool"角色是把工具执行结果以 assistant 视角回灌给 LLM 的占位字段，
     * 它会在 {@link #renderHistory} 中被格式化成"Knowledge base results: ..."等可读文本。
     * <p>
     * 之所以用 {@code List<Map<String,String>>} 而非专用 record，
     * 是为了零依赖地保持和 LLM "messages" 数组的 schema 兼容。
     */
    private final List<Map<String, String>> history = new ArrayList<>();

    /**
     * 构造方法（由 PicoCLI 手动装配；不是 Spring Bean）。
     *
     * @param llm            DeepSeek HTTP 客户端
     * @param codeTool       本地文件工具（Spring Bean 注入）
     * @param webSearchTool  网页搜索工具（Spring Bean 注入）
     * @param ragRetriever   RAG 检索包装
     * @param workspaceRoot  工作区根目录，所有文件操作都被 resolve() 限制在此目录下
     * @param maxSteps       Agent 循环最大步数（防死循环）
     * @param maxNewTokens   单次 LLM 调用的 max_tokens
     */
    public PicoAgent(DeepSeekClient llm, CodeTool codeTool, WebSearchTool webSearchTool,
                     RagRetriever ragRetriever, Path workspaceRoot, int maxSteps, int maxNewTokens) {
        this.llm = llm;
        this.codeTool = codeTool;
        this.webSearchTool = webSearchTool;
        this.ragRetriever = ragRetriever;
        this.workspaceRoot = workspaceRoot;
        this.maxSteps = maxSteps;
        this.maxNewTokens = maxNewTokens;
    }

    // ===================================================================
    //                          对外入口：ask()
    // ===================================================================

    /**
     * Agent 循环的对外入口：处理用户一条消息并返回最终答案。
     *
     * <h3>循环不变量</h3>
     * <ul>
     *   <li>{@code toolSteps} 统计成功执行的工具调用次数（达到 maxSteps 强制停止）</li>
     *   <li>{@code attempts} 统计 LLM 调用总次数（达到 maxSteps+3 强制停止，防 LLM 反复输出非法格式）</li>
     *   <li>每次循环都把 LLM 的输出写回 history，让 LLM "看到自己上一轮说了什么"，从而有机会自纠</li>
     * </ul>
     *
     * <h3>四类 ParseResult 的处理</h3>
     * <ul>
     *   <li>{@code error}  — 终止并返回错误信息（不写 history，避免污染后续 prompt）</li>
     *   <li>{@code retry}  — 把"为什么需要重试"写回 history，让 LLM 自纠</li>
     *   <li>{@code tool}   — 执行工具、把结果写回 history、toolSteps++、继续循环</li>
     *   <li>{@code final}  — 终止并返回最终答案</li>
     * </ul>
     *
     * @param userMessage 用户的原始输入
     * @return 最终答案（final 标签内容 / 错误信息 / 步数耗尽提示）
     */
    public String ask(String userMessage) {
        // 1) 把用户消息压入历史，作为后续 prompt 的"当前请求"
        history.add(Map.of("role", "user", "content", userMessage));

        // 2) 两套上限：toolSteps 控"做了多少事"，attempts 控"叫 LLM 多少次"
        int toolSteps = 0;
        int attempts = 0;
        int maxAttempts = maxSteps + 3;

        // 3) Agent 主循环
        while (toolSteps < maxSteps && attempts < maxAttempts) {
            attempts++;

            // 3.1 拼 prompt（系统提示 + 工具表 + 工作区快照 + 历史 + 当前请求）
            String prompt = buildPrompt(userMessage, history);

            // 3.2 调 LLM，并打日志
            System.out.println(C_DIM + "\n--- [PicoAgent] LLM call #" + attempts + " ---" + C_RESET);
            long start = System.currentTimeMillis();
            String raw = llm.complete(prompt, maxNewTokens);
            long elapsed = System.currentTimeMillis() - start;
            System.out.println(C_DIM + "--- [PicoAgent] response in " + elapsed + "ms ---" + C_RESET);
            // 只有非 final 响应才打印原始输出（避免和最终答案重复打印）
            if (!raw.contains("<final>")) {
                System.out.println(C_CYAN + truncate(raw, 300) + C_RESET);
            }

            // 3.3 解析 LLM 输出，按 4 种 kind 分发
            ParseResult parsed = parse(raw);
            switch (parsed.kind) {
                case "error" -> {
                    // 错误：直接返回，不再继续循环
                    history.add(Map.of("role", "assistant", "content", raw));
                    return "Error: " + parsed.content;
                }
                case "retry" -> {
                    // LLM 没产出合法标签，把"为什么需要重试"写回 history 让它自纠
                    history.add(Map.of("role", "assistant", "content", parsed.content));
                    // 继续循环
                }
                case "tool" -> {
                    // 工具调用：执行 → 写回 history → 继续循环
                    toolSteps++;
                    String toolResult = executeTool(parsed.name, parsed.args);

                    // 给知识库/网页搜索结果加醒目的可视化分隔，方便用户在终端跟踪
                    if ("search_knowledge".equals(parsed.name)) {
                        System.out.println(C_BLUE + "\n========== Knowledge Base Results ==========" + C_RESET);
                        System.out.println(C_YELLOW + toolResult + C_RESET);
                        System.out.println(C_BLUE + "=============================================\n" + C_RESET);
                    } else if ("web_search".equals(parsed.name)) {
                        System.out.println(C_BLUE + "\n============ Web Search Results ============" + C_RESET);
                        System.out.println(C_YELLOW + toolResult + C_RESET);
                        System.out.println(C_BLUE + "=============================================\n" + C_RESET);
                    }

                    // 工具结果入栈，下一轮 prompt 渲染时会被格式化成"Tool result (xxx): ..."
                    history.add(Map.of("role", "tool", "name", parsed.name,
                                       "args", String.valueOf(parsed.args),
                                       "content", toolResult));
                }
                case "final" -> {
                    // 收到 final：终止循环，返回答案
                    history.add(Map.of("role", "assistant", "content", parsed.content));
                    return parsed.content;
                }
            }
        }

        // 4) 循环退出：要么 step 用完，要么 LLM 反复乱答
        String msg = toolSteps >= maxSteps
                ? C_RED + "Stopped after reaching the step limit (" + maxSteps + ")." + C_RESET
                : C_RED + "Stopped after too many malformed responses." + C_RESET;
        history.add(Map.of("role", "assistant", "content", msg));
        return msg;
    }

    /**
     * 返回历史记录的不可变副本（防止外部修改内部状态）。
     * <p>
     * PicoCLI 的 /history 命令会调用此方法。
     */
    public List<Map<String, String>> getHistory() {
        return List.copyOf(history);
    }

    // ===================================================================
    //                      Prompt 构建（系统提示 + 上下文）
    // ===================================================================

    /**
     * 拼装一次 LLM 调用的完整 prompt。
     *
     * <h3>Prompt 结构（自上而下）</h3>
     * <ol>
     *   <li><b>角色定义</b>：告诉 LLM 你是 java-pico，专攻 Java 面试</li>
     *   <li><b>决策规则</b>：Java 面试题 → 先 search_knowledge；闲聊 → 直答；实时信息 → web_search</li>
     *   <li><b>输出协议</b>：强制 EXACTLY 输出 {@code <tool>} / {@code <final>} 之一（这是 Agent 循环能跑通的前提）</li>
     *   <li><b>工具清单</b>：所有可用工具的 JSON Schema 简版</li>
     *   <li><b>工作区快照</b>：让 LLM 知道当前有哪些文件可操作</li>
     *   <li><b>历史消息</b>：最近 12 条（renderHistory 控制）</li>
     *   <li><b>当前用户请求</b>：用 {@code User: ...} 引导 LLM 回答</li>
     * </ol>
     *
     * <h3>为什么用 text prompt 而非 ChatML/JSON Schema tools？</h3>
     * DeepSeek-V4 兼容 OpenAI ChatML 格式，但其原生 function-calling 协议在多轮工具调用场景下
     * 行为不稳定。本项目刻意把所有"工具调用"压成 XML 标签 + JSON 字符串，让 LLM 在一次完整上下文里
     * 自由决定何时调用、调用哪个、自己推理，再由 {@link #parse} 解析回来 —— 实际效果更稳。
     *
     * @param userMessage 当前用户消息（同时也作为 history 的最新一条；保留两个参数仅为阅读清晰）
     * @param history     当前会话的全部历史（含 user / assistant / tool 三种 role）
     * @return 喂给 {@link DeepSeekClient#complete} 的完整 prompt
     */
    private String buildPrompt(String userMessage, List<Map<String, String>> history) {
        String workspace = workspaceRoot.toAbsolutePath().normalize().toString();
        String snapshot = workspaceSnapshot();
        String h = renderHistory(history);
        return """
You are java-pico, an AI assistant specialized in Java interview preparation.
You have a KNOWLEDGE BASE with high-quality Java interview content (JVM, JUC, Spring, MySQL, Redis, MQ, etc.).

RULES:
- For Java interview questions: FIRST call search_knowledge ONCE, then answer based on results.
- For general questions (about tools, chat, etc.): answer directly, no search needed.
- For real-time info (news, jobs): use web_search.

Reply with EXACTLY one of:
<tool>{"name":"...","args":{...}}</tool>
<final>Your answer</final>

=== KNOWLEDGE BASE (use FIRST for interview questions) ===
- search_knowledge {"query":"..."} — search Java interview knowledge base

=== WEB SEARCH (use for real-time info, news, jobs) ===
- web_search {"query":"..."} — search the web for latest information

=== File operations (use only when asked) ===
- list_files {"path": "."} — list directory
- read_file {"path":"...","start":1,"end":120} — read file by line range
- write_file {"path":"...","content":"..."} — write file
- patch_file {"path":"...","old_text":"...","new_text":"..."} — edit text
- search {"pattern":"...","path":"."} — search text

Workspace: """
                + workspace
                + "\nFiles:\n" + snapshot
                + "\nHistory:\n" + h
                + "\nUser: " + userMessage + "\n";
    }

    /**
     * 工作区文件快照：调用 {@link CodeTool#listFiles} 拿到顶层目录结构。
     * <p>
     * 失败时返回降级字符串（不让 LLM 看到堆栈），避免在 prompt 里出现大段异常文本膨胀 token。
     */
    private String workspaceSnapshot() {
        try {
            return codeTool.listFiles(".");
        } catch (Exception e) {
            return "(workspace snapshot failed: " + e.getMessage() + ")";
        }
    }

    /**
     * 把结构化 history 渲染成可喂给 LLM 的纯文本。
     * <p>
     * 规则：
     * <ul>
     *   <li>只保留最近 12 条（更早的会被截断，控制 prompt 长度和成本）</li>
     *   <li>{@code tool} 角色：根据工具名映射成"Knowledge base results:" / "Web search results:" / "Tool result (xxx):"</li>
     *   <li>非 tool 角色：直接拼成 {@code role: content} 形式</li>
     *   <li>长内容会被截断（tool 结果截 300 字符，其他 200 字符），防止 prompt 爆炸</li>
     * </ul>
     */
    private static String renderHistory(List<Map<String, String>> history) {
        if (history.isEmpty()) return "(empty)";
        List<Map<String, String>> recent = history.size() > 12
                ? history.subList(history.size() - 12, history.size())
                : history;
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> item : recent) {
            String role = item.getOrDefault("role", "");
            if ("tool".equals(role)) {
                String name = item.getOrDefault("name", "");
                String content = item.getOrDefault("content", "");
                if ("search_knowledge".equals(name)) {
                    sb.append("Knowledge base results:\n").append(content).append("\n");
                } else if ("web_search".equals(name)) {
                    sb.append("Web search results:\n").append(content).append("\n");
                } else {
                    sb.append("Tool result (").append(name).append("): ").append(truncate(content, 300)).append("\n");
                }
            } else {
                sb.append(role).append(": ").append(truncate(item.getOrDefault("content", ""), 200)).append("\n");
            }
        }
        return sb.toString();
    }

    // ===================================================================
    //                  解析层：把 LLM 原始输出翻译成 ParseResult
    // ===================================================================

    /**
     * 解析结果四元组。
     * <p>
     * 一种"discriminated union"的轻量实现：靠 {@link #kind} 字段区分四种语义，
     * 其它字段在不同 kind 下意义不同：
     * <ul>
     *   <li>{@code "tool"}  → {@link #name} 是工具名，{@link #args} 是参数 Map，{@link #content} 为空</li>
     *   <li>{@code "final"} → {@link #content} 是最终答案</li>
     *   <li>{@code "error"} → {@link #content} 是错误描述</li>
     *   <li>{@code "retry"} → {@link #content} 是"为什么需要重试"的提示</li>
     * </ul>
     */
    private static class ParseResult {
        final String kind; // "tool", "final", "retry", "error"
        final String name;
        final Map<String, Object> args;
        final String content;

        ParseResult(String kind, String name, Map<String, Object> args, String content) {
            this.kind = kind;
            this.name = name;
            this.args = args;
            this.content = content;
        }
    }

    /**
     * 把 LLM 的原始输出解析成结构化 {@link ParseResult}。
     *
     * <h3>解析优先级（关键）</h3>
     * <ol>
     *   <li>{@code <error>} 最高优先（遇到立即终止循环）</li>
     *   <li>{@code <tool>...</tool>} 次之（参数必须是合法 JSON）</li>
     *   <li>{@code <final>...</final>} 再次（非空才算合法）</li>
     *   <li>兜底：长文本视为 final（防止 maxNewTokens 截断导致 {@code </final>} 丢失）</li>
     *   <li>否则视为 retry（让 LLM 再试一次）</li>
     * </ol>
     *
     * <h3>为什么用正则而不是结构化 function calling？</h3>
     * 兼容所有 LLM（包括没开 function calling 的小模型），并且不依赖 LangChain4j AiServices。
     * 代价是要容忍 LLM 输出格式错误（通过 retry + maxAttempts 兜底）。
     */
    @SuppressWarnings("unchecked")
    private static ParseResult parse(String raw) {
        if (raw == null || raw.isBlank()) {
            return new ParseResult("retry", "", Map.of(), "model returned empty response");
        }

        // 优先检查 error
        Matcher em = ERROR_TAG.matcher(raw);
        if (em.find()) {
            return new ParseResult("error", "", Map.of(), em.group(1).trim());
        }

        // 检查 tool 标签
        Matcher tm = TOOL_TAG.matcher(raw);
        if (tm.find()) {
            String jsonStr = tm.group(1).trim();
            try {
                var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                var root = mapper.readTree(jsonStr);
                String name = root.path("name").asText("");
                Map<String, Object> args = new LinkedHashMap<>();
                if (root.has("args")) {
                    var argsNode = root.get("args");
                    argsNode.fieldNames().forEachRemaining(k -> args.put(k, argsNode.get(k).asText()));
                }
                if (name.isBlank()) {
                    return new ParseResult("retry", "", Map.of(), "tool tag missing 'name' field");
                }
                return new ParseResult("tool", name, args, "");
            } catch (Exception e) {
                return new ParseResult("retry", "", Map.of(), "model returned malformed tool JSON: " + e.getMessage());
            }
        }

        // 检查 final 标签
        Matcher fm = FINAL_TAG.matcher(raw);
        if (fm.find()) {
            String finalText = fm.group(1).trim();
            if (!finalText.isEmpty()) {
                return new ParseResult("final", "", Map.of(), finalText);
            }
            return new ParseResult("retry", "", Map.of(), "model returned empty final answer");
        }

        // 兜底：如果没有 tool/error/final 标签，但有大段文本，当作 final 答案
        // （防止 maxNewTokens 截断导致 </final> 丢失）
        String trimmed = raw.trim();
        if (trimmed.length() > 50 && !trimmed.contains("<tool>")) {
            return new ParseResult("final", "", Map.of(), trimmed);
        }

        return new ParseResult("retry", "", Map.of(), "model returned neither <tool> nor <final>");
    }

    // ===================================================================
    //                       工具执行层（dispatcher）
    // ===================================================================

    /**
     * 工具分发器：把 LLM 决定的 (name, args) 路由到对应的工具实现。
     * <p>
     * 当前共 7 个工具，按"文件操作 / 知识检索 / 联网搜索"三类组织：
     * <ul>
     *   <li>文件类（5 个）→ {@link CodeTool}：list_files / read_file / write_file / patch_file / search</li>
     *   <li>知识类（1 个）→ {@link RagRetriever}：search_knowledge</li>
     *   <li>联网类（1 个）→ {@link WebSearchTool}：web_search</li>
     * </ul>
     * 任何异常都会被 try/catch 吞掉并以 {@code "error: ..."} 字符串返回，
     * 这样 LLM 能看到错误信息并在下一次循环里自纠，而不是让 Agent 整体崩掉。
     */
    @SuppressWarnings("unchecked")
    private String executeTool(String name, Map<String, Object> args) {
        try {
            return switch (name) {
                case "list_files" -> codeTool.listFiles(asStr(args.get("path"), "."));
                case "read_file" -> codeTool.readFile(
                        asStr(args.get("path"), ""),
                        asInt(args.get("start"), 1),
                        asInt(args.get("end"), 120));
                case "write_file" -> codeTool.writeFile(
                        asStr(args.get("path"), ""),
                        asStr(args.get("content"), ""));
                case "patch_file" -> codeTool.patchFile(
                        asStr(args.get("path"), ""),
                        asStr(args.get("old_text"), ""),
                        asStr(args.get("new_text"), ""));
                case "search" -> codeTool.search(
                        asStr(args.get("pattern"), ""),
                        asStr(args.get("path"), "."));
                case "search_knowledge" -> ragRetriever.search(asStr(args.get("query"), ""));
                case "web_search" -> webSearchTool.search(asStr(args.get("query"), ""));
                default -> "error: unknown tool '" + name + "'";
            };
        } catch (Exception e) {
            return "error: tool " + name + " failed: " + e.getMessage();
        }
    }

    /**
     * 把任意 Object 安全转 String（null → 默认值）。
     * LLM 给的参数可能本来就是 String，但 Map.of 推导后类型会变成 Object，这里统一兜底。
     */
    private static String asStr(Object o, String def) {
        return o != null ? o.toString() : def;
    }

    /**
     * 把任意 Object 安全转 int。
     * 支持 Number（直接 intValue）和 String 数字两种形式；都失败时回退到默认值。
     */
    private static int asInt(Object o, int def) {
        if (o instanceof Number n) return n.intValue();
        try { return Integer.parseInt(o.toString()); } catch (Exception e) { return def; }
    }

    /**
     * 字符串截断（带省略号）。用于防止工具结果或 LLM 输出塞爆 prompt。
     */
    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() > max ? s.substring(0, max) + "..." : s;
    }
}
