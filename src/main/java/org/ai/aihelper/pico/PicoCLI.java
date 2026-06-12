package org.ai.aihelper.pico;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import org.ai.aihelper.ai.tools.CodeTool;
import org.ai.aihelper.ai.tools.WebSearchTool;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Scanner;

/**
 * 【Agent 架构入口】终端 REPL（Read-Eval-Print Loop）入口。
 * <p>
 * 整个 Agent 的"启动器 + 用户交互界面"：
 * <ol>
 *   <li>Spring Boot 启动时，因为标了 {@link Profile}("cli")，本类会被自动注册为 {@link ApplicationRunner}</li>
 *   <li>Spring 注入 {@link CodeTool}、{@link ContentRetriever}（来自 RagConfig）、{@link WebSearchTool}</li>
 *   <li>本类再手动 new 出 {@link DeepSeekClient}、{@link RagRetriever}、{@link PicoAgent}，因为它们和 Spring 容器无直接关系</li>
 *   <li>进入 REPL 循环：读 stdin → 调 agent.ask() → 打印答案 → 循环</li>
 * </ol>
 *
 * <h2>用法</h2>
 * <pre>
 *   java -jar ai-helper.jar --spring.profiles.active=cli
 *   # 或本地开发时指定 profile
 *   mvn spring-boot:run -Dspring-boot.run.profiles=cli
 * </pre>
 *
 * <h2>为什么不是 Spring Bean 全家桶？</h2>
 * PicoAgent 需要把 LLM 客户端、RAG 检索器、工作区路径等拼装起来，这是一次性组合；
 * 把它做成 Spring Bean 会引入很多配置类，反而增加复杂度。当前模式是"瘦 Spring 容器 + 手动装配核心"，
 * 调试时也能直接 new 出来跑单测。
 *
 * <h2>支持的交互指令</h2>
 * <ul>
 *   <li>普通文本 → 走 Agent 主循环</li>
 *   <li>{@code /exit} 或 {@code /quit} → 退出</li>
 *   <li>{@code /history} → 打印当前 Agent 的历史消息</li>
 * </ul>
 */
@Profile("cli")
@Component
public class PicoCLI implements ApplicationRunner {

    private final CodeTool codeTool;
    private final ContentRetriever contentRetriever;
    private final WebSearchTool webSearchTool;

    // ANSI colors
    private static final String C_RESET = "[0m";
    private static final String C_CYAN = "[36m";
    private static final String C_GREEN = "[32m";
    private static final String C_YELLOW = "[33m";
    private static final String C_DIM = "[2m";
    private static final String C_BOLD = "[1m";
    private static final String C_BRIGHT_GREEN = "[92m";

    public PicoCLI(CodeTool codeTool, ContentRetriever contentRetriever, WebSearchTool webSearchTool) {
        this.codeTool = codeTool;
        this.contentRetriever = contentRetriever;
        this.webSearchTool = webSearchTool;
    }

    /**
     * Spring Boot 启动完成后自动调用。
     * <p>
     * 整体流程：
     * <pre>
     *   1) 打印 ASCII 横幅
     *   2) 读 PicoConfig（env > 文件 > 默认）
     *   3) 校验 API Key
     *   4) 装配 DeepSeekClient / RagRetriever / PicoAgent
     *   5) 修复控制台编码（Windows 中文乱码）
     *   6) 进入 REPL 主循环
     * </pre>
     */
    @Override
    public void run(ApplicationArguments args) {
        System.out.println();
        System.out.println(C_CYAN + "╔══════════════════════════════════════╗");
        System.out.println("║        java-pico - AI Agent          ║");
        System.out.println("║  Type /exit to quit                  ║");
        System.out.println("╚══════════════════════════════════════╝" + C_RESET);
        System.out.println();

        // 加载配置
        PicoConfig config = new PicoConfig();
        if (config.apiKey.isBlank()) {
            System.err.println("Error: API_KEY is not set.");
            System.err.println("Set it via environment variable or create .pico/config.json");
            return;
        }

        // 初始化三个 LLM 客户端（路由 + 中等 + 最强）
        DeepSeekClient routerLlm = new DeepSeekClient(
                config.apiKey,
                config.baseUrl,
                config.routerModel
        );
        DeepSeekClient mediumLlm = new DeepSeekClient(
                config.apiKey,
                config.baseUrl,
                config.mediumModel
        );
        DeepSeekClient maxLlm = new DeepSeekClient(
                config.apiKey,
                config.baseUrl,
                config.maxModel
        );

        // 初始化 RAG 检索
        RagRetriever rag = new RagRetriever(contentRetriever);

        // 初始化 Agent
        Path workspace = Path.of(config.workspacePath).toAbsolutePath().normalize();
        System.out.println(C_DIM + "Workspace: " + C_YELLOW + workspace + C_RESET);
        System.out.println(C_DIM + "Router:  " + C_YELLOW + config.routerModel + C_RESET);
        System.out.println(C_DIM + "Medium:  " + C_YELLOW + config.mediumModel + C_RESET);
        System.out.println(C_DIM + "Max:     " + C_YELLOW + config.maxModel + C_RESET);
        System.out.println(C_DIM + "Max steps: " + C_YELLOW + config.maxSteps + C_RESET);
        System.out.println();

        // 修复控制台编码，确保中文 I/O 正常工作（Windows + Git Bash 场景）
        fixConsoleEncoding();

        PicoAgent agent = new PicoAgent(routerLlm, mediumLlm, maxLlm,
                config.routerModel, config.mediumModel, config.maxModel,
                codeTool, webSearchTool, rag, workspace,
                config.maxSteps, config.maxNewTokens);

        // 加载持久化记忆
        Path memoryFile = Path.of(".pico", "memory.json");
        if (Files.isRegularFile(memoryFile)) {
            try {
                String json = Files.readString(memoryFile, StandardCharsets.UTF_8);
                agent.getMemory().loadFrom(json);
                System.out.println(C_DIM + "Memory loaded: " + C_YELLOW + memoryFile + C_RESET);
            } catch (Exception e) {
                System.out.println(C_DIM + "Memory load failed: " + e.getMessage() + C_RESET);
            }
        }

        // REPL 循环（使用 UTF-8 读取输入以支持中文）
        @SuppressWarnings("resource")
        Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8);
        while (true) {
            // 显示上次对话使用的模型名（首次显示默认推理模型）
            String modelTag = agent.getCurrentModelName();
            String modelDisplay = modelTag != null ? " [" + modelTag + "]" : "";
            System.out.print(C_BOLD + C_GREEN + "java-pico" + C_RESET
                    + C_DIM + modelDisplay + C_RESET
                    + C_BOLD + C_GREEN + "> " + C_RESET);
            String input;
            try {
                input = scanner.nextLine().trim();
            } catch (Exception e) {
                saveMemory(agent, memoryFile);
                System.out.println();
                break;
            }

            if (input.isBlank()) continue;
            if (input.equals("/exit") || input.equals("/quit")) {
                saveMemory(agent, memoryFile);
                System.out.println(C_YELLOW + "bye!" + C_RESET);
                break;
            }
            if (input.equals("/history")) {
                agent.getHistory().forEach(h -> System.out.println(C_DIM + "  " + h + C_RESET));
                continue;
            }

            long start = System.currentTimeMillis();
            String answer = agent.ask(input);
            long elapsed = System.currentTimeMillis() - start;

            System.out.println();
            System.out.println(C_BRIGHT_GREEN + answer + C_RESET);
            System.out.println();
            System.out.println(C_DIM + "--- (" + elapsed + "ms) ["
                    + C_YELLOW + agent.getCurrentModelName() + C_RESET + C_DIM + "] ---" + C_RESET);
            System.out.println();
        }
    }

    /**
     * 保存会话记忆到文件。
     */
    private static void saveMemory(PicoAgent agent, Path memoryFile) {
        try {
            String json = agent.getMemory().serialize();
            if (json != null && !json.equals("{}")) {
                Files.createDirectories(memoryFile.getParent());
                Files.writeString(memoryFile, json, StandardCharsets.UTF_8);
                System.out.println(C_DIM + "Memory saved: " + C_YELLOW + memoryFile + C_RESET);
            }
        } catch (Exception e) {
            System.err.println("Warning: failed to save memory: " + e.getMessage());
        }
    }

    /**
     * 修复控制台 I/O 编码为 UTF-8，确保中文输入输出在 Windows + Git Bash 中正常。
     * <p>
     * 问题: Git Bash 以 UTF-8 编码发送输入，但 JVM 默认使用系统编码 (中文 Windows = GBK/CP936)。
     * 这导致 Scanner 将 UTF-8 字节按 GBK 解码，产生乱码。
     * 解决方案:
     * - 输入: Scanner 显式使用 UTF-8 编码
     * - 输出: System.out/System.err 重新包装为 UTF-8 PrintStream
     */
    private static void fixConsoleEncoding() {
        // 仅在非 UTF-8 环境做输出修复
        if (!StandardCharsets.UTF_8.equals(Charset.defaultCharset())) {
            try {
                System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8.name()));
                System.setErr(new PrintStream(System.err, true, StandardCharsets.UTF_8.name()));
            } catch (Exception e) {
                System.err.println("Warning: could not set console to UTF-8: " + e.getMessage());
            }
        }
    }
}
