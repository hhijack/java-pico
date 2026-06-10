package org.ai.aihelper.pico;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * 【Agent 架构 / 配置层】mini-pico 风格的配置加载器。
 *
 * <h2>配置源优先级（高 → 低）</h2>
 * <ol>
 *   <li>环境变量（如 {@code DEEPSEEK_API_KEY}）</li>
 *   <li>项目根目录下的 {@code .pico/config.json} 文件</li>
 *   <li>代码内默认值</li>
 * </ol>
 *
 * <h2>它在整个 Agent 体系里的位置</h2>
 * PicoCLI 启动时会 {@code new PicoConfig()}，把读到的字段（API key、base url、模型名、
 * 工作区路径、最大步数、max_tokens 等）注入到 DeepSeekClient 和 PicoAgent。
 *
 * <h2>为什么不是 Spring @ConfigurationProperties？</h2>
 * 这套配置主要供 CLI 模式使用，且每条都需要"env > file > default"的三段降级，
 * 用纯手写 first(...) 链更直观，比自定义 PropertySource + EnvironmentPostProcessor 简洁。
 */
public class PicoConfig {

    public final String deepseekApiKey;
    public final String deepseekBaseUrl;
    public final String deepseekModel;
    public final String workspacePath;
    public final String ragUrl;
    public final int maxSteps;
    public final int maxNewTokens;

    public PicoConfig() {
        // 尝试从 .pico/config.json 读取（值可能是 String 或 Integer）
        Map<String, Object> fileCfg = loadFileConfig();

        this.deepseekApiKey = first(
                System.getenv("DEEPSEEK_API_KEY"),
                str(fileCfg.get("deepseek_api_key")),
                ""
        );
        this.deepseekBaseUrl = first(
                System.getenv("DEEPSEEK_BASE_URL"),
                str(fileCfg.get("deepseek_base_url")),
                "https://api.deepseek.com"
        );
        this.deepseekModel = first(
                System.getenv("DEEPSEEK_MODEL"),
                str(fileCfg.get("deepseek_model")),
                "deepseek-v4-flash"
        );
        this.workspacePath = first(
                str(fileCfg.get("workspace")),
                "."
        );
        this.ragUrl = first(
                System.getenv("RAG_URL"),
                str(fileCfg.get("rag_url")),
                "http://localhost:8080"
        );
        this.maxSteps = intVal(fileCfg.get("max_steps"), 8);
        this.maxNewTokens = intVal(fileCfg.get("max_new_tokens"), 2048);
    }

    /** 把 Object 转成 String（null 返回 null） */
    private static String str(Object v) {
        return v != null ? v.toString() : null;
    }

    /** 从 Object 读 int（支持 Integer 或 String 格式的数字） */
    private static int intVal(Object v, int def) {
        if (v instanceof Number n) return n.intValue();
        if (v != null) {
            try { return Integer.parseInt(v.toString().trim()); } catch (Exception ignored) {}
        }
        return def;
    }

    private static String first(String... values) {
        for (String v : values) {
            if (v != null && !v.isBlank()) return v;
        }
        return "";
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> loadFileConfig() {
        Path cfgFile = Path.of(".pico", "config.json");
        if (Files.isRegularFile(cfgFile)) {
            try {
                String text = Files.readString(cfgFile, StandardCharsets.UTF_8);
                return new com.fasterxml.jackson.databind.ObjectMapper()
                        .readValue(text, java.util.LinkedHashMap.class);
            } catch (IOException e) {
                System.err.println("Warning: failed to read .pico/config.json: " + e.getMessage());
            }
        }
        return Map.of();
    }
}
