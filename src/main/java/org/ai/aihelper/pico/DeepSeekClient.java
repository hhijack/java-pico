package org.ai.aihelper.pico;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * 【Agent 架构 / LLM 客户端】DeepSeek API 的极简 HTTP 客户端。
 *
 * <h2>1. 它在整个 Agent 体系里的位置</h2>
 * PicoAgent 在 Agent 循环里需要把"拼好的 prompt"发到 LLM 并取回文本，
 * 这一步由本类完成。它是 Agent 大脑和外部 LLM 之间的"喉舌"。
 *
 * <h2>2. 为什么自己写而不直接用 LangChain4j ChatModel？</h2>
 * 故意绕开 LangChain4j 的 ChatModel 抽象，原因：
 * <ul>
 *   <li>让 prompt 结构对自己完全透明（方便调试、调优、加自定义标签）</li>
 *   <li>避免 LangChain4j 在多轮工具调用场景下"自作主张"地重排消息</li>
 *   <li>代码量更小、依赖更少</li>
 * </ul>
 *
 * <h2>3. 协议</h2>
 * 使用 DeepSeek 的 OpenAI 兼容端点：{@code POST {baseUrl}/v1/chat/completions}。
 * 请求体采用 ChatML 风格 messages 数组（仅一条 user message，整个 prompt 串在 content 里）。
 *
 * <h2>4. 错误处理</h2>
 * 任何异常 / 非 200 状态码都会被包成 {@code <error>...</error>} 字符串返回，
 * 这样 PicoAgent 的 {@code parse()} 能识别并立即终止循环，不会让 Agent 死锁。
 */
public class DeepSeekClient {

    private static final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    private static final ObjectMapper json = new ObjectMapper();

    private final String apiKey;
    private final String baseUrl;
    private final String model;

    public DeepSeekClient(String apiKey, String baseUrl, String model) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.model = model;
    }

    /**
     * 同步调一次 LLM，返回模型生成的文本。
     *
     * <h3>请求体构造</h3>
     * <pre>
     * {
     *   "model": "{model}",
     *   "max_tokens": {maxNewTokens},
     *   "temperature": 0.7,
     *   "messages": [
     *     { "role": "user", "content": "{prompt}" }
     *   ]
     * }
     * </pre>
     * <p>
     * 注意：整个 prompt（包括 system 规则 + 工具表 + 历史 + 当前请求）都被塞进单条 user message 的 content，
     * 这是 PicoAgent 决定的做法（见 PicoAgent.buildPrompt 注释）。
     *
     * <h3>响应解析</h3>
     * 从 {@code choices[0].message.content} 取文本；任何异常都返回 {@code <error>...</error>}。
     *
     * @param prompt       完整 prompt 文本
     * @param maxNewTokens 单次响应允许的最大 token 数
     * @return LLM 输出的文本（成功）/ {@code <error>...</error>} 字符串（失败）
     */
    public String complete(String prompt, int maxNewTokens) {
        try {
            ObjectNode body = json.createObjectNode();
            body.put("model", model);
            body.put("max_tokens", maxNewTokens);
            body.put("temperature", 0.7);
            ArrayNode messages = body.putArray("messages");
            ObjectNode msg = messages.addObject();
            msg.put("role", "user");
            msg.put("content", prompt);

            // 用字节方式发送，确保 UTF-8 编码
            byte[] jsonBytes = json.writeValueAsBytes(body);

            // 调试：打印实际发送的 prompt（截断）
            String debug = prompt.length() > 60 ? prompt.substring(0, 60) + "..." : prompt;
            System.out.println("--- [DeepSeek] sending " + jsonBytes.length + " bytes, prompt preview: " + debug + " ---");

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/chat/completions"))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json; charset=utf-8")
                    .timeout(Duration.ofSeconds(120))
                    .POST(HttpRequest.BodyPublishers.ofByteArray(jsonBytes))
                    .build();

            HttpResponse<byte[]> resp = http.send(req, HttpResponse.BodyHandlers.ofByteArray());
            if (resp.statusCode() != 200) {
                String errBody = new String(resp.body(), StandardCharsets.UTF_8);
                return "<error>LLM API error: HTTP " + resp.statusCode() + " - " + truncate(errBody, 200) + "</error>";
            }

            String respBody = new String(resp.body(), StandardCharsets.UTF_8);
            JsonNode root = json.readTree(respBody);
            String content = root.path("choices").get(0).path("message").path("content").asText("");
            return content != null ? content : "";

        } catch (Exception e) {
            return "<error>LLM call failed: " + e.getClass().getSimpleName() + " - " + e.getMessage() + "</error>";
        }
    }

    /** 错误响应截断（避免把超长 LLM 错误体打到终端和 history 里） */
    private static String truncate(String s, int max) {
        return s != null && s.length() > max ? s.substring(0, max) + "..." : s;
    }
}
