package org.ai.aihelper.ai.tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.Tool;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

/**
 * 【Agent 架构 / 工具层】网页搜索工具（封装 Tavily Search API）。
 *
 * <h2>1. 它在整个 Agent 体系里的位置</h2>
 * PicoAgent 通过 {@code executeTool("web_search", {query})} 把调用路由到本类的 {@link #search}。
 * 当用户问"今天新闻"、"最新股价"、"某个工具的版本"等需要实时信息的问题时，LLM 会主动调它。
 *
 * <h2>2. 与 search_knowledge 的差异</h2>
 * <ul>
 *   <li>{@code search_knowledge} → 查本地向量库（已切分好的 Java 面试 PDF）</li>
 *   <li>{@code web_search}        → 查互联网（Tavily API，结果含 url）</li>
 * </ul>
 *
 * <h2>3. 为什么用 Tavily？</h2>
 * Tavily 是面向 LLM 的搜索 API：返回的 snippet 已经做了去噪和摘要，更适合直接喂给 LLM。
 * 替代品（SerpAPI / Bing Search）需要自己做网页抓取 + 摘要。
 */
@Slf4j
@Component
public class WebSearchTool {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${search.api-key:}")
    private String apiKey;

    @Tool(name = "web_search", value = """
            搜索网页获取信息。
            当用户询问实时新闻、最新资讯、天气、股价等需要最新信息的问题时使用。
            输入是搜索关键词，返回搜索结果摘要。
            """)
    public String search(String query) {
        try {
            log.info("执行网页搜索: {}", query);

            String jsonBody = """
                {
                    "query": "%s",
                    "max_results": 5,
                    "search_depth": "basic"
                }
                """.formatted(query);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("https://api.tavily.com/search"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            JsonNode root = objectMapper.readTree(response.body());
            JsonNode results = root.get("results");

            StringBuilder sb = new StringBuilder();
            if (results != null && results.isArray()) {
                for (int i = 0; i < Math.min(5, results.size()); i++) {
                    JsonNode item = results.get(i);
                    String title = item.get("title").asText();
                    String snippet = item.get("content").asText();
                    String url = item.has("url") ? item.get("url").asText() : "";
                    sb.append(i + 1).append(". ").append(title).append("\n").append(snippet);
                    if (!url.isEmpty()) {
                        sb.append("\n🔗 ").append(url);
                    }
                    sb.append("\n\n");
                }
            }

            String result = sb.length() > 0 ? sb.toString() : "未找到相关结果";
            System.out.println("\n========== TOOL RESULT ==========");
            System.out.println("Tool: web_search");
            System.out.println("Query: " + query);
            System.out.println("Results: " + result);
            System.out.println("=================================\n");
            return result;
        } catch (Exception e) {
            log.error("搜索失败: {} - {}", e.getClass().getName(), e.getMessage(), e);
            return "搜索失败: " + e.getMessage();
        }
    }
}