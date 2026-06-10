package org.ai.aihelper.ai.rag;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * DashScope Rerank 模型客户端
 * <p>
 * 支持两种模型及对应 API 格式：
 * - qwen3-rerank：使用 OpenAI 兼容格式 (compatible-api/v1/reranks)，query/documents 在顶层
 * - gte-rerank-v2：使用 DashScope 原生格式 (api/v1/services/rerank/text-rerank/text-rerank)，嵌套 input/parameters
 * <p>
 * API 文档: https://help.aliyun.com/zh/model-studio/developer-reference/text-rerank-api
 * <p>
 * 注意：gte-rerank-v2 将于 2026-05-30 下线，推荐使用 qwen3-rerank
 */
public class DashScopeRerankModel {

    // qwen3-rerank 使用 OpenAI 兼容格式
    private static final String COMPATIBLE_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks";
    // gte-rerank-v2 使用 DashScope 原生格式
    private static final String NATIVE_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank";

    private final String apiKey;
    private final String modelName;
    private final String apiUrl;
    private final boolean useCompatibleApi;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public DashScopeRerankModel(String apiKey, String modelName) {
        this.apiKey = apiKey;
        this.modelName = modelName;
        // qwen3-rerank 使用兼容格式，其他使用原生格式
        this.useCompatibleApi = modelName.startsWith("qwen3-rerank");
        this.apiUrl = useCompatibleApi ? COMPATIBLE_RERANK_URL : NATIVE_RERANK_URL;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * 对查询和文档列表进行重排序。
     *
     * <h3>双协议</h3>
     * 根据 modelName 自动选择请求格式：
     * <ul>
     *   <li>{@code qwen3-rerank*} → OpenAI 兼容格式（query/documents/top_n 在顶层）</li>
     *   <li>其他（含 {@code gte-rerank-v2}）→ DashScope 原生格式（嵌套 input/parameters）</li>
     * </ul>
     * 响应解析也对应分支处理（results 在顶层 vs output 下）。
     *
     * <h3>降级策略</h3>
     * API 调用失败 / 非 200 / 解析异常 → fallbackResults() 返回"原始顺序的前 N 条"，分数标记为 -1.0。
     * 调用方（RerankingContentRetriever）会把 -1.0 当作"不参与过滤"的标记。
     *
     * @param query     用户查询
     * @param documents 候选文档文本列表
     * @param topN      最终返回前 N 条
     * @return 排序后的结果（index 指向 documents 中的位置，按 score 降序）
     */
    public List<RerankResult> rerank(String query, List<String> documents, int topN) {
        if (documents.isEmpty()) {
            return List.of();
        }

        try {
            String requestBody = useCompatibleApi
                    ? buildCompatibleRequestBody(query, documents, topN)
                    : buildNativeRequestBody(query, documents, topN);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(apiUrl))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(Duration.ofSeconds(60))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                System.err.println("Rerank API 调用失败, status: " + response.statusCode() + ", body: " + response.body());
                return fallbackResults(documents, topN);
            }

            return parseResponse(response.body());

        } catch (Exception e) {
            System.err.println("Rerank API 调用异常: " + e.getMessage());
            return fallbackResults(documents, topN);
        }
    }

    /**
     * 构建 qwen3-rerank 的 OpenAI 兼容格式请求体
     * query/documents/top_n 在顶层
     */
    private String buildCompatibleRequestBody(String query, List<String> documents, int topN) throws Exception {
        ObjectNode root = objectMapper.createObjectNode();
        root.put("model", modelName);
        root.put("query", query);
        root.put("top_n", Math.min(topN, documents.size()));

        ArrayNode docsArray = root.putArray("documents");
        for (String doc : documents) {
            docsArray.add(doc);
        }

        return objectMapper.writeValueAsString(root);
    }

    /**
     * 构建 gte-rerank-v2 的 DashScope 原生格式请求体
     * query/documents 在 input 对象中，top_n 在 parameters 对象中
     */
    private String buildNativeRequestBody(String query, List<String> documents, int topN) throws Exception {
        ObjectNode root = objectMapper.createObjectNode();
        root.put("model", modelName);

        ObjectNode input = root.putObject("input");
        input.put("query", query);
        ArrayNode docsArray = input.putArray("documents");
        for (String doc : documents) {
            docsArray.add(doc);
        }

        ObjectNode parameters = root.putObject("parameters");
        parameters.put("top_n", Math.min(topN, documents.size()));
        parameters.put("return_documents", true);

        return objectMapper.writeValueAsString(root);
    }

    /**
     * 解析 API 响应
     * - qwen3-rerank：results 在顶层
     * - gte-rerank-v2：results 在 output 对象中
     */
    private List<RerankResult> parseResponse(String responseBody) throws Exception {
        JsonNode root = objectMapper.readTree(responseBody);

        // qwen3-rerank 兼容格式：results 在顶层
        // gte-rerank-v2 原生格式：results 在 output 下
        JsonNode results = useCompatibleApi
                ? root.path("results")
                : root.path("output").path("results");

        List<RerankResult> rerankResults = new ArrayList<>();
        for (JsonNode result : results) {
            int index = result.path("index").asInt();
            double score = result.path("relevance_score").asDouble();
            rerankResults.add(new RerankResult(index, score));
        }

        rerankResults.sort(Comparator.comparingDouble(RerankResult::score).reversed());

        System.out.println("Rerank API 调用成功 (" + modelName + "), 输入文档数 → 输出 " + rerankResults.size() + " 条");
        return rerankResults;
    }

    /**
     * 降级策略：当 rerank API 不可用时，保留原始顺序的前N条
     */
    private List<RerankResult> fallbackResults(List<String> documents, int topN) {
        System.out.println("Rerank 降级：使用原始向量检索顺序");
        List<RerankResult> results = new ArrayList<>();
        int limit = Math.min(topN, documents.size());
        for (int i = 0; i < limit; i++) {
            results.add(new RerankResult(i, -1.0)); // -1.0 标记为降级结果
        }
        return results;
    }

    public String getModelName() {
        return modelName;
    }

    /**
     * Rerank 结果记录
     *
     * @param index 原始文档列表中的索引
     * @param score Rerank 相关性分数
     */
    public record RerankResult(int index, double score) {
    }
}
