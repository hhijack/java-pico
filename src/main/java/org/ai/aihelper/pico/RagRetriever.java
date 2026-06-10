package org.ai.aihelper.pico;

import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;

import java.util.List;

/**
 * 【Agent 架构 / 工具层】RAG 检索的 Agent 友好包装。
 *
 * <h2>1. 它在整个 Agent 体系里的位置</h2>
 * PicoAgent 的 {@code search_knowledge} 工具就是路由到本类。
 * <pre>
 *   PicoAgent.executeTool("search_knowledge", {query})
 *        │
 *        ▼
 *   RagRetriever.search(query)              ← 把结果格式化成文本返回
 *        │
 *        ▼
 *   ContentRetriever.retrieve(Query)        ← LangChain4j 抽象的检索器
 *        │
 *        ▼
 *   EmbeddingStoreContentRetriever (向量召回)
 *        │
 *        ▼
 *   RerankingContentRetriever (Rerank 精排)
 * </pre>
 *
 * <h2>2. 为什么需要"包装"层？</h2>
 * LangChain4j 的 {@link ContentRetriever#retrieve} 返回的是 {@code List<Content>}，
 * 包含 TextSegment 和 metadata；PicoAgent 需要的是"一段可直接拼进 prompt 的纯文本"。
 * 本类负责：
 * <ul>
 *   <li>把每个 Content 格式化成"--- Result N (source: xxx) [embedding=...] [rerank=...] ---" + 文本</li>
 *   <li>截断超过 1000 字符的片段（防 prompt 爆炸）</li>
 *   <li>空查询 / 异常统一返回 {@code "error: ..."} 字符串</li>
 * </ul>
 */
public class RagRetriever {

    /** 底层 LangChain4j 检索器（已含向量召回 + Rerank 精排，由 RagConfig 装配） */
    private final ContentRetriever retriever;

    public RagRetriever(ContentRetriever retriever) {
        this.retriever = retriever;
    }

    /**
     * 用一段查询去知识库检索，返回给 LLM 的可读文本。
     *
     * <h3>输出格式示例</h3>
     * <pre>
     * --- Result 1 (source: JVM.pdf) [embedding=0.8765] [rerank=0.9123] ---
     * (文档片段正文)
     *
     * --- Result 2 (source: JUC.pdf) [embedding=0.8120] [rerank=0.8541] ---
     * (文档片段正文)
     * </pre>
     *
     * @param queryText 用户查询（必填）
     * @return 格式化后的检索结果；空结果 / 异常也返回字符串而非抛异常
     */
    public String search(String queryText) {
        if (queryText == null || queryText.isBlank()) {
            return "error: query is empty";
        }
        try {
            List<Content> results = retriever.retrieve(Query.from(queryText));
            if (results == null || results.isEmpty()) {
                return "(knowledge base returned no results)";
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < results.size(); i++) {
                Content c = results.get(i);
                String text = c.textSegment().text();
                String file = c.textSegment().metadata().getString("file_name");
                Object embScore = c.metadata().get(ContentMetadata.SCORE);
                Object rerankScore = c.metadata().get(ContentMetadata.RERANKED_SCORE);

                sb.append("--- Result ").append(i + 1).append(" (source: ").append(file != null ? file : "unknown").append(")");
                if (embScore instanceof Number n) sb.append(" [embedding=").append(String.format("%.4f", n.doubleValue())).append("]");
                if (rerankScore instanceof Number n) sb.append(" [rerank=").append(String.format("%.4f", n.doubleValue())).append("]");
                sb.append(" ---\n");
                // 截断过长文本
                if (text.length() > 1000) text = text.substring(0, 1000) + "...";
                sb.append(text).append("\n\n");
            }
            return sb.toString().trim();
        } catch (Exception e) {
            return "error: RAG retrieval failed: " + e.getClass().getSimpleName() + " - " + e.getMessage();
        }
    }
}
