package org.ai.aihelper.ai.rag;

import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.content.retriever.ContentRetriever;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 两阶段检索器：先向量检索获取候选，再用 Rerank 模型精排
 * <p>
 * 第一阶段：EmbeddingStoreContentRetriever 向量召回（宽召回，数量多）
 * 第二阶段：DashScopeRerankModel 精排（精准排序，数量少）
 * <p>
 * 这种两阶段策略兼顾了召回率和精确率：
 * - 向量检索速度快，但语义匹配精度有限
 * - Rerank 使用交叉编码器，精度更高但速度慢
 * - 先宽后精，既不漏掉相关文档，又保证最相关的排在前面
 */
public class RerankingContentRetriever implements ContentRetriever {

    private final ContentRetriever delegate;            // 第一阶段：向量检索器
    private final DashScopeRerankModel rerankModel;     // 第二阶段：Rerank 模型
    private final int maxResults;                       // 最终返回的结果数
    private final double minScore;                      // 最低相关性分数阈值（Rerank 分数）
    private final boolean rerankEnabled;                // 是否启用 Rerank

    public RerankingContentRetriever(ContentRetriever delegate,
                                     DashScopeRerankModel rerankModel,
                                     int maxResults,
                                     double minScore,
                                     boolean rerankEnabled) {
        this.delegate = delegate;
        this.rerankModel = rerankModel;
        this.maxResults = maxResults;
        this.minScore = minScore;
        this.rerankEnabled = rerankEnabled;
    }

    /**
     * 两阶段检索主入口。
     *
     * <h3>执行流程</h3>
     * <ol>
     *   <li>delegate.retrieve() → 向量召回 N 条（宽召回）</li>
     *   <li>若未启用 Rerank / Rerank 调用失败 → 直接返回向量结果</li>
     *   <li>rerankModel.rerank() → 用交叉编码器对 N 条精排，取前 K 条</li>
     *   <li>把 Rerank 分数写入 ContentMetadata.RERANKED_SCORE 后返回</li>
     *   <li>兜底：精排后所有结果被阈值过滤掉 → 回退到向量召回的 top K</li>
     * </ol>
     */
    @Override
    public List<Content> retrieve(Query query) {
        // 第一阶段：向量检索，获取候选
        List<Content> candidates;
        try {
            candidates = delegate.retrieve(query);
        } catch (Exception e) {
            System.err.println("RAG: 向量检索异常: " + e.getMessage());
            return List.of();
        }

        if (candidates.isEmpty()) {
            System.out.println("RAG: 向量检索无结果，跳过 Rerank");
            return candidates;
        }

        // 如果未启用 Rerank，直接返回向量检索结果
        if (!rerankEnabled) {
            System.out.println("RAG: Rerank 未启用，直接返回向量检索结果 (" + candidates.size() + " 条)");
            return candidates;
        }

        // 第二阶段：Rerank 精排
        String queryText = query.text();
        List<String> docTexts = candidates.stream()
                .map(content -> content.textSegment().text())
                .toList();

        System.out.println("RAG: 开始 Rerank 精排, 候选 " + candidates.size() + " 条, 查询: " + queryText);

        List<DashScopeRerankModel.RerankResult> rerankResults;
        try {
            rerankResults = rerankModel.rerank(queryText, docTexts, maxResults);
        } catch (Exception e) {
            System.err.println("RAG: Rerank 精排异常，回退到向量检索结果: " + e.getMessage());
            return candidates;
        }

        // 构建 Rerank 后的结果
        List<Content> rerankedContents = new ArrayList<>();
        for (DashScopeRerankModel.RerankResult result : rerankResults) {
            // 跳过低于阈值的结果（-1.0 是降级标记，不过滤）
            if (result.score() >= 0 && result.score() < minScore) {
                continue;
            }

            // 找到对应的原始 Content，将 Rerank 分数存入 ContentMetadata
            Content original = candidates.get(result.index());

            // 合并原有 metadata，并写入 RERANKED_SCORE
            Map<ContentMetadata, Object> newMetadata = new HashMap<>(original.metadata());
            if (result.score() >= 0) {
                newMetadata.put(ContentMetadata.RERANKED_SCORE, result.score());
            }

            Content rerankedContent = Content.from(original.textSegment(), newMetadata);
            rerankedContents.add(rerankedContent);
        }

        // 兜底：如果 Rerank 后所有结果都被过滤掉了，回退到向量检索的 top 结果
        if (rerankedContents.isEmpty() && !candidates.isEmpty()) {
            int fallbackCount = Math.min(maxResults, candidates.size());
            System.out.println("RAG: Rerank 精排后所有结果被过滤，回退到向量检索前 " + fallbackCount + " 条");
            return candidates.subList(0, fallbackCount);
        }

        System.out.println("RAG: Rerank 完成, 候选 " + candidates.size() + " 条 → 精排后 " + rerankedContents.size() + " 条");

        return rerankedContents;
    }
}
