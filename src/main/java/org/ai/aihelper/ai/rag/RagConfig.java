package org.ai.aihelper.ai.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import org.ai.aihelper.ai.tools.CurrentTimeTool;
import org.ai.aihelper.ai.tools.WebSearchTool;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.List;

/**
 * 【Agent 架构 / RAG 配置】Spring 配置类，组装向量库 + 嵌入模型 + 检索器 + Rerank。
 *
 * <h2>1. 它在整个 Agent 体系里的位置</h2>
 * 提供两件事：
 * <ol>
 *   <li>把 {@link CurrentTimeTool}、{@link WebSearchTool} 注册成 Spring Bean，供 PicoAgent 注入</li>
 *   <li>构造并暴露一个 {@link ContentRetriever} Bean（PicoCLI 注入后包成 RagRetriever 交给 PicoAgent）</li>
 * </ol>
 *
 * <h2>2. RAG 启动流程</h2>
 * <pre>
 *   1) 读取 application.yml / .properties 里的 spring.datasource.* (Postgres 连接)
 *   2) 构造 PgVectorEmbeddingStore (向量库)
 *   3) FileSystemDocumentLoader 加载 src/main/resources/docs/pdf 下的所有 PDF
 *   4) 检查 Postgres 表 document_embeddings 是否为空
 *        ├─ 空 → EmbeddingStoreIngestor 切分 + 嵌入 + 入库
 *        └─ 非空 → 跳过（增量启动）
 *   5) 构造 Rerank 模型（可选，未配置 API Key 时降级为纯向量检索）
 *   6) 返回 ContentRetriever：
 *        ├─ 未启用 Rerank → EmbeddingStoreContentRetriever（纯向量）
 *        └─ 启用 Rerank   → RerankingContentRetriever（向量召回 + 精排）
 * </pre>
 *
 * <h2>3. 为什么是 Spring @Configuration 而不是硬编码？</h2>
 * 数据源 URL、API Key 等需要通过 application.yml 切换 profile（local / cli），
 * 借助 Spring 生态可以让 @Value 注入、profile 切换、配置热更新都"零成本"获得。
 */
@Configuration
public class RagConfig {

    @Bean
    public CurrentTimeTool currentTimeTool() {
        return new CurrentTimeTool();
    }

    @Bean
    public WebSearchTool webSearchTool() {
        return new WebSearchTool();
    }

    @Value("${spring.datasource.username}")
    private String dbUser;

    @Value("${spring.datasource.password}")
    private String dbPassword;

    @Value("${langchain4j.store.embedding.pgvector.table-name:document_embeddings}")
    private String tableName;

    @Value("${langchain4j.store.embedding.pgvector.dimension:1024}")
    private Integer dimension;

    @Value("${rerank.api-key:}")
    private String rerankApiKey;

    @Value("${rerank.model-name:gte-rerank-v2}")
    private String rerankModelName;

    @Value("${rerank.enabled:true}")
    private boolean rerankEnabled;

    @Value("${rerank.candidate-count:20}")
    private int rerankCandidateCount;

    @Value("${rerank.max-results:5}")
    private int rerankMaxResults;

    @Value("${rerank.min-score:0.3}")
    private double rerankMinScore;

    @Value("${rerank.embedding-min-score:0.5}")
    private double embeddingMinScore;

    // DashScope EmbeddingModel 会通过 Spring Boot Starter 自动装配

    @Bean
    public EmbeddingStore<TextSegment> embeddingStore() {
        return PgVectorEmbeddingStore.builder()
                .host("localhost")
                .port(5432)
                .user(dbUser)
                .password(dbPassword)
                .database("ai_helper_db")
                .table(tableName)
                .dimension(dimension)
                .createTable(true)
                .build();
    }

    /**
     * 构造 LangChain4j 抽象的 {@link ContentRetriever}。
     * <p>
     * 这是 Agent 知识检索的核心入口：
     * <ul>
     *   <li>Spring 自动注入 DashScope 起步器提供的 {@link EmbeddingModel} Bean（text-embedding-v4，1024 维）</li>
     *   <li>首次启动会触发 PDF 加载 + 向量化入库；后续启动直接复用</li>
     *   <li>两阶段检索可降级为单阶段（纯向量）</li>
     * </ul>
     *
     * @param embeddingModel Spring 容器里的 DashScope 嵌入模型
     * @param dataSource     Spring 容器里的 JDBC 数据源（用于查表判断是否需要重建索引）
     * @return 最终使用的 ContentRetriever（两阶段 or 单阶段）
     */
    @Bean
    public ContentRetriever contentRetriever(EmbeddingModel embeddingModel, DataSource dataSource) {
        System.out.println("\n=== RAG 初始化（使用 DashScope text-embedding-v4 模型 + Rerank 精排）===");
        System.out.println("EmbeddingModel: " + embeddingModel.getClass().getSimpleName());
        System.out.println("向量维度: " + dimension);

        EmbeddingStore<TextSegment> store = embeddingStore();

        List<Document> documents = FileSystemDocumentLoader.loadDocuments(
                "src/main/resources/docs/pdf",
                new ApachePdfBoxDocumentParser(true)
        );
        System.out.println("加载文档数量: " + documents.size());

        // 检查数据库是否已有数据
        boolean isEmpty = checkIfStoreIsEmpty(dataSource);

        if (isEmpty) {
            System.out.println("数据库为空，开始向量化...");
            EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                    .documentSplitter(new DocumentByParagraphSplitter(1000, 200))
                    .embeddingModel(embeddingModel)
                    .embeddingStore(store)
                    .build();

            ingestor.ingest(documents);
            System.out.println("向量化完成（使用 DashScope 模型）");
        } else {
            System.out.println("数据库已有数据，跳过向量化步骤");
        }

        // 初始化 Rerank 模型
        DashScopeRerankModel rerankModel = createRerankModel();

        // 第一阶段：向量检索（宽召回，获取更多候选）
        ContentRetriever embeddingRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(rerankModel != null ? rerankCandidateCount : 10)
                .minScore(rerankModel != null ? embeddingMinScore : 0.75)
                .build();

        // 如果 Rerank 未启用，直接返回向量检索结果
        if (rerankModel == null) {
            System.out.println("使用纯向量检索模式 (maxResults=10, minScore=0.75)");
            return embeddingRetriever;
        }

        // 两阶段检索：向量召回 + Rerank 精排
        System.out.println("使用两阶段检索: 向量召回 " + rerankCandidateCount + " 条 (minScore=" + embeddingMinScore + ") → Rerank 精排 " + rerankMaxResults + " 条 (minScore=" + rerankMinScore + ")");

        return new RerankingContentRetriever(
                embeddingRetriever,
                rerankModel,
                rerankMaxResults,
                rerankMinScore,
                rerankEnabled
        );
    }

    /**
     * 创建 Rerank 模型实例（非 Spring Bean，避免 null bean 问题）
     */
    private DashScopeRerankModel createRerankModel() {
        System.out.println("\n=== Rerank 模型初始化 ===");
        System.out.println("模型: " + rerankModelName);
        System.out.println("启用状态: " + rerankEnabled);

        if (!rerankEnabled || rerankApiKey == null || rerankApiKey.isBlank()) {
            System.out.println("Rerank 未启用或未配置 API Key，将使用纯向量检索");
            return null;
        }

        System.out.println("Rerank 模型初始化成功: " + rerankModelName);
        return new DashScopeRerankModel(rerankApiKey, rerankModelName);
    }

    /**
     * 检查向量数据库是否为空
     */
    private boolean checkIfStoreIsEmpty(DataSource dataSource) {
        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM " + tableName)) {
            if (rs.next()) {
                long count = rs.getLong(1);
                System.out.println("数据库当前记录数: " + count);
                return count == 0;
            }
        } catch (Exception e) {
            System.err.println("检查数据库状态失败: " + e.getMessage());
        }
        return true; // 默认认为空，安全起见
    }
}