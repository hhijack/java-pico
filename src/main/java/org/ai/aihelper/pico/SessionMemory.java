package org.ai.aihelper.pico;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 【Agent 架构 / 记忆层】结构化多层会话记忆管理。
 *
 * <h2>四层记忆结构</h2>
 * <ol>
 *   <li><b>working.task_summary</b> — 当前任务摘要（历史裁剪后仍保留的锚点）</li>
 *   <li><b>working.recent_files</b> — 最近访问的文件列表（去重 + 上限）</li>
 *   <li><b>file_summaries</b> — 文件内容摘要 + 新鲜度追踪（内容 MD5 哈希）</li>
 *   <li><b>episodic_notes</b> — 跨轮笔记（标签命中 + 关键词重叠 + 时间排序召回）</li>
 * </ol>
 *
 * <h2>设计要点</h2>
 * <ul>
 *   <li>文件摘要带 freshness（内容哈希），文件被修改后旧摘要自动失效</li>
 *   <li>笔记按标签/关键词召回，非 embedding 方案，零依赖</li>
 *   <li>工具执行链路自动触发记忆写入（read_file → 摘要；write/patch → 失效）</li>
 *   <li>支持 JSON 序列化/反序列化，可持久化到磁盘</li>
 * </ul>
 */
public class SessionMemory {

    private static final int WORKING_FILE_LIMIT = 5;
    private static final int NOTE_LIMIT = 50;
    private static final int SUMMARY_CHARS = 300;

    // ===================================================================
    //  Layer 1 & 2: Working Memory
    // ===================================================================
    private String taskSummary = "";
    private final List<String> recentFiles = new ArrayList<>();

    // ===================================================================
    //  Layer 3: File Summaries with Freshness Tracking
    // ===================================================================
    private final Map<String, FileSummaryEntry> fileSummaries = new LinkedHashMap<>();

    // ===================================================================
    //  Layer 4: Episodic Notes (tagged, retrievable)
    // ===================================================================
    private final List<NoteEntry> episodicNotes = new ArrayList<>();
    private int nextNoteIndex = 0;

    // ===================================================================
    //  Inner Types
    // ===================================================================
    static class FileSummaryEntry {
        String summary;
        long createdAt;
        String freshness; // MD5 hex of file content

        FileSummaryEntry() {}

        FileSummaryEntry(String summary, String freshness) {
            this.summary = summary;
            this.createdAt = System.currentTimeMillis();
            this.freshness = freshness;
        }
    }

    static class NoteEntry {
        int index;
        String content;
        Set<String> tags;
        long createdAt;
        String source;

        NoteEntry() {}

        NoteEntry(int index, String content, Set<String> tags, String source) {
            this.index = index;
            this.content = content;
            this.tags = new HashSet<>(tags);
            this.createdAt = System.currentTimeMillis();
            this.source = source;
        }
    }

    // ===================================================================
    //  Layer 1: Task Summary
    // ===================================================================

    /** 设置当前任务摘要（取用户消息的第一行/前 100 字符） */
    public void setTaskSummary(String summary) {
        if (summary == null || summary.isBlank()) return;
        int idx = summary.indexOf('\n');
        this.taskSummary = idx > 0 ? summary.substring(0, idx) : summary;
        if (this.taskSummary.length() > 100) {
            this.taskSummary = this.taskSummary.substring(0, 100) + "...";
        }
    }

    public String getTaskSummary() {
        return taskSummary;
    }

    // ===================================================================
    //  Layer 2: Recent Files
    // ===================================================================

    /** 记录文件访问（自动去重，保留最近 N 个） */
    public void rememberFile(String path) {
        if (path == null || path.isBlank()) return;
        recentFiles.remove(path);
        recentFiles.add(path);
        if (recentFiles.size() > WORKING_FILE_LIMIT) {
            recentFiles.removeFirst();
        }
    }

    public List<String> getRecentFiles() {
        return List.copyOf(recentFiles);
    }

    // ===================================================================
    //  Layer 3: File Summaries with Freshness
    // ===================================================================

    /** 写入文件摘要并记录当前内容的 MD5 作为新鲜度标记 */
    public void setFileSummary(String path, String summary) {
        if (path == null || summary == null) return;
        String freshness = computeFreshness(path);
        fileSummaries.put(path, new FileSummaryEntry(summary, freshness));
    }

    /** 使文件摘要失效（write_file / patch_file 后调用） */
    public void invalidateFileSummary(String path) {
        if (path == null) return;
        FileSummaryEntry entry = fileSummaries.get(path);
        if (entry != null) {
            entry.freshness = "";
        }
    }

    /** 获取文件摘要（已过期返回空字符串） */
    public String getFileSummary(String path) {
        if (path == null) return "";
        FileSummaryEntry entry = fileSummaries.get(path);
        if (entry == null || entry.summary.isEmpty()) return "";
        String currentFreshness = computeFreshness(path);
        if (!entry.freshness.equals(currentFreshness) || currentFreshness.isEmpty()) {
            return "";
        }
        return entry.summary;
    }

    // ===================================================================
    //  Layer 4: Episodic Notes
    // ===================================================================

    /** 追加一条笔记（自动管理索引和上限） */
    public void appendNote(String content, Set<String> tags, String source) {
        if (content == null || content.isBlank()) return;
        episodicNotes.add(new NoteEntry(nextNoteIndex++, content,
                tags != null ? tags : Set.of(), source));
        if (episodicNotes.size() > NOTE_LIMIT) {
            episodicNotes.removeFirst();
        }
    }

    /**
     * 按 query 召回相关笔记。
     * 排序规则：标签精确命中 > 关键词重叠 > 时间新旧。
     * 纯文本匹配，零 embedding 依赖。
     */
    public List<NoteEntry> retrievalCandidates(String query, int limit) {
        if (episodicNotes.isEmpty() || query == null || query.isBlank()) return List.of();

        Set<String> queryTokens = tokenize(query);
        List<ScoredNote> scored = new ArrayList<>();

        for (NoteEntry note : episodicNotes) {
            Set<String> noteTokens = tokenize(note.content);
            Set<String> noteTags = note.tags.stream()
                    .map(t -> t.toLowerCase(Locale.ROOT))
                    .collect(Collectors.toSet());

            int tagMatch = 0;
            for (String qt : queryTokens) {
                if (noteTags.contains(qt)) tagMatch++;
            }

            int overlap = 0;
            for (String qt : queryTokens) {
                if (noteTokens.contains(qt)) overlap++;
            }

            double score = tagMatch * 10.0 + overlap * 2.0 + note.createdAt / 1e12;
            scored.add(new ScoredNote(score, note));
        }

        scored.sort((a, b) -> Double.compare(b.score, a.score));
        return scored.subList(0, Math.min(limit, scored.size())).stream()
                .map(sn -> sn.note)
                .collect(Collectors.toList());
    }

    private record ScoredNote(double score, NoteEntry note) {}

    // ===================================================================
    //  Render for Prompt
    // ===================================================================

    /**
     * 渲染"稳定仪表盘"部分（当前任务 + 最近文件 + 有效文件摘要 + 笔记数量）。
     * 每次调用都会重新校验文件摘要的新鲜度。
     */
    public String renderMemoryText(Path workspaceRoot) {
        StringBuilder sb = new StringBuilder();
        boolean hasContent = false;

        if (!taskSummary.isEmpty()) {
            sb.append("=== Current Task ===\n").append(taskSummary).append("\n\n");
            hasContent = true;
        }

        if (!recentFiles.isEmpty()) {
            sb.append("=== Recent Files ===\n");
            for (String f : recentFiles) {
                sb.append("- ").append(f).append("\n");
            }
            sb.append("\n");
            hasContent = true;
        }

        // Fresh file summaries — re-validate freshness on every render
        List<String> freshSummaries = new ArrayList<>();
        for (Map.Entry<String, FileSummaryEntry> e : fileSummaries.entrySet()) {
            String path = e.getKey();
            FileSummaryEntry entry = e.getValue();
            if (entry.summary.isEmpty()) continue;

            String currentFreshness = computeFreshness(path);
            if (entry.freshness.equals(currentFreshness) && !currentFreshness.isEmpty()) {
                freshSummaries.add("- " + path + ": " + truncate(entry.summary, 150));
            }
        }
        if (!freshSummaries.isEmpty()) {
            sb.append("=== File Summaries ===\n");
            for (String s : freshSummaries) {
                sb.append(s).append("\n");
            }
            sb.append("\n");
            hasContent = true;
        }

        if (!episodicNotes.isEmpty()) {
            sb.append("=== Notes ===\n")
                    .append("(").append(episodicNotes.size()).append(" notes available)\n\n");
            hasContent = true;
        }

        return hasContent ? sb.toString() : "";
    }

    /**
     * 按当前 query 召回相关笔记文本块（用于注入 prompt）。
     * 召回 top 3，按分数排序。
     */
    public String renderRelevantMemory(String query) {
        List<NoteEntry> relevant = retrievalCandidates(query, 3);
        if (relevant.isEmpty()) return "";

        StringBuilder sb = new StringBuilder("=== Relevant Memory ===\n");
        for (NoteEntry note : relevant) {
            sb.append("- [").append(note.source != null ? note.source : "note").append("] ")
                    .append(truncate(note.content, 200)).append("\n");
        }
        return sb.toString();
    }

    // ===================================================================
    //  Persistence
    // ===================================================================

    /** 序列化为 JSON 字符串 */
    public String serialize() {
        try {
            ObjectMapper mapper = new ObjectMapper();
            ObjectNode root = mapper.createObjectNode();

            root.put("taskSummary", taskSummary);

            ArrayNode filesArr = root.putArray("recentFiles");
            for (String f : recentFiles) filesArr.add(f);

            ObjectNode fsObj = root.putObject("fileSummaries");
            for (Map.Entry<String, FileSummaryEntry> e : fileSummaries.entrySet()) {
                ObjectNode entryObj = fsObj.putObject(e.getKey());
                entryObj.put("summary", e.getValue().summary);
                entryObj.put("freshness", e.getValue().freshness);
                entryObj.put("createdAt", e.getValue().createdAt);
            }

            ArrayNode notesArr = root.putArray("episodicNotes");
            for (NoteEntry n : episodicNotes) {
                ObjectNode nObj = notesArr.addObject();
                nObj.put("index", n.index);
                nObj.put("content", n.content);
                nObj.put("createdAt", n.createdAt);
                nObj.put("source", n.source != null ? n.source : "");
                ArrayNode tagsArr = nObj.putArray("tags");
                for (String t : n.tags) tagsArr.add(t);
            }
            root.put("nextNoteIndex", nextNoteIndex);

            return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(root);
        } catch (Exception e) {
            return "{}";
        }
    }

    /** 从 JSON 字符串反序列化 */
    public static SessionMemory deserialize(String json) {
        SessionMemory mem = new SessionMemory();
        if (json == null || json.isBlank() || json.equals("{}")) return mem;

        try {
            ObjectMapper mapper = new ObjectMapper();
            var root = mapper.readTree(json);

            mem.taskSummary = root.path("taskSummary").asText("");

            var filesArr = root.path("recentFiles");
            if (filesArr.isArray()) {
                for (var n : filesArr) mem.recentFiles.add(n.asText());
            }

            var fsObj = root.path("fileSummaries");
            if (fsObj.isObject()) {
                fsObj.fieldNames().forEachRemaining(key -> {
                    var entry = fsObj.get(key);
                    FileSummaryEntry fse = new FileSummaryEntry();
                    fse.summary = entry.path("summary").asText("");
                    fse.freshness = entry.path("freshness").asText("");
                    fse.createdAt = entry.path("createdAt").asLong(0);
                    mem.fileSummaries.put(key, fse);
                });
            }

            var notesArr = root.path("episodicNotes");
            if (notesArr.isArray()) {
                for (var n : notesArr) {
                    NoteEntry ne = new NoteEntry();
                    ne.index = n.path("index").asInt(0);
                    ne.content = n.path("content").asText("");
                    ne.createdAt = n.path("createdAt").asLong(0);
                    ne.source = n.path("source").asText("");
                    ne.tags = new HashSet<>();
                    var tagsArr = n.path("tags");
                    if (tagsArr.isArray()) {
                        for (var t : tagsArr) ne.tags.add(t.asText());
                    }
                    mem.episodicNotes.add(ne);
                }
            }
            mem.nextNoteIndex = root.path("nextNoteIndex").asInt(0);
        } catch (Exception ignored) {}

        return mem;
    }

    /** 从 JSON 字符串加载到此实例（用于 PicoCLI 启动时恢复记忆） */
    public void loadFrom(String json) {
        SessionMemory loaded = deserialize(json);
        this.taskSummary = loaded.taskSummary;
        this.recentFiles.clear();
        this.recentFiles.addAll(loaded.recentFiles);
        this.fileSummaries.clear();
        this.fileSummaries.putAll(loaded.fileSummaries);
        this.episodicNotes.clear();
        this.episodicNotes.addAll(loaded.episodicNotes);
        this.nextNoteIndex = loaded.nextNoteIndex;
    }

    // ===================================================================
    //  File Content Summarization
    // ===================================================================

    /**
     * 把读取到的文件内容压成短摘要（用于 file_summaries 存储）。
     * 非 LLM 方案：提取首行 + 统计信息 + 尾行。
     */
    public static String summarizeContent(String content) {
        if (content == null || content.isBlank()) return "(empty)";
        String trimmed = content.trim();
        if (trimmed.length() <= SUMMARY_CHARS) return trimmed;

        int nl = trimmed.indexOf('\n');
        String firstLine = nl > 0 ? trimmed.substring(0, nl).trim() : "";
        int lines = trimmed.split("\n", -1).length;

        StringBuilder sb = new StringBuilder();
        if (!firstLine.isEmpty()) sb.append(firstLine).append("\n");
        sb.append("(").append(lines).append(" lines, ").append(trimmed.length()).append(" chars)");

        String result = sb.toString();
        return result.length() <= SUMMARY_CHARS ? result : result.substring(0, SUMMARY_CHARS) + "...";
    }

    // ===================================================================
    //  Helpers
    // ===================================================================

    /** 计算文件内容的 MD5 哈希（作为新鲜度标记） */
    private static String computeFreshness(String filePath) {
        if (filePath == null || filePath.isEmpty()) return "";
        try {
            Path p = Path.of(filePath);
            if (!Files.isRegularFile(p)) return "";
            byte[] content = Files.readAllBytes(p);
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(content);
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) sb.append(String.format("%02x", b));
            return sb.toString();
        } catch (Exception e) {
            return "";
        }
    }

    /** 文本分词（英文单词 + 中文词组，>=2 字符） */
    private static Set<String> tokenize(String text) {
        if (text == null || text.isBlank()) return Set.of();
        String lower = text.toLowerCase(Locale.ROOT);
        String[] parts = lower.split("[^a-z0-9一-鿿]+");
        Set<String> tokens = new HashSet<>();
        for (String p : parts) {
            if (p.length() >= 2) tokens.add(p);
        }
        return tokens;
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() > max ? s.substring(0, max) + "..." : s;
    }
}
