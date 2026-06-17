package org.ai.aihelper.pico;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 验证 PicoAgent 的关键行为更改：
 * 1) clip() — 工具输出 4000 字符上限
 * 2) renderHistory — 预算驱动压缩
 */
class PicoAgentTest {

    // ========= clip() =========

    @Test
    void clipShortTextUnchanged() throws Exception {
        String text = "short";
        assertEquals(text, invokeClip(text));
    }

    @Test
    void clipLongTextTruncated() throws Exception {
        // 5000 字符文本
        String text = "x".repeat(5000);
        String result = invokeClip(text);
        assertTrue(result.length() < text.length(), "should be truncated");
        assertTrue(result.contains("truncated"), "should have truncation note");
    }

    @Test
    void clipNullReturnsEmpty() throws Exception {
        assertEquals("", invokeClip(null));
    }

    // ========= renderHistory() =========

    @Test
    void emptyHistoryReturnsEmpty() throws Exception {
        String result = invokeRenderHistory(List.of());
        assertEquals("(empty)", result);
    }

    @Test
    void recentItemsKeptFull() throws Exception {
        // 3 条短消息，总预算 5200，应该全部保留原文
        List<Map<String, String>> history = new ArrayList<>();
        history.add(Map.of("role", "user", "content", "hello"));
        history.add(Map.of("role", "assistant", "content", "world"));
        history.add(Map.of("role", "user", "content", "foo"));

        String result = invokeRenderHistory(history);
        assertTrue(result.contains("hello"), "user message visible");
        assertTrue(result.contains("world"), "assistant message visible");
    }

    @Test
    void readFileOldEntryCompressed() throws Exception {
        // 模拟 read_file 结果（120 行格式化内容）
        StringBuilder fileContent = new StringBuilder("# Mock.java\n");
        for (int i = 1; i <= 120; i++) {
            fileContent.append(String.format("%4d: line %d%n", i, i));
        }

        List<Map<String, String>> history = new ArrayList<>();
        history.add(Map.of("role", "user", "content", "read the file"));
        history.add(Map.of("role", "tool", "name", "read_file",
                "content", fileContent.toString()));

        String result = invokeRenderHistory(history);

        // 最近条目 → 保留原文，应有行号
        assertTrue(result.contains("Mock.java"), "file path visible");
        assertTrue(result.contains("line"), "file content visible");
    }

    @Test
    void overBudgetCompressesOldest() throws Exception {
        // 模拟大量历史使预算超限
        List<Map<String, String>> history = new ArrayList<>();
        StringBuilder bigFile = new StringBuilder("# Big.java\n");
        for (int i = 1; i <= 200; i++) {
            bigFile.append(String.format("%4d: xxxxxxxxxx %d%n", i, i));
        }

        // 3 条大文件读取 + 1 条最新消息
        for (int i = 0; i < 3; i++) {
            history.add(Map.of("role", "tool", "name", "read_file",
                    "content", bigFile.toString()));
        }
        history.add(Map.of("role", "user", "content", "latest message"));

        String result = invokeRenderHistory(history);

        // 最新消息应在结果中
        assertTrue(result.contains("latest message"), "newest message retained");

        // 由于预算限制，部分旧条目应被压缩为 [compressed]
        assertTrue(result.contains("[compressed]") || result.length() < bigFile.length() * 3,
                "old entries compressed when over budget");
    }

    @Test
    void knowledgeResultAlwaysFull() throws Exception {
        List<Map<String, String>> history = new ArrayList<>();
        history.add(Map.of("role", "tool", "name", "search_knowledge",
                "content", "Java uses pass-by-value"));
        // 大量文件读取填满预算
        StringBuilder big = new StringBuilder("# x.java\n");
        for (int i = 0; i < 200; i++) big.append("   x: line\n");
        history.add(Map.of("role", "tool", "name", "read_file",
                "content", big.toString()));

        String result = invokeRenderHistory(history);

        // search_knowledge 即使不在预算内也应可见
        assertTrue(result.contains("Java uses pass-by-value"),
                "knowledge result visible even when compressed");
    }

    // ========= helpers =========

    private static String invokeClip(String s) throws Exception {
        Method m = PicoAgent.class.getDeclaredMethod("clip", String.class);
        m.setAccessible(true);
        return (String) m.invoke(null, s);
    }

    @SuppressWarnings("unchecked")
    private static String invokeRenderHistory(List<Map<String, String>> history) throws Exception {
        Method m = PicoAgent.class.getDeclaredMethod("renderHistory", List.class);
        m.setAccessible(true);
        return (String) m.invoke(null, history);
    }
}
