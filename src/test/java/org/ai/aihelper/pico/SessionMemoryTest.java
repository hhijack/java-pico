package org.ai.aihelper.pico;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 验证 SessionMemory.summarizeContent 的截断行为。
 * 这个方法是 memory 层用的（不是 tool result），
 * 确保它仍产出 head + tail 格式的摘要供后续历史压缩复用。
 */
class SessionMemoryTest {

    @Test
    void shortContentReturnsFull() {
        String shortContent = "package foo;\npublic class Foo {}";
        String result = SessionMemory.summarizeContent(shortContent);
        assertEquals(shortContent, result);
    }

    @Test
    void emptyReturnsPlaceholder() {
        assertEquals("(empty)", SessionMemory.summarizeContent(""));
        assertEquals("(empty)", SessionMemory.summarizeContent("   "));
        assertEquals("(empty)", SessionMemory.summarizeContent(null));
    }

    @Test
    void longContentHasHeadAndTail() {
        // 生成 50 行内容，每行 ~30 字符，总长 > 1000
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= 50; i++) {
            sb.append(String.format("line_%02d: " + "x".repeat(25) + "%n", i));
        }
        String content = sb.toString();

        String result = SessionMemory.summarizeContent(content);

        // 摘要应包含头行
        assertTrue(result.contains("line_01:"), "should contain head");
        // 摘要应包含尾行
        assertTrue(result.contains("line_50:"), "should contain tail");
        // 摘要应包含省略标记
        assertTrue(result.contains("more lines") || result.contains("..."), "should have truncation marker");

        // 摘要应比原始内容短很多
        assertTrue(result.length() < content.length(), "summary shorter than original");
    }

    @Test
    void summaryAlwaysShorterThanOriginal() {
        // 确保即使对于各种 size，摘要都不膨胀
        for (int lines = 10; lines <= 200; lines += 30) {
            StringBuilder sb = new StringBuilder();
            for (int i = 1; i <= lines; i++) {
                sb.append(String.format("line_%03d: " + "y".repeat(20) + "%n", i));
            }
            String content = sb.toString();
            String result = SessionMemory.summarizeContent(content);
            assertTrue(result.length() < content.length(),
                    "summary shorter for " + lines + " lines");
        }
    }
}
