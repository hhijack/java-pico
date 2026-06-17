package org.ai.aihelper.ai.tools;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 验证 CodeTool.readFile 的行号格式化。
 * LLM 靠行号识别读到哪了、下一步该读哪。
 */
class CodeToolTest {

    @TempDir
    Path tempDir;

    private CodeTool newCodeTool() {
        return new CodeTool(tempDir.toString());
    }

    @Test
    void readFileIncludesLineNumbers() throws Exception {
        // 写一个 5 行的测试文件
        Path testFile = tempDir.resolve("Test.java");
        Files.writeString(testFile, "line1\nline2\nline3\nline4\nline5\n",
                StandardCharsets.UTF_8);

        String result = newCodeTool().readFile("Test.java", 1, 5);

        // 格式应是 # 路径 + 行号
        assertTrue(result.startsWith("# "), "should start with # path");
        assertTrue(result.contains("   1: line1"), "line 1 with right-aligned number");
        assertTrue(result.contains("   5: line5"), "line 5 with right-aligned number");
    }

    @Test
    void readFilePartialRange() throws Exception {
        Path testFile = tempDir.resolve("data.txt");
        Files.writeString(testFile, "a\nb\nc\nd\ne\nf\ng\nh\n", StandardCharsets.UTF_8);

        String result = newCodeTool().readFile("data.txt", 3, 6);

        assertTrue(result.contains("   3: c"), "starts at line 3");
        assertTrue(result.contains("   6: f"), "ends at line 6");
        assertFalse(result.contains("   1:"), "no line 1");
        assertFalse(result.contains("   7:"), "no line 7");
    }

    @Test
    void readFileOutOfRange() throws Exception {
        Path testFile = tempDir.resolve("tiny.txt");
        Files.writeString(testFile, "only one line\n", StandardCharsets.UTF_8);

        String result = newCodeTool().readFile("tiny.txt", 10, 20);
        assertTrue(result.contains("空") || result.contains("超出范围"),
                "should report out of range");
    }

    @Test
    void pathTraversalBlocked() {
        CodeTool tool = newCodeTool();
        String result = tool.readFile("../etc/passwd", 1, 10);
        assertTrue(result.startsWith("error:") || result.startsWith("错误:"),
                "path traversal should be blocked");
    }
}
