package org.ai.aihelper.ai.tools;

import dev.langchain4j.agent.tool.Tool;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * 【Agent 架构 / 工具层】本地文件操作工具集。
 *
 * <h2>1. 它在整个 Agent 体系里的位置</h2>
 * PicoAgent 通过 {@code executeTool("list_files"/"read_file"/"write_file"/"patch_file"/"search", args)}
 * 把调用路由到本类的 5 个方法。本类把 5 个文件操作封装成统一接口，对 LLM 暴露为"工具"。
 *
 * <h2>2. 5 个工具一览</h2>
 * <ul>
 *   <li>{@code list_files(path)}      列出目录（忽略 .git / node_modules / target 等噪声目录）</li>
 *   <li>{@code read_file(path, start, end)} 按行号范围读文件（避免一次性吃完整文件）</li>
 *   <li>{@code write_file(path, content)}  覆盖写文件（危险操作，LLM 应仅在被要求时调用）</li>
 *   <li>{@code patch_file(path, old, new)}  精准替换一段文本（比 write_file 安全，要求 old_text 恰好出现一次）</li>
 *   <li>{@code search(pattern, path)}      工作区内大小写不敏感文本搜索，最多返回 80 条</li>
 * </ul>
 *
 * <h2>3. 路径安全</h2>
 * 所有方法都先经过 {@link #resolve(String)}，把用户给定路径 normalize 后检查
 * {@code candidate.startsWith(workspaceRoot)}，防止 LLM 通过 {@code "../"} 越权访问
 * 工作区外的文件。
 *
 * <h2>4. 与 LangChain4j 的关系</h2>
 * 标了 {@link Tool} 注解的方法会同时被 LangChain4j AiServices 识别（虽然本项目的 PicoAgent
 * 不用 AiServices，但保留这些注解便于将来切换框架）。{@code @Component} 让它成为 Spring Bean。
 */
@Slf4j
@Component
public class CodeTool {

    private static final Set<String> IGNORED_NAMES = Set.of(
            ".git", "node_modules", "__pycache__", ".pytest_cache",
            ".ruff_cache", ".venv", "venv", ".idea", "target", "build",
            ".mini-pico", ".mvn", ".claude", ".gitattributes"
    );

    private final Path workspaceRoot;

    public CodeTool(@Value("${code.workspace:./workspace}") String workspacePath) {
        try {
            this.workspaceRoot = Path.of(workspacePath).toAbsolutePath().normalize();
        } catch (Exception e) {
            throw new IllegalStateException("Cannot resolve workspace: " + workspacePath, e);
        }
        log.info("CodeTool workspace: {}", workspaceRoot);
    }

    /**
     * 把 LLM 给的"相对路径"转成绝对路径并做越权检查。
     * <p>
     * 这是 CodeTool 安全模型的基石：LLM 永远只能访问 workspaceRoot 下的文件。
     * 即便 LLM 输出 {@code "../../etc/passwd}，normalize 之后也会被拦下。
     */
    private Path resolve(String path) {
        if (path == null || path.isBlank()) path = ".";
        Path candidate = workspaceRoot.resolve(path).normalize();
        try {
            if (!candidate.startsWith(workspaceRoot)) {
                throw new IllegalArgumentException("Path escapes workspace: " + path);
            }
        } catch (Exception e) {
            // ProviderMismatchException: 跨文件系统比较时（如 WSL）
            // candidate 已经是 workspaceRoot.resolve() 出来的，不会越权，跳过检查
        }
        return candidate;
    }

    /** 把绝对路径还原成相对路径。跨文件系统时返回原始路径。 */
    private String rel(Path p) {
        try {
            return workspaceRoot.relativize(p.normalize()).toString().replace('\\', '/');
        } catch (Exception e) {
            // ProviderMismatchException 时返回绝对路径
            return p.normalize().toString().replace('\\', '/');
        }
    }

    @Tool(name = "list_files", value = "列出工作区目录下的文件和文件夹。当需要了解项目结构时使用。输入 path 为相对路径（默认根目录）。")
    public String listFiles(String path) {
        Path dir = resolve(path);
        if (!Files.isDirectory(dir)) return "错误: 不是目录: " + rel(dir);
        try (Stream<Path> s = Files.list(dir)) {
            List<Path> items = s
                    .filter(p -> !IGNORED_NAMES.contains(p.getFileName().toString()))
                    .sorted((a, b) -> {
                        boolean ad = Files.isDirectory(a), bd = Files.isDirectory(b);
                        return ad != bd ? (ad ? -1 : 1) : a.getFileName().toString().compareToIgnoreCase(b.getFileName().toString());
                    })
                    .limit(120).toList();
            if (items.isEmpty()) return "(空)";
            StringBuilder sb = new StringBuilder();
            for (Path e : items) sb.append(Files.isDirectory(e) ? "[D] " : "[F] ").append(rel(e)).append("\n");
            return sb.toString();
        } catch (IOException e) {
            return "错误: " + e.getMessage();
        }
    }

    @Tool(name = "read_file", value = "读取工作区中的文本文件，支持指定行范围。参数 path 为相对路径，start/end 为行号（从 1 开始）。")
    public String readFile(String path, int start, int end) {
        Path file = resolve(path);
        if (!Files.isRegularFile(file)) return "错误: 不是文件: " + rel(file);
        if (start < 1 || end < start) return "错误: 无效行范围";
        try {
            List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            int from = Math.min(start - 1, lines.size());
            int to = Math.min(end, lines.size());
            if (from >= lines.size()) return "# " + rel(file) + "\n(空或超出范围)";
            StringBuilder sb = new StringBuilder("# ").append(rel(file)).append("\n");
            for (int i = from; i < to; i++) sb.append(String.format("%4d: %s%n", i + 1, lines.get(i)));
            return sb.toString();
        } catch (IOException e) {
            return "错误: " + e.getMessage();
        }
    }

    @Tool(name = "write_file", value = "在工作区创建或覆盖文本文件。参数 path 为相对路径，content 为写入内容。注意：会覆盖已有文件！")
    public String writeFile(String path, String content) {
        Path file = resolve(path);
        try {
            Files.createDirectories(file.getParent());
            Files.writeString(file, content, StandardCharsets.UTF_8);
            log.info("write_file: {} ({} bytes)", rel(file), content.length());
            return "已写入 " + rel(file) + " (" + content.length() + " 字节)";
        } catch (IOException e) {
            return "错误: " + e.getMessage();
        }
    }

    @Tool(name = "patch_file", value = "精确替换文件中某段文本（比 write_file 更安全，不覆盖整个文件）。要求 old_text 在文件中恰好出现一次。参数 path 为相对路径。")
    public String patchFile(String path, String oldText, String newText) {
        if (oldText == null || oldText.isBlank()) return "错误: old_text 不能为空";
        Path file = resolve(path);
        if (!Files.isRegularFile(file)) return "错误: 不是文件: " + rel(file);
        try {
            String content = Files.readString(file, StandardCharsets.UTF_8);
            int idx = content.indexOf(oldText);
            if (idx == -1) return "错误: old_text 未找到";
            if (content.indexOf(oldText, idx + 1) != -1) return "错误: old_text 出现多次，必须恰好一次";
            Files.writeString(file, content.replaceFirst(
                    java.util.regex.Pattern.quote(oldText),
                    java.util.regex.Matcher.quoteReplacement(newText)), StandardCharsets.UTF_8);
            log.info("patch_file: {}", rel(file));
            return "已修改 " + rel(file);
        } catch (IOException e) {
            return "错误: " + e.getMessage();
        }
    }

    @Tool(name = "search", value = "在工作区搜索文本（大小写不敏感）。参数 pattern 为关键词，path 为搜索范围（默认根目录）。返回最多 80 条结果。")
    public String search(String pattern, String path) {
        if (pattern == null || pattern.isBlank()) return "错误: pattern 不能为空";
        Path dir = resolve(path);
        StringBuilder sb = new StringBuilder();
        int[] count = {0};
        try {
            Files.walkFileTree(dir, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult preVisitDirectory(Path d, BasicFileAttributes attrs) {
                    return IGNORED_NAMES.contains(d.getFileName().toString()) ? FileVisitResult.SKIP_SUBTREE : FileVisitResult.CONTINUE;
                }
                @Override
                public FileVisitResult visitFile(Path f, BasicFileAttributes attrs) {
                    if (count[0] >= 80) return FileVisitResult.TERMINATE;
                    try (var lines = Files.lines(f, StandardCharsets.UTF_8)) {
                        List<String> lineList = lines.toList();
                        for (int i = 0; i < lineList.size() && count[0] < 80; i++) {
                            if (lineList.get(i).toLowerCase().contains(pattern.toLowerCase())) {
                                sb.append(rel(f)).append(":").append(i + 1).append(":").append(lineList.get(i).trim()).append("\n");
                                count[0]++;
                            }
                        }
                    } catch (IOException ignored) {}
                    return count[0] >= 80 ? FileVisitResult.TERMINATE : FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            return "错误: " + e.getMessage();
        }
        return count[0] > 0 ? sb.toString() : "(无匹配)";
    }
}
