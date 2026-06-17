package org.ai.aihelper.pico;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 【Agent 架构 / 审计层】单次 ask() 调用的运行记录器。
 *
 * <p>每次 {@link PicoAgent#ask} 产生一次 run，在 {@code .pico/runs/<run_id>/} 下生成：
 * <ul>
 *   <li>{@code trace.jsonl} — 事件流水（追加写），记录 prompt、模型调用、工具执行等全流程</li>
 *   <li>{@code report.json} — 最终报告（写一次），汇总结果和关键指标</li>
 * </ul>
 *
 * <p>session 级别的持久化由 {@link SessionStore} 管理，run 只记录单次调用的审计痕迹。
 */
public class RunStore {

    private static final ObjectMapper json = new ObjectMapper();
    private static final DateTimeFormatter RUN_ID_FMT = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");

    private final Path runsBaseDir;
    private Path currentRunDir;
    private String currentRunId;
    private long runStartTime;

    public RunStore(Path picoDir) {
        this.runsBaseDir = picoDir.resolve("runs");
    }

    /** 启动一次新的 run，创建目录并写入 run_started 事件 */
    public String startRun(String userRequest) {
        currentRunId = "run_" + LocalDateTime.now().format(RUN_ID_FMT);
        currentRunDir = runsBaseDir.resolve(currentRunId);
        runStartTime = System.currentTimeMillis();

        try {
            Files.createDirectories(currentRunDir);
        } catch (Exception e) {
            System.err.println("Warning: failed to create run dir: " + e.getMessage());
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("user_request", truncate(userRequest, 300));
        emit("run_started", payload);

        return currentRunId;
    }

    /** 追加一条 trace 事件到 trace.jsonl */
    public void emit(String event, Map<String, Object> payload) {
        if (currentRunDir == null) return;
        try {
            ObjectNode root = json.createObjectNode();
            root.put("event", event);
            root.put("created_at", java.time.ZonedDateTime.now(java.time.ZoneId.of("Asia/Shanghai")).toString());
            if (payload != null) {
                for (Map.Entry<String, Object> e : payload.entrySet()) {
                    Object v = e.getValue();
                    if (v instanceof String s) root.put(e.getKey(), s);
                    else if (v instanceof Integer i) root.put(e.getKey(), i);
                    else if (v instanceof Long l) root.put(e.getKey(), l);
                    else if (v instanceof Boolean b) root.put(e.getKey(), b);
                    else if (v != null) root.put(e.getKey(), v.toString());
                }
            }
            Path traceFile = currentRunDir.resolve("trace.jsonl");
            Files.writeString(traceFile,
                    json.writeValueAsString(root) + "\n",
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (Exception e) {
            System.err.println("Warning: failed to write trace: " + e.getMessage());
        }
    }

    /** 快速写入事件（无 payload） */
    public void emit(String event) {
        emit(event, null);
    }

    /** run 结束时写入 report.json */
    public void writeReport(String status, String stopReason, String finalAnswer,
                            int toolSteps, int attempts, String modelName) {
        if (currentRunDir == null) return;
        try {
            ObjectNode root = json.createObjectNode();
            root.put("run_id", currentRunId);
            root.put("status", status);
            root.put("stop_reason", stopReason != null ? stopReason : "");
            root.put("final_answer", truncate(finalAnswer, 500));
            root.put("tool_steps", toolSteps);
            root.put("attempts", attempts);
            root.put("model", modelName != null ? modelName : "");
            root.put("run_duration_ms", System.currentTimeMillis() - runStartTime);

            Path reportFile = currentRunDir.resolve("report.json");
            json.writerWithDefaultPrettyPrinter().writeValue(reportFile.toFile(), root);
        } catch (Exception e) {
            System.err.println("Warning: failed to write report: " + e.getMessage());
        }
    }

    public String getCurrentRunId() { return currentRunId; }

    private static String truncate(String s, int max) {
        return s != null && s.length() > max ? s.substring(0, max) + "..." : s;
    }
}
