package org.ai.aihelper.pico;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 【Agent 架构 / 持久化层】跨 run 的会话状态持久化。
 *
 * <p>管理 {@code .pico/session.json}，包含：
 * <ul>
 *   <li>{@code history} — 完整对话记录（user / assistant / tool）</li>
 *   <li>{@code memory} — 结构化记忆的 JSON 快照</li>
 *   <li>{@code last_run_id} — 最后一次 run 的 ID</li>
 * </ul>
 *
 * <p>每次 {@link PicoAgent#ask} 完成后保存一次，启动时加载恢复。
 * 与 {@link RunStore} 的定位区别：
 * <ul>
 *   <li>SessionStore → 恢复现场用，覆写保存</li>
 *   <li>RunStore → 审计复盘用，追加 + 不可篡改</li>
 * </ul>
 */
public class SessionStore {

    private static final ObjectMapper json = new ObjectMapper();

    private final Path sessionFile;

    public SessionStore(Path picoDir) {
        this.sessionFile = picoDir.resolve("session.json");
    }

    /** 保存当前会话（history + memory）到 session.json */
    public void save(List<Map<String, String>> history, SessionMemory memory, String lastRunId) {
        try {
            ObjectNode root = json.createObjectNode();
            root.put("saved_at", java.time.Instant.now().toString());
            if (lastRunId != null) root.put("last_run_id", lastRunId);

            // history 只保留最近 50 条，防止文件无限膨胀
            ArrayNode histArr = root.putArray("history");
            List<Map<String, String>> toSave = history.size() > 50
                    ? history.subList(history.size() - 50, history.size())
                    : history;
            for (Map<String, String> entry : toSave) {
                ObjectNode eObj = histArr.addObject();
                for (Map.Entry<String, String> kv : entry.entrySet()) {
                    eObj.put(kv.getKey(), kv.getValue());
                }
            }

            // memory — 复用 SessionMemory 自己的序列化
            String memJson = memory.serialize();
            root.set("memory", json.readTree(memJson));

            Files.createDirectories(sessionFile.getParent());
            json.writerWithDefaultPrettyPrinter().writeValue(sessionFile.toFile(), root);
        } catch (Exception e) {
            System.err.println("Warning: failed to save session: " + e.getMessage());
        }
    }

    /** 从 session.json 加载历史记录（返回 null 表示无文件或失败） */
    public List<Map<String, String>> loadHistory() {
        if (!Files.isRegularFile(sessionFile)) return null;
        try {
            String text = Files.readString(sessionFile, StandardCharsets.UTF_8);
            var root = json.readTree(text);
            var histArr = root.path("history");
            if (!histArr.isArray()) return null;

            List<Map<String, String>> history = new ArrayList<>();
            for (var item : histArr) {
                Map<String, String> map = new java.util.LinkedHashMap<>();
                item.fieldNames().forEachRemaining(k -> map.put(k, item.get(k).asText()));
                history.add(map);
            }
            return history;
        } catch (Exception e) {
            System.err.println("Warning: failed to load session history: " + e.getMessage());
            return null;
        }
    }

    /** 从 session.json 恢复记忆到 memory 对象 */
    public void loadMemory(SessionMemory memory) {
        if (!Files.isRegularFile(sessionFile)) return;
        try {
            String text = Files.readString(sessionFile, StandardCharsets.UTF_8);
            var root = json.readTree(text);
            var memNode = root.path("memory");
            if (!memNode.isObject()) return;
            memory.loadFrom(json.writerWithDefaultPrettyPrinter().writeValueAsString(memNode));
        } catch (Exception e) {
            System.err.println("Warning: failed to load session memory: " + e.getMessage());
        }
    }

    public String loadLastRunId() {
        if (!Files.isRegularFile(sessionFile)) return null;
        try {
            var root = json.readTree(Files.readString(sessionFile, StandardCharsets.UTF_8));
            return root.path("last_run_id").asText(null);
        } catch (Exception e) {
            return null;
        }
    }

    public Path getSessionFile() { return sessionFile; }
}
