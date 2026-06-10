package org.ai.aihelper.ai.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * 【Agent 架构 / 工具层】获取当前时间的小工具。
 *
 * <h2>1. 它在整个 Agent 体系里的位置</h2>
 * 标了 {@link Tool} 注解，本来期望被 PicoAgent 路由使用。
 * <b>当前 PicoAgent.executeTool 没有 dispatch "get_current_time"，</b>
 * 所以它暂时没被主 Agent 循环调用，但保留作为可扩展工具示例。
 *
 * <h2>2. 为什么需要这个工具？</h2>
 * LLM 的训练数据有截止时间，对"现在几点""今天几号"无能为力。
 * 显式提供一个时间工具，让 LLM 能在需要时拿到权威当前时间。
 */
public class CurrentTimeTool {

    @Tool(name = "get_current_time", value = """
            获取当前的日期和时间。
            当用户询问当前时间、今天几号、现在几点等问题时调用这个工具。
            """)
    public String getCurrentTime() {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH:mm:ss");
        String result = "当前时间: " + now.format(formatter);
        System.out.println("\n========== TOOL RESULT ==========");
        System.out.println("Tool: get_current_time");
        System.out.println("Result: " + result);
        System.out.println("=================================\n");
        return result;
    }
}