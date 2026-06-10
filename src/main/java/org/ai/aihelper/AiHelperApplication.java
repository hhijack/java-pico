package org.ai.aihelper;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 【Agent 架构 / 启动入口】Spring Boot 应用启动类。
 *
 * <h2>启动流程</h2>
 * <ol>
 *   <li>{@code SpringApplication.run(...)} 启动 Spring 容器</li>
 *   <li>容器根据 profile 决定加载哪些 Bean：
 *     <ul>
 *       <li><b>通用</b>：RagConfig（向量库 + 检索器 + Rerank）、CodeTool、WebSearchTool、CurrentTimeTool</li>
 *       <li><b>cli profile</b>：PicoCLI（终端 REPL 入口）</li>
 *     </ul>
 *   </li>
 *   <li>若是 cli profile，PicoCLI.run() 在 Spring 就绪后被回调，进入 REPL 循环</li>
 * </ol>
 *
 * <h2>常用启动命令</h2>
 * <pre>
 *   # CLI 模式（REPL）
 *   mvn spring-boot:run -Dspring-boot.run.profiles=cli
 *
 *   # 启动 Web 服务（如未来要加 HTTP 端点）
 *   mvn spring-boot:run
 * </pre>
 */
@SpringBootApplication
public class AiHelperApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiHelperApplication.class, args);
    }

}
