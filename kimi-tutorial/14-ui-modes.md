# 第 14 章：UI 模式

同一个 Agent，不同的使用场景。

- 👨‍💻 开发者：想要**命令行交互**
- 🤖 CI/CD：需要**脚本化执行**
- 💻 IDE 用户：希望**编辑器集成**

一个好的 Agent 应该支持多种 UI 模式。kimi-cli 支持 4 种：Shell、Print、ACP、Wire。

## 14.1 为什么需要多种 UI 模式？

### 场景 1：本地开发调试

你正在开发一个新功能，希望与 Agent 交互式对话：

```bash
$ kimi
> 你: 帮我分析这个错误日志
Agent: 让我看看...
> 你: 能修复吗？
Agent: 可以，我需要修改 3 个文件...
```

**需要**: 交互式 Shell 界面

### 场景 2：CI/CD 自动化

你的 CI 流程需要自动生成文档：

```bash
# .github/workflows/docs.yml
- name: Generate docs
  run: kimi --command "生成 API 文档" --mode print
```

**需要**: 非交互式、脚本友好的输出

### 场景 3：IDE 集成

你在 VSCode 中编码，希望 Agent 实时显示进度：

```
Cursor IDE
┌─────────────────────────────────┐
│ Agent: 正在重构代码...          │
│ ▓▓▓▓▓▓▓▓░░░░ 60%               │
│                                 │
│ 已完成:                         │
│ ✓ 重命名变量                    │
│ ✓ 提取函数                      │
│ ⏳ 更新测试...                  │
└─────────────────────────────────┘
```

**需要**: 结构化、可解析的协议

### 场景 4：自定义集成

你构建了一个 Web 界面，需要完全控制通信：

```javascript
// 自定义 Web UI
const agent = new AgentClient({ mode: "wire" });
await agent.send({ type: "query", content: "..." });
const response = await agent.receive();
```

**需要**: 低级别的 JSON-RPC 协议

## 14.2 四种 UI 模式详解

### Mode 1: Shell（交互式）

**特点**:
- 富文本输出（颜色、格式）
- 实时流式显示
- 支持用户输入
- 适合人类使用

**完整实现**:

```python
# ui/shell.py

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress
from rich.live import Live

class ShellUI:
    """交互式 Shell UI"""

    def __init__(self):
        self.console = Console()
        self.message_count = 0

    async def display_message(self, role: str, content: str):
        """显示消息"""
        self.message_count += 1

        if role == "user":
            self.console.print(
                Panel(
                    content,
                    title="[bold blue]你[/bold blue]",
                    border_style="blue"
                )
            )
        elif role == "assistant":
            # 渲染 Markdown
            md = Markdown(content)
            self.console.print(
                Panel(
                    md,
                    title="[bold yellow]Agent[/bold yellow]",
                    border_style="yellow"
                )
            )

    async def get_user_input(self) -> str:
        """获取用户输入"""
        return Prompt.ask("\n[bold blue]你[/bold blue]")

    async def show_tool_call(self, tool_name: str, params: dict):
        """显示工具调用"""
        self.console.print(
            f"[dim]🔧 {tool_name}({', '.join(f'{k}={v}' for k, v in params.items())})[/dim]"
        )

    async def show_progress(self, task: str, total: int):
        """显示进度条"""
        with Progress() as progress:
            task_id = progress.add_task(f"[cyan]{task}", total=total)
            for i in range(total):
                await asyncio.sleep(0.1)
                progress.update(task_id, advance=1)

    async def stream_content(self, content_stream):
        """流式显示内容"""
        with Live("", console=self.console) as live:
            buffer = ""
            for chunk in content_stream:
                buffer += chunk
                live.update(Markdown(buffer))
```

**使用示例**:

```python
# 启动交互式会话
ui = ShellUI()
agent = Agent(ui=ui)

while True:
    user_input = await ui.get_user_input()
    if user_input.lower() in ["quit", "exit"]:
        break

    await agent.run(user_input)
```

### Mode 2: Print（脚本化）

**特点**:
- 纯文本输出
- 不支持交互
- 适合日志和脚本
- 可配置输出格式（text/json）

**完整实现**:

```python
# ui/print.py

import json
import sys
from datetime import datetime

class PrintUI:
    """非交互式 Print UI"""

    def __init__(self, output_format: str = "text", verbose: bool = False):
        self.output_format = output_format
        self.verbose = verbose
        self.start_time = datetime.now()

    async def display_message(self, role: str, content: str):
        """显示消息"""
        if self.output_format == "text":
            print(f"[{role}] {content}")
        elif self.output_format == "json":
            print(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content
            }))

    async def get_user_input(self) -> str:
        """Print 模式不支持交互"""
        raise NotImplementedError(
            "Print mode doesn't support user input. "
            "Use --command to provide input."
        )

    async def show_tool_call(self, tool_name: str, params: dict):
        """显示工具调用"""
        if not self.verbose:
            return

        if self.output_format == "text":
            print(f"[TOOL] {tool_name}: {params}")
        elif self.output_format == "json":
            print(json.dumps({
                "type": "tool_call",
                "tool": tool_name,
                "params": params
            }))

    async def show_error(self, error: str):
        """显示错误"""
        if self.output_format == "text":
            print(f"[ERROR] {error}", file=sys.stderr)
        elif self.output_format == "json":
            print(json.dumps({
                "type": "error",
                "message": error
            }), file=sys.stderr)

    async def show_summary(self):
        """显示执行摘要"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.output_format == "text":
            print(f"\n完成！耗时: {elapsed:.2f}s")
```

**使用示例**:

```bash
# 文本输出
$ kimi --command "分析代码" --mode print
[assistant] 正在分析...
[assistant] 发现 3 个问题...

# JSON 输出（便于解析）
$ kimi --command "分析代码" --mode print --format json
{"timestamp":"2025-01-15T10:30:00","role":"assistant","content":"正在分析..."}
{"timestamp":"2025-01-15T10:30:05","role":"assistant","content":"发现 3 个问题..."}

# 在脚本中使用
#!/bin/bash
OUTPUT=$(kimi --command "生成测试" --mode print --format json)
echo "$OUTPUT" | jq '.content'
```

### Mode 3: ACP（Agent Client Protocol）

**特点**:
- IDE 集成标准协议
- 支持进度报告
- 结构化输出
- 类似 LSP (Language Server Protocol)

**完整实现**:

```python
# ui/acp.py

import asyncio
import json
from typing import AsyncIterator

class ACPServer:
    """Agent Client Protocol Server"""

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.message_id = 0

    async def send_notification(self, method: str, params: dict):
        """发送通知（不需要响应）"""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        await self._write_message(message)

    async def send_request(self, method: str, params: dict) -> dict:
        """发送请求（需要响应）"""
        self.message_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self.message_id,
            "method": method,
            "params": params
        }
        await self._write_message(message)

        # 等待响应
        response = await self._read_message()
        return response.get("result")

    async def _write_message(self, message: dict):
        """写入消息"""
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self.writer.write(header.encode() + content.encode())
        await self.writer.drain()

    async def _read_message(self) -> dict:
        """读取消息"""
        # 读取 header
        header = await self.reader.readuntil(b"\r\n\r\n")
        content_length = int(header.split(b":")[1].strip())

        # 读取 body
        content = await self.reader.read(content_length)
        return json.loads(content)

class ACPUI:
    """ACP UI 实现"""

    def __init__(self, server: ACPServer):
        self.server = server

    async def display_message(self, role: str, content: str):
        """显示消息"""
        await self.server.send_notification("agent/message", {
            "role": role,
            "content": content
        })

    async def show_tool_call(self, tool_name: str, params: dict):
        """显示工具调用"""
        await self.server.send_notification("agent/toolCall", {
            "tool": tool_name,
            "params": params
        })

    async def show_progress(self, task: str, progress: float):
        """显示进度"""
        await self.server.send_notification("agent/progress", {
            "task": task,
            "progress": progress
        })

    async def request_approval(self, action: str) -> bool:
        """请求用户批准"""
        result = await self.server.send_request("agent/requestApproval", {
            "action": action
        })
        return result.get("approved", False)
```

**IDE 集成示例**:

```typescript
// VSCode 扩展
import { ACPClient } from 'agent-client-protocol';

class KimiExtension {
    private client: ACPClient;

    async activate() {
        // 启动 Agent 服务器
        this.client = new ACPClient({
            command: 'kimi',
            args: ['--mode', 'acp']
        });

        // 监听进度
        this.client.onNotification('agent/progress', (params) => {
            vscode.window.showProgress({
                title: params.task,
                percentage: params.progress
            });
        });

        // 监听消息
        this.client.onNotification('agent/message', (params) => {
            this.appendToChat(params.role, params.content);
        });
    }

    async sendQuery(query: string) {
        const response = await this.client.sendRequest('agent/query', {
            content: query
        });
        return response;
    }
}
```

### Mode 4: Wire（自定义协议）

**特点**:
- 最底层的协议
- 完全控制通信
- JSON-RPC 2.0
- 适合自定义集成

**完整实现**:

```python
# ui/wire.py

import json
import asyncio
from typing import AsyncIterator

class WireProtocol:
    """Wire Protocol 实现"""

    def __init__(self):
        self.handlers = {}

    def register_handler(self, method: str, handler):
        """注册消息处理器"""
        self.handlers[method] = handler

    async def handle_message(self, message: dict) -> dict:
        """处理消息"""
        method = message.get("method")
        params = message.get("params", {})

        if method not in self.handlers:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

        try:
            result = await self.handlers[method](params)
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }

class WireUI:
    """Wire UI 实现"""

    def __init__(self, protocol: WireProtocol):
        self.protocol = protocol
        self.event_queue = asyncio.Queue()

    async def display_message(self, role: str, content: str):
        """发送消息事件"""
        await self.event_queue.put({
            "type": "message",
            "role": role,
            "content": content
        })

    async def show_tool_call(self, tool_name: str, params: dict):
        """发送工具调用事件"""
        await self.event_queue.put({
            "type": "tool_call",
            "tool": tool_name,
            "params": params
        })

    async def get_events(self) -> AsyncIterator[dict]:
        """获取事件流"""
        while True:
            event = await self.event_queue.get()
            yield event
```

## 14.3 模式切换实现

### 统一的 UI 接口

```python
# ui/base.py

from typing import Protocol, AsyncIterator

class UI(Protocol):
    """UI 统一接口"""

    async def display_message(self, role: str, content: str):
        """显示消息"""
        ...

    async def get_user_input(self) -> str:
        """获取用户输入（可选）"""
        ...

    async def show_tool_call(self, tool_name: str, params: dict):
        """显示工具调用"""
        ...
```

### UI 工厂

```python
# ui/factory.py

from .shell import ShellUI
from .print import PrintUI
from .acp import ACPUI
from .wire import WireUI

class UIFactory:
    """UI 工厂"""

    @staticmethod
    def create(mode: str, **kwargs) -> UI:
        """创建 UI 实例"""
        if mode == "shell":
            return ShellUI()
        elif mode == "print":
            return PrintUI(
                output_format=kwargs.get("format", "text"),
                verbose=kwargs.get("verbose", False)
            )
        elif mode == "acp":
            # ACP 需要 reader/writer
            return ACPUI(kwargs["server"])
        elif mode == "wire":
            return WireUI(kwargs["protocol"])
        else:
            raise ValueError(f"Unknown UI mode: {mode}")
```

### CLI 集成

```python
# main.py

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["shell", "print", "acp", "wire"],
                       default="shell")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--command", help="非交互式命令")

    args = parser.parse_args()

    # 创建 UI
    ui = UIFactory.create(
        mode=args.mode,
        format=args.format
    )

    # 创建 Agent
    agent = Agent(ui=ui)

    # 运行
    if args.command:
        # 非交互式
        asyncio.run(agent.run(args.command))
    else:
        # 交互式
        asyncio.run(agent.run_interactive())
```

## 14.4 流式输出处理

### 流式显示实现

```python
class StreamingUI:
    """支持流式输出的 UI"""

    async def stream_response(self, stream: AsyncIterator[str]):
        """流式显示 LLM 响应"""
        buffer = ""

        async for chunk in stream:
            buffer += chunk

            # 实时更新显示
            await self._update_display(buffer)

        # 最终显示
        await self.display_message("assistant", buffer)

    async def _update_display(self, content: str):
        """更新显示（子类实现）"""
        pass

# Shell 模式的流式实现
class ShellUI(StreamingUI):
    async def _update_display(self, content: str):
        # 使用 Rich 的 Live
        self.live.update(Markdown(content))

# Print 模式的流式实现
class PrintUI(StreamingUI):
    async def _update_display(self, content: str):
        # 简单地打印最后一行
        print(f"\r{content[-100:]}", end="", flush=True)
```

## 14.5 常见陷阱

### 陷阱 1：混淆交互式和非交互式

```python
# ❌ 错误：在 Print 模式调用 get_user_input
if args.mode == "print":
    ui = PrintUI()
    user_input = await ui.get_user_input()  # 会抛出异常！

# ✅ 正确：检查是否支持交互
def supports_interaction(ui: UI) -> bool:
    return hasattr(ui, 'get_user_input') and callable(ui.get_user_input)

if supports_interaction(ui):
    user_input = await ui.get_user_input()
else:
    user_input = args.command
```

### 陷阱 2：在脚本中使用富文本

```python
# ❌ 错误：在 CI 中输出颜色代码
$ kimi --mode shell > output.log
# output.log 包含 ANSI 颜色代码，难以解析

# ✅ 正确：使用 Print 模式
$ kimi --mode print --format json > output.json
```

### 陷阱 3：忽略错误输出

```python
# ❌ 错误：所有输出到 stdout
print(f"Error: {error}")  # 混入正常输出

# ✅ 正确：错误输出到 stderr
import sys
print(f"Error: {error}", file=sys.stderr)
```

## 14.6 最佳实践

### 1. 自动检测模式

```python
def auto_detect_mode() -> str:
    """自动检测合适的 UI 模式"""
    import sys

    # 检查是否在 TTY
    if not sys.stdout.isatty():
        return "print"

    # 检查环境变量
    if os.getenv("KIMI_MODE"):
        return os.getenv("KIMI_MODE")

    # 默认交互式
    return "shell"
```

### 2. 优雅降级

```python
class RobustUI:
    """支持降级的 UI"""

    def __init__(self, preferred_mode: str):
        try:
            self.ui = UIFactory.create(preferred_mode)
        except ImportError:
            # 如果 rich 库不可用，降级到 Print
            logger.warning(f"Cannot create {preferred_mode} UI, falling back to print")
            self.ui = PrintUI()
```

### 3. 进度反馈

```python
# 长时间操作应该显示进度
async def long_operation(ui: UI):
    await ui.show_progress("处理文件", 0.0)

    for i, file in enumerate(files):
        process(file)
        progress = (i + 1) / len(files)
        await ui.show_progress("处理文件", progress)
```

## 14.7 FAQ

**Q: 如何在 SSH 会话中使用 Shell 模式？**

A: Shell 模式依赖终端特性。在 SSH 中确保：
```bash
# 检查 TERM 环境变量
echo $TERM  # 应该不是 "dumb"

# 如果有问题，设置正确的 TERM
export TERM=xterm-256color
```

**Q: Print 模式的 JSON 输出如何解析？**

A: 每行是一个独立的 JSON 对象：
```bash
kimi --mode print --format json | while read line; do
    echo "$line" | jq '.content'
done
```

**Q: 如何在 Web UI 中使用 Wire 协议？**

A: 通过 WebSocket 桥接：
```python
# server.py
import websockets

async def handle_client(websocket):
    protocol = WireProtocol()
    ui = WireUI(protocol)
    agent = Agent(ui=ui)

    async for message in websocket:
        data = json.loads(message)
        response = await protocol.handle_message(data)
        await websocket.send(json.dumps(response))
```

## 14.8 练习

### 练习 1: 实现彩色输出

扩展 PrintUI，支持彩色输出（但在非 TTY 时自动禁用）：

```python
class ColoredPrintUI(PrintUI):
    def __init__(self):
        super().__init__()
        # TODO: 检测是否支持颜色
        # TODO: 实现彩色输出
```

### 练习 2: 实现进度条

为 ShellUI 添加进度条支持：

```python
async def show_progress(self, task: str, current: int, total: int):
    # TODO: 使用 rich.progress 显示进度条
    pass
```

### 练习 3: WebSocket UI

实现一个基于 WebSocket 的 UI：

```python
class WebSocketUI:
    def __init__(self, websocket):
        self.websocket = websocket

    async def display_message(self, role: str, content: str):
        # TODO: 发送 WebSocket 消息
        pass
```

## 14.9 小结

本章学习了：

- ✅ **四种 UI 模式**：Shell、Print、ACP、Wire
- ✅ **使用场景**：交互式、脚本、IDE、自定义
- ✅ **实现细节**：协议、流式输出、错误处理
- ✅ **最佳实践**：自动检测、优雅降级、进度反馈

**关键要点**:

1. UI 模式与使用场景匹配
2. 统一接口，灵活实现
3. 支持流式输出提升体验
4. 错误处理要区分 stdout/stderr

多种 UI 模式让 Agent 适应不同场景：

- 🖥️ **Shell**: 开发调试
- 📜 **Print**: 自动化脚本
- 🔌 **ACP**: IDE 集成
- 🔧 **Wire**: 自定义集成

---

**上一章**：[第 13 章：上下文压缩](./13-context-compaction.md) ←
**下一章**：[第 15 章：配置系统](./15-config-system.md) →
