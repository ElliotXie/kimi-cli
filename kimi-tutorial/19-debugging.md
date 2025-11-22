# 第 19 章：调试技巧

你的 Agent 不按预期工作。它可能：

- 🔄 陷入无限循环
- 🎲 给出随机、不一致的结果
- 🐛 调用错误的工具
- 💥 神秘崩溃
- 🤔 "理解错误"用户意图

传统调试：
```python
def add(a, b):
    print(f"Debug: a={a}, b={b}")  # 加个 print
    return a + b
```

Agent 调试：
```python
async def agent_run(input):
    # LLM 内部发生了什么？🤔
    # 为什么选择这个工具？🤔
    # 上下文里有什么？🤔
    # 哪一步出错了？🤔
    ...
```

别担心！本章教你系统化的 Agent 调试技术。

## 19.1 调试的特殊挑战

### 挑战 1：不确定性

同样的代码，不同的行为：

```python
# 第一次运行
> Agent: 我将读取 README.md
[调用 read_file]
> Agent: 文件内容是...

# 第二次运行（完全相同的输入！）
> Agent: 让我先列出所有文件
[调用 list_files]
> Agent: 现在读取 README.md
[调用 read_file]
...
```

**原因**：LLM 是概率模型，有随机性。

### 挑战 2：黑盒推理

你看不到 LLM 的"思考过程"：

```
用户输入 → [??? 神秘的神经网络 ???] → 工具调用
```

不像传统代码，你可以单步执行、查看变量。

### 挑战 3：长链调用

Agent 可能执行很多步：

```
输入 → LLM1 → 工具1 → LLM2 → 工具2 → LLM3 → 工具3 → ... → 输出
```

哪一步出错了？很难定位。

### 挑战 4：上下文依赖

Agent 的行为依赖整个对话历史。问题可能源于很久之前的交互。

## 19.2 分层调试策略

```
┌─────────────────────────────┐
│ Level 4: 理解问题           │  为什么 LLM 做这个决定？
├─────────────────────────────┤
│ Level 3: 追踪流程           │  执行了哪些步骤？
├─────────────────────────────┤
│ Level 2: 检查状态           │  上下文、变量的值
├─────────────────────────────┤
│ Level 1: 日志输出           │  基本的 print/log
└─────────────────────────────┘
```

从简单到复杂，层层深入。

## 19.3 Level 1: 结构化日志

### 基础日志配置

```python
# debug/logger.py

import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO, log_file=None):
    """配置日志系统"""

    # 格式化
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 文件输出（可选）
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # 根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)

    return root_logger
```

### 在 Agent 中添加日志

```python
# agent.py

import logging

logger = logging.getLogger(__name__)

class Agent:
    async def run(self, user_input: str):
        logger.info(f"收到用户输入: {user_input!r}")

        # 构建消息
        messages = self.context.to_messages()
        logger.debug(f"发送给 LLM 的消息数: {len(messages)}")

        # 调用 LLM
        logger.info("调用 LLM...")
        response = await self.llm.generate(messages, tools=self.tools)

        logger.info(f"LLM 响应类型: {'tool_call' if response.tool_calls else 'text'}")

        if response.tool_calls:
            for tc in response.tool_calls:
                logger.info(f"工具调用: {tc.name}({tc.arguments})")

                # 执行工具
                try:
                    result = await self.execute_tool(tc)
                    logger.debug(f"工具结果: {result[:100]}...")  # 截断长输出
                except Exception as e:
                    logger.error(f"工具执行失败: {e}", exc_info=True)
                    raise

        logger.info(f"Agent 回复: {response.content[:100]}...")
        return response.content
```

### 使用日志

```bash
# 默认级别（INFO）
python main.py

# 调试级别（更详细）
python main.py --log-level DEBUG

# 保存到文件
python main.py --log-file agent.log
```

输出示例：

```
2025-01-15 10:30:00 - agent - INFO - 收到用户输入: '读取 README.md'
2025-01-15 10:30:00 - agent - DEBUG - 发送给 LLM 的消息数: 3
2025-01-15 10:30:00 - agent - INFO - 调用 LLM...
2025-01-15 10:30:01 - agent - INFO - LLM 响应类型: tool_call
2025-01-15 10:30:01 - agent - INFO - 工具调用: read_file({'path': 'README.md'})
2025-01-15 10:30:01 - agent - DEBUG - 工具结果: # My Project\n\nThis is a test...
2025-01-15 10:30:01 - agent - INFO - Agent 回复: 文件内容如下：\n\n# My Project...
```

## 19.4 Level 2: 状态检查器

查看 Agent 的内部状态：

```python
# debug/inspector.py

from typing import Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

class AgentInspector:
    """Agent 状态检查器"""

    def __init__(self, agent):
        self.agent = agent

    def show_context(self):
        """显示上下文"""
        console.print(Panel("[bold]上下文状态[/bold]"))

        table = Table(show_header=True)
        table.add_column("角色", style="cyan")
        table.add_column("内容", style="white")
        table.add_column("Token 数", style="yellow")

        for msg in self.agent.context.messages:
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            token_count = len(msg["content"]) // 4  # 粗略估算

            table.add_row(
                msg["role"],
                content,
                str(token_count)
            )

        console.print(table)
        console.print(f"\n总消息数: {len(self.agent.context.messages)}")
        console.print(f"估算总 tokens: {self.agent.context.total_tokens}")

    def show_tools(self):
        """显示可用工具"""
        console.print(Panel("[bold]可用工具[/bold]"))

        for tool in self.agent.tools:
            console.print(f"[cyan]• {tool.name}[/cyan]: {tool.description}")

    def show_last_interaction(self):
        """显示最后一次交互"""
        if not self.agent.context.messages:
            console.print("[red]没有交互历史[/red]")
            return

        console.print(Panel("[bold]最后一次交互[/bold]"))

        # 最后的用户消息
        user_msgs = [m for m in self.agent.context.messages if m["role"] == "user"]
        if user_msgs:
            last_user = user_msgs[-1]
            console.print(f"[bold blue]用户[/bold blue]: {last_user['content']}")

        # 最后的助手消息
        assistant_msgs = [m for m in self.agent.context.messages if m["role"] == "assistant"]
        if assistant_msgs:
            last_assistant = assistant_msgs[-1]
            console.print(f"[bold yellow]Agent[/bold yellow]: {last_assistant['content']}")

    def show_statistics(self):
        """显示统计信息"""
        console.print(Panel("[bold]运行统计[/bold]"))

        stats = {
            "总调用次数": self.agent.llm_call_count,
            "总 tokens": self.agent.total_tokens,
            "估算成本": f"${self.agent.estimated_cost:.4f}",
            "工具调用次数": self.agent.tool_call_count,
            "平均每次 tokens": self.agent.total_tokens // max(self.agent.llm_call_count, 1),
        }

        for key, value in stats.items():
            console.print(f"{key}: [cyan]{value}[/cyan]")

    def export_trace(self, filename: str):
        """导出完整追踪"""
        import json

        trace = {
            "messages": self.agent.context.messages,
            "tool_calls": self.agent.tool_call_history,
            "statistics": {
                "llm_calls": self.agent.llm_call_count,
                "total_tokens": self.agent.total_tokens,
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

        console.print(f"[green]追踪已保存到 {filename}[/green]")
```

### 使用检查器

```python
# 在 REPL 或脚本中
from debug.inspector import AgentInspector

inspector = AgentInspector(agent)

# 运行 Agent
await agent.run("读取所有 Python 文件")

# 检查状态
inspector.show_context()
inspector.show_statistics()
inspector.show_last_interaction()

# 导出追踪
inspector.export_trace("debug_trace.json")
```

## 19.5 Level 3: 流程追踪

### 追踪每一步执行

```python
# debug/tracer.py

import time
from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class TraceStep:
    """追踪的一步"""
    step_num: int
    timestamp: float
    type: str  # "llm_call", "tool_call", "user_input"
    data: dict

class ExecutionTracer:
    """执行追踪器"""

    def __init__(self):
        self.steps: List[TraceStep] = []
        self.step_num = 0

    def record_user_input(self, user_input: str):
        """记录用户输入"""
        self.steps.append(TraceStep(
            step_num=self.step_num,
            timestamp=time.time(),
            type="user_input",
            data={"input": user_input}
        ))
        self.step_num += 1

    def record_llm_call(self, messages, response):
        """记录 LLM 调用"""
        self.steps.append(TraceStep(
            step_num=self.step_num,
            timestamp=time.time(),
            type="llm_call",
            data={
                "message_count": len(messages),
                "response_type": "tool_call" if response.tool_calls else "text",
                "response_preview": response.content[:100] if response.content else "",
                "tool_calls": [
                    {"name": tc.name, "args": tc.arguments}
                    for tc in response.tool_calls
                ] if response.tool_calls else []
            }
        ))
        self.step_num += 1

    def record_tool_call(self, tool_name: str, arguments: dict, result: str, success: bool):
        """记录工具调用"""
        self.steps.append(TraceStep(
            step_num=self.step_num,
            timestamp=time.time(),
            type="tool_call",
            data={
                "tool": tool_name,
                "arguments": arguments,
                "success": success,
                "result_preview": result[:200] if result else ""
            }
        ))
        self.step_num += 1

    def print_trace(self):
        """打印追踪"""
        from rich.console import Console
        from rich.tree import Tree

        console = Console()
        tree = Tree("[bold]执行追踪[/bold]")

        for step in self.steps:
            time_str = datetime.fromtimestamp(step.timestamp).strftime("%H:%M:%S.%f")[:-3]

            if step.type == "user_input":
                node = tree.add(f"[cyan]{step.step_num}. 用户输入[/cyan] ({time_str})")
                node.add(f"内容: {step.data['input']!r}")

            elif step.type == "llm_call":
                node = tree.add(f"[yellow]{step.step_num}. LLM 调用[/yellow] ({time_str})")
                node.add(f"消息数: {step.data['message_count']}")
                node.add(f"响应类型: {step.data['response_type']}")

                if step.data['tool_calls']:
                    tc_node = node.add("[magenta]工具调用请求:[/magenta]")
                    for tc in step.data['tool_calls']:
                        tc_node.add(f"• {tc['name']}({tc['args']})")

            elif step.type == "tool_call":
                status = "✅" if step.data['success'] else "❌"
                node = tree.add(f"[green]{step.step_num}. 工具执行[/green] ({time_str}) {status}")
                node.add(f"工具: {step.data['tool']}")
                node.add(f"参数: {step.data['arguments']}")
                node.add(f"结果: {step.data['result_preview']}")

        console.print(tree)
```

### 集成到 Agent

```python
class Agent:
    def __init__(self, ..., enable_tracing=False):
        # ...
        self.tracer = ExecutionTracer() if enable_tracing else None

    async def run(self, user_input: str):
        if self.tracer:
            self.tracer.record_user_input(user_input)

        # ...
        response = await self.llm.generate(messages, tools=self.tools)

        if self.tracer:
            self.tracer.record_llm_call(messages, response)

        # 执行工具
        if response.tool_calls:
            for tc in response.tool_calls:
                try:
                    result = await self.execute_tool(tc)
                    if self.tracer:
                        self.tracer.record_tool_call(tc.name, tc.arguments, result, True)
                except Exception as e:
                    if self.tracer:
                        self.tracer.record_tool_call(tc.name, tc.arguments, str(e), False)
                    raise

        return response.content
```

### 查看追踪

```python
agent = Agent(enable_tracing=True)
await agent.run("分析项目结构")
agent.tracer.print_trace()
```

输出：

```
执行追踪
├── 0. 用户输入 (10:30:00.123)
│   └── 内容: '分析项目结构'
├── 1. LLM 调用 (10:30:00.234)
│   ├── 消息数: 2
│   ├── 响应类型: tool_call
│   └── 工具调用请求:
│       └── • list_files({'pattern': '**/*'})
├── 2. 工具执行 (10:30:00.345) ✅
│   ├── 工具: list_files
│   ├── 参数: {'pattern': '**/*'}
│   └── 结果: ['README.md', 'src/main.py', 'tests/test_main.py']
├── 3. LLM 调用 (10:30:01.456)
│   ├── 消息数: 4
│   └── 响应类型: text
...
```

## 19.6 Level 4: 交互式调试

### 断点调试

```python
# debug/breakpoints.py

from rich.console import Console
from rich.prompt import Prompt

console = Console()

class DebugBreakpoint:
    """调试断点"""

    def __init__(self, agent):
        self.agent = agent
        self.enabled = True

    async def hit(self, context: str, data: dict = None):
        """触发断点"""
        if not self.enabled:
            return

        console.print(f"\n[bold red]🔴 断点触发: {context}[/bold red]")

        if data:
            console.print("[yellow]当前数据:[/yellow]")
            for key, value in data.items():
                console.print(f"  {key}: {value}")

        # 交互菜单
        while True:
            console.print("\n[bold]调试命令:[/bold]")
            console.print("  [c]ontinue - 继续执行")
            console.print("  [s]tatus   - 查看状态")
            console.print("  [m]essages - 查看消息历史")
            console.print("  [t]ools    - 查看可用工具")
            console.print("  [q]uit     - 退出程序")

            cmd = Prompt.ask("选择", choices=["c", "s", "m", "t", "q"], default="c")

            if cmd == "c":
                break
            elif cmd == "s":
                from debug.inspector import AgentInspector
                inspector = AgentInspector(self.agent)
                inspector.show_statistics()
            elif cmd == "m":
                for i, msg in enumerate(self.agent.context.messages):
                    console.print(f"{i}. [{msg['role']}]: {msg['content'][:100]}")
            elif cmd == "t":
                for tool in self.agent.tools:
                    console.print(f"• {tool.name}: {tool.description}")
            elif cmd == "q":
                import sys
                sys.exit(0)
```

### 在 Agent 中使用

```python
class Agent:
    def __init__(self, ..., debug_mode=False):
        # ...
        self.breakpoint = DebugBreakpoint(self) if debug_mode else None

    async def run(self, user_input: str):
        # 在关键点添加断点
        if self.breakpoint:
            await self.breakpoint.hit("用户输入", {"input": user_input})

        # ...
        response = await self.llm.generate(...)

        if self.breakpoint and response.tool_calls:
            await self.breakpoint.hit(
                "LLM 决定调用工具",
                {"tools": [tc.name for tc in response.tool_calls]}
            )

        # ...
```

## 19.7 常见问题诊断

### 问题 1：Agent 陷入循环

**症状**：重复调用相同工具

**诊断**：

```python
# 检查工具调用历史
inspector = AgentInspector(agent)
tool_calls = agent.tool_call_history

# 查找重复模式
from collections import Counter
counter = Counter(tc['name'] for tc in tool_calls)
console.print("工具调用频率:", counter)

# 如果某个工具被调用太多次...
if counter.most_common(1)[0][1] > 5:
    console.print("[red]检测到可能的循环！[/red]")
```

**解决**：

```python
class LoopDetector:
    """循环检测器"""

    def __init__(self, max_repeats=3):
        self.history = []
        self.max_repeats = max_repeats

    def check(self, tool_name: str, arguments: dict) -> bool:
        """检查是否循环

        Returns:
            True if loop detected
        """
        signature = (tool_name, frozenset(arguments.items()))
        self.history.append(signature)

        # 检查最近的调用
        recent = self.history[-self.max_repeats:]
        if len(recent) == self.max_repeats and len(set(recent)) == 1:
            return True  # 循环！

        return False

# 在 Agent 中使用
class Agent:
    def __init__(self, ...):
        self.loop_detector = LoopDetector()

    async def execute_tool(self, tool_call):
        # 检测循环
        if self.loop_detector.check(tool_call.name, tool_call.arguments):
            raise LoopDetectedError(
                f"检测到循环: 重复调用 {tool_call.name}"
            )

        # 正常执行
        return await super().execute_tool(tool_call)
```

### 问题 2：工具返回错误被忽略

**症状**：工具失败了，但 Agent 继续执行

**诊断**：

```python
# 检查工具结果
for step in agent.tracer.steps:
    if step.type == "tool_call" and not step.data['success']:
        console.print(f"[red]工具失败: {step.data['tool']}[/red]")
        console.print(f"错误: {step.data['result_preview']}")
```

**解决**：确保错误信息被添加到上下文

```python
async def execute_tool(self, tool_call):
    try:
        result = await tool.execute(tool_call.arguments)
        return f"成功: {result}"
    except Exception as e:
        # 重要：返回明确的错误信息给 LLM
        error_msg = f"错误: 工具 {tool_call.name} 执行失败\n原因: {str(e)}\n请尝试其他方法。"
        logger.error(error_msg)
        return error_msg  # LLM 会看到这个错误
```

### 问题 3：上下文爆炸

**症状**：Token 数暴增，成本高

**诊断**：

```python
inspector = AgentInspector(agent)
inspector.show_context()

# 查看哪些消息占用最多 tokens
for msg in agent.context.messages:
    tokens = len(msg['content']) // 4
    if tokens > 1000:
        console.print(f"[yellow]大消息 ({tokens} tokens):[/yellow]")
        console.print(f"  角色: {msg['role']}")
        console.print(f"  内容: {msg['content'][:200]}...")
```

**解决**：启用上下文压缩

```python
agent = Agent(
    ...,
    enable_compression=True,  # 自动压缩旧消息
    max_context_tokens=8000   # 限制上下文大小
)
```

### 问题 4：LLM 误解指令

**症状**：Agent 做了奇怪的事情

**诊断**：

```python
# 查看发送给 LLM 的完整消息
messages = agent.context.to_messages()
for msg in messages:
    console.print(f"[{msg['role']}]")
    console.print(msg['content'])
    console.print("---")
```

**解决**：改进提示词

```python
# 添加更明确的指令
system_prompt = """
你是一个文件操作助手。

重要规则：
1. 在修改文件前，MUST 先用 read_file 读取
2. 如果工具返回错误，不要重试相同操作
3. 完成任务后，明确告诉用户"任务完成"
"""
```

## 19.8 调试工具箱

### 工具 1：LLM 响应查看器

```python
# debug/llm_viewer.py

def view_llm_response(response):
    """美化显示 LLM 响应"""
    from rich.panel import Panel

    console.print(Panel("[bold]LLM 响应[/bold]"))

    if response.content:
        console.print(f"[cyan]内容[/cyan]:")
        console.print(response.content)

    if response.tool_calls:
        console.print(f"\n[yellow]工具调用 ({len(response.tool_calls)})[/yellow]:")
        for i, tc in enumerate(response.tool_calls, 1):
            console.print(f"{i}. {tc.name}")
            for key, value in tc.arguments.items():
                console.print(f"   {key}: {value!r}")
```

### 工具 2：差异对比器

```python
# debug/differ.py

def compare_runs(trace1_file: str, trace2_file: str):
    """对比两次运行的差异"""
    import json
    from difflib import unified_diff

    with open(trace1_file) as f:
        trace1 = json.load(f)
    with open(trace2_file) as f:
        trace2 = json.load(f)

    # 对比工具调用序列
    tools1 = [tc['tool'] for tc in trace1['tool_calls']]
    tools2 = [tc['tool'] for tc in trace2['tool_calls']]

    if tools1 != tools2:
        console.print("[red]工具调用序列不同！[/red]")
        console.print("运行1:", tools1)
        console.print("运行2:", tools2)
    else:
        console.print("[green]工具调用序列相同[/green]")
```

### 工具 3：性能分析器

```python
# debug/profiler.py

import time
from contextlib import contextmanager

class Profiler:
    """性能分析器"""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def measure(self, name: str):
        """测量代码块执行时间"""
        start = time.time()
        yield
        elapsed = time.time() - start

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

    def report(self):
        """生成报告"""
        console.print(Panel("[bold]性能报告[/bold]"))

        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            total = sum(times)
            console.print(f"{name}:")
            console.print(f"  调用次数: {len(times)}")
            console.print(f"  平均耗时: {avg:.3f}s")
            console.print(f"  总耗时: {total:.3f}s")

# 使用
profiler = Profiler()

with profiler.measure("llm_call"):
    await llm.generate(...)

with profiler.measure("tool_execution"):
    await tool.execute(...)

profiler.report()
```

## 19.9 最佳实践

### 1. 始终启用基础日志

```python
# ✅ 好：生产环境也保留 INFO 日志
setup_logging(level=logging.INFO)

# ❌ 坏：完全关闭日志
logging.disable(logging.CRITICAL)
```

### 2. 保存调试追踪

```python
# ✅ 好：出问题时可以回溯
agent = Agent(enable_tracing=True)
# ... 运行后
agent.tracer.export("trace.json")

# ❌ 坏：追踪丢失
agent = Agent(enable_tracing=False)
```

### 3. 使用结构化日志

```python
# ✅ 好：可以解析
logger.info("tool_executed", extra={
    "tool": "read_file",
    "args": {"path": "test.txt"},
    "success": True
})

# ❌ 坏：难以解析
logger.info("Tool read_file executed on test.txt successfully")
```

### 4. 分环境配置

```python
# ✅ 好
if os.getenv("ENV") == "production":
    setup_logging(level=logging.WARNING)
else:
    setup_logging(level=logging.DEBUG)
```

## 19.10 FAQ

**Q: 生产环境应该启用调试功能吗？**

A: 部分功能：
- ✅ 基础日志（INFO 级别）
- ✅ 错误追踪
- ❌ 详细追踪（太慢）
- ❌ 断点（会卡住）

**Q: 如何调试随机性问题？**

A: 设置固定的 random seed（如果 LLM API 支持）：

```python
response = await llm.generate(
    messages,
    temperature=0,  # 降低随机性
    seed=42         # 固定种子（某些 API 支持）
)
```

**Q: 调试时如何避免花费太多 API 成本？**

A: 使用 Mock LLM（参见第 18 章），或缓存 LLM 响应：

```python
@cache_llm_responses
async def generate(...):
    # 相同输入返回缓存结果
    ...
```

## 19.11 练习

### 练习 1：实现时间旅行调试

允许"回放"之前的执行：

```python
class TimeTravel:
    def save_checkpoint(self, name: str):
        """保存当前状态"""
        pass

    def restore_checkpoint(self, name: str):
        """恢复到某个状态"""
        pass
```

### 练习 2：可视化工具调用图

绘制 Agent 的执行流程图：

```
用户输入
  ↓
LLM 思考
  ↓
read_file("README.md") → 成功
  ↓
LLM 思考
  ↓
write_file("summary.txt", ...) → 成功
  ↓
最终回复
```

### 练习 3：自动化问题检测

实现一个自动检测常见问题的工具：
- 循环检测
- 上下文过大警告
- 工具失败模式识别
- 性能回归检测

## 19.12 小结

Agent 调试技术：

- 📝 **结构化日志**：基础但必不可少
- 🔍 **状态检查器**：查看 Agent 内部状态
- 🎬 **执行追踪**：记录每一步
- 🐛 **交互式调试**：断点和命令行调试
- 🔧 **问题诊断**：识别常见问题模式

记住：
- 🏗️ **分层调试**：从简单日志开始，逐步深入
- 💾 **保存追踪**：问题总是在不便调试时出现
- 🎯 **针对性**：针对 Agent 的特殊性（不确定性、LLM 黑盒）
- ⚖️ **平衡**：调试开销 vs. 可观测性

下一章，我们将学习如何部署和分发 Agent。

---

**上一章**：[第 18 章：测试策略](./18-testing.md) ←
**下一章**：[第 20 章：部署和分发](./20-deployment.md) →
