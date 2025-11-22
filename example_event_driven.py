"""
事件驱动观察者模式示例
演示如何在 Agent 系统中使用事件来解耦组件
"""
import asyncio
from typing import Callable, Dict, List, Any
from enum import Enum


class EventType(str, Enum):
    """事件类型枚举"""
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_DONE = "tool_call_done"
    TEXT_DELTA = "text_delta"
    ERROR = "error"


class EventEmitter:
    """事件发射器基类 - 实现观察者模式"""

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event_type: str, listener: Callable):
        """注册事件监听器"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(listener)

    def off(self, event_type: str, listener: Callable):
        """移除事件监听器"""
        if event_type in self._listeners:
            self._listeners[event_type].remove(listener)

    async def _emit(self, event_type: str, data: Any = None):
        """发射事件给所有订阅者"""
        if event_type in self._listeners:
            for listener in self._listeners[event_type]:
                # 支持异步监听器
                if asyncio.iscoroutinefunction(listener):
                    await listener(data)
                else:
                    listener(data)


class Tool:
    """模拟工具"""

    async def execute(self, command: str) -> str:
        """模拟工具执行"""
        await asyncio.sleep(1)  # 模拟耗时操作
        return f"Command '{command}' executed successfully"


class SimpleSoul(EventEmitter):
    """简化的 Agent 执行引擎 - 发布者角色"""

    def __init__(self):
        super().__init__()
        self.tool = Tool()

    async def execute_tool(self, tool_name: str, params: dict):
        """执行工具并发射事件"""

        # 1. 发射"工具开始"事件
        await self._emit(EventType.TOOL_CALL_STARTED, {
            "tool": tool_name,
            "params": params
        })

        try:
            # 2. 执行工具
            result = await self.tool.execute(params.get("command", ""))

            # 3. 发射"工具完成"事件
            await self._emit(EventType.TOOL_CALL_DONE, {
                "tool": tool_name,
                "result": result
            })

        except Exception as e:
            # 4. 发射错误事件
            await self._emit(EventType.ERROR, {
                "tool": tool_name,
                "error": str(e)
            })

    async def stream_text(self, text: str):
        """模拟流式文本输出"""
        for char in text:
            await self._emit(EventType.TEXT_DELTA, char)
            await asyncio.sleep(0.05)  # 模拟流式延迟


class ConsoleUI:
    """控制台 UI - 订阅者角色"""

    def __init__(self, soul: SimpleSoul):
        self.soul = soul

        # 订阅所有感兴趣的事件
        soul.on(EventType.TOOL_CALL_STARTED, self._on_tool_started)
        soul.on(EventType.TOOL_CALL_DONE, self._on_tool_done)
        soul.on(EventType.TEXT_DELTA, self._on_text_delta)
        soul.on(EventType.ERROR, self._on_error)

    async def _on_tool_started(self, data: dict):
        """处理工具开始事件"""
        print(f"\n🔧 Starting tool: {data['tool']}")
        print(f"   Parameters: {data['params']}")

    async def _on_tool_done(self, data: dict):
        """处理工具完成事件"""
        print(f"\n✓ Tool {data['tool']} completed")
        print(f"   Result: {data['result']}")

    async def _on_text_delta(self, delta: str):
        """处理流式文本事件"""
        print(delta, end="", flush=True)

    async def _on_error(self, data: dict):
        """处理错误事件"""
        print(f"\n❌ Error in {data['tool']}: {data['error']}")


class Logger:
    """日志记录器 - 另一个订阅者"""

    def __init__(self, soul: SimpleSoul):
        self.soul = soul
        self.log_file = []

        # 订阅事件
        soul.on(EventType.TOOL_CALL_DONE, self._log_tool_call)
        soul.on(EventType.ERROR, self._log_error)

    async def _log_tool_call(self, data: dict):
        """记录工具调用"""
        self.log_file.append(f"[TOOL] {data['tool']}: {data['result']}")

    async def _log_error(self, data: dict):
        """记录错误"""
        self.log_file.append(f"[ERROR] {data['tool']}: {data['error']}")

    def print_logs(self):
        """打印日志"""
        print("\n\n=== Logs ===")
        for log in self.log_file:
            print(log)


async def main():
    """演示事件驱动系统"""

    # 创建 Agent
    soul = SimpleSoul()

    # 创建多个订阅者
    ui = ConsoleUI(soul)
    logger = Logger(soul)

    print("=== Event-Driven Observer Pattern Demo ===\n")

    # 场景 1: 流式文本输出
    print("Scenario 1: Streaming text output")
    await soul.stream_text("Hello, this is a streaming message!")

    # 场景 2: 执行工具
    print("\n\nScenario 2: Tool execution")
    await soul.execute_tool("Shell", {"command": "ls -la"})

    await soul.execute_tool("ReadFile", {"command": "cat README.md"})

    # 打印日志
    logger.print_logs()

    print("\n\n=== Demo completed ===")


if __name__ == "__main__":
    asyncio.run(main())
