# 第 18 章：测试策略

Agent 是复杂的系统：它调用 LLM、执行工具、管理状态。如何确保它正常工作？

传统软件测试：
```python
def add(a, b):
    return a + b

# 简单！
assert add(2, 3) == 5
```

Agent 测试：
```python
async def agent_run(user_input):
    # 调用 LLM（不确定性！）
    # 可能调用多个工具（顺序不固定！）
    # 返回自然语言（难以精确验证！）
    ...

# 怎么测试？🤔
```

别慌！本章将教你一套完整的 Agent 测试策略。

## 18.1 测试的挑战

### 挑战 1：LLM 的不确定性

同样的输入，LLM 可能返回不同的输出：

```python
# 第一次运行
response = await llm.generate("介绍一下 Python")
# "Python 是一门编程语言..."

# 第二次运行
response = await llm.generate("介绍一下 Python")
# "Python 是一种高级语言..."  # 内容不同！
```

传统的断言不适用：
```python
assert response == "Python 是一门编程语言..."  # ❌ 太脆弱！
```

### 挑战 2：外部依赖

Agent 依赖外部服务：
- 💰 LLM API（需要 API key，花钱）
- 🐌 网络调用（慢）
- 🔄 状态变化（文件系统）

在测试中直接调用这些服务会导致：
- 测试慢
- 测试不稳定（网络问题）
- 花钱
- 副作用（创建真实文件）

### 挑战 3：复杂的交互流程

Agent 的执行流程是动态的：

```
用户输入 → LLM 思考 → 调用工具1 → LLM 再思考 → 调用工具2 → 最终回复
                              ↓                    ↓
                           可能失败             可能失败
```

如何测试这样的流程？

## 18.2 测试金字塔

对于 Agent，我们采用分层测试策略：

```
         /\
        /  \       E2E 测试 (5%)
       /────\      - 完整对话流程
      /      \     - 使用真实 LLM（少量）
     /────────\
    /          \   集成测试 (25%)
   /────────────\  - 工具 + Mock LLM
  /              \ - 上下文管理
 /────────────────\
/                  \ 单元测试 (70%)
────────────────────
- 单个工具
- 辅助函数
- 数据结构
```

### 为什么这样分配？

- **单元测试**：快、稳定、便宜，应该最多
- **集成测试**：测试组件协作，适度使用
- **E2E 测试**：慢、贵、不稳定，少量即可

## 18.3 单元测试：测试工具

工具是独立的、纯粹的功能，最容易测试。

### 测试文件读取工具

```python
# tests/test_tools.py

import pytest
from pathlib import Path
from tools.read_file import ReadFileTool
from kaos.memory import MemoryKaos

@pytest.fixture
def kaos():
    """测试夹具：提供内存文件系统"""
    k = MemoryKaos()
    # 准备测试数据
    k.writetext("hello.txt", "Hello, World!")
    k.writetext("data.json", '{"name": "test"}')
    return k

@pytest.mark.asyncio
async def test_read_existing_file(kaos):
    """测试读取存在的文件"""
    tool = ReadFileTool(kaos)

    result = await tool.execute({"path": "hello.txt"})

    assert "Hello, World!" in result
    assert "文件内容" in result  # 验证格式

@pytest.mark.asyncio
async def test_read_nonexistent_file(kaos):
    """测试读取不存在的文件"""
    tool = ReadFileTool(kaos)

    result = await tool.execute({"path": "not-exist.txt"})

    assert "错误" in result
    assert "不存在" in result

@pytest.mark.asyncio
async def test_read_json_file(kaos):
    """测试读取 JSON 文件"""
    tool = ReadFileTool(kaos)

    result = await tool.execute({"path": "data.json"})

    assert "test" in result
```

### 测试文件写入工具

```python
from tools.write_file import WriteFileTool

@pytest.mark.asyncio
async def test_write_new_file(kaos):
    """测试写入新文件"""
    tool = WriteFileTool(kaos)

    result = await tool.execute({
        "path": "new.txt",
        "content": "New content"
    })

    assert "成功" in result
    # 验证文件确实被创建
    assert kaos.exists("new.txt")
    assert kaos.readtext("new.txt") == "New content"

@pytest.mark.asyncio
async def test_write_creates_parent_dirs(kaos):
    """测试自动创建父目录"""
    tool = WriteFileTool(kaos)

    await tool.execute({
        "path": "dir1/dir2/file.txt",
        "content": "test"
    })

    # 验证目录被创建
    assert kaos.is_dir("dir1")
    assert kaos.is_dir("dir1/dir2")
    assert kaos.is_file("dir1/dir2/file.txt")

@pytest.mark.asyncio
async def test_write_readonly_kaos():
    """测试在只读文件系统写入"""
    from kaos.local import LocalKaos
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        kaos = LocalKaos(Path(tmpdir), readonly=True)
        tool = WriteFileTool(kaos)

        result = await tool.execute({
            "path": "test.txt",
            "content": "test"
        })

        assert "错误" in result
        assert "权限" in result
```

### 测试 Bash 工具

```python
from tools.bash import BashTool

@pytest.mark.asyncio
async def test_bash_simple_command(kaos):
    """测试简单的 bash 命令"""
    tool = BashTool(kaos)

    result = await tool.execute({"command": "echo 'hello'"})

    assert "hello" in result

@pytest.mark.asyncio
async def test_bash_command_with_output(kaos):
    """测试带输出的命令"""
    # 先创建文件
    kaos.writetext("test.txt", "line1\nline2\nline3")

    tool = BashTool(kaos)
    result = await tool.execute({"command": "wc -l test.txt"})

    assert "3" in result  # 3 行

@pytest.mark.asyncio
async def test_bash_command_error(kaos):
    """测试命令执行失败"""
    tool = BashTool(kaos)

    result = await tool.execute({"command": "ls /nonexistent"})

    # 应该包含错误信息
    assert "错误" in result.lower() or "cannot" in result.lower()
```

## 18.4 Mock LLM：隔离外部依赖

测试时不想真的调用 LLM API。我们需要 Mock。

### 简单 Mock

```python
# tests/mocks.py

from dataclasses import dataclass
from typing import List

@dataclass
class MockMessage:
    role: str
    content: str

@dataclass
class MockToolCall:
    name: str
    arguments: dict

@dataclass
class MockResponse:
    content: str
    tool_calls: List[MockToolCall]

class MockLLM:
    """Mock LLM for testing"""

    def __init__(self):
        self.call_count = 0
        self.messages_history = []

        # 预设响应
        self.responses = []
        self.response_index = 0

    def add_response(self, content: str = "", tool_calls: list = None):
        """添加预设响应"""
        self.responses.append(MockResponse(
            content=content,
            tool_calls=tool_calls or []
        ))

    async def generate(self, messages, tools=None):
        """模拟生成响应"""
        self.call_count += 1
        self.messages_history.append(messages)

        # 返回预设的响应
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response

        # 默认响应
        return MockResponse(content="Mock response", tool_calls=[])
```

### 使用 Mock LLM 测试 Agent

```python
# tests/test_agent.py

import pytest
from agent import Agent
from tests.mocks import MockLLM, MockToolCall
from kaos.memory import MemoryKaos

@pytest.mark.asyncio
async def test_agent_simple_query():
    """测试简单查询（不调用工具）"""
    llm = MockLLM()
    llm.add_response(content="这是一个测试回复")

    kaos = MemoryKaos()
    agent = Agent(llm=llm, kaos=kaos)

    response = await agent.run("你好")

    assert "测试回复" in response
    assert llm.call_count == 1  # 验证只调用了一次

@pytest.mark.asyncio
async def test_agent_calls_tool():
    """测试 Agent 调用工具"""
    llm = MockLLM()

    # 第一次调用：Agent 决定调用工具
    llm.add_response(
        content="",
        tool_calls=[MockToolCall(
            name="read_file",
            arguments={"path": "test.txt"}
        )]
    )

    # 第二次调用：基于工具结果回复
    llm.add_response(content="文件内容是...")

    kaos = MemoryKaos()
    kaos.writetext("test.txt", "test content")

    agent = Agent(llm=llm, kaos=kaos)
    response = await agent.run("读取 test.txt")

    assert llm.call_count == 2  # 调用了两次
    assert "文件内容是" in response

@pytest.mark.asyncio
async def test_agent_multiple_tools():
    """测试 Agent 调用多个工具"""
    llm = MockLLM()

    # 调用工具1
    llm.add_response(tool_calls=[
        MockToolCall(name="read_file", arguments={"path": "a.txt"})
    ])

    # 调用工具2
    llm.add_response(tool_calls=[
        MockToolCall(name="write_file", arguments={
            "path": "b.txt",
            "content": "new content"
        })
    ])

    # 最终回复
    llm.add_response(content="任务完成")

    kaos = MemoryKaos()
    kaos.writetext("a.txt", "old content")

    agent = Agent(llm=llm, kaos=kaos)
    response = await agent.run("复制 a.txt 到 b.txt")

    assert kaos.exists("b.txt")
    assert "任务完成" in response
```

### 智能 Mock：基于规则

有时你想让 Mock 更智能：

```python
class SmartMockLLM:
    """智能 Mock：根据输入决定输出"""

    async def generate(self, messages, tools=None):
        last_message = messages[-1]["content"]

        # 规则：如果提到"读取"，返回读取工具调用
        if "读取" in last_message or "read" in last_message.lower():
            # 提取文件名（简单版）
            import re
            match = re.search(r'[\w.]+\.txt', last_message)
            if match:
                filename = match.group(0)
                return MockResponse(
                    content="",
                    tool_calls=[MockToolCall(
                        name="read_file",
                        arguments={"path": filename}
                    )]
                )

        # 规则：如果提到"写入"，返回写入工具调用
        elif "写入" in last_message or "write" in last_message.lower():
            return MockResponse(
                content="",
                tool_calls=[MockToolCall(
                    name="write_file",
                    arguments={
                        "path": "output.txt",
                        "content": "mocked content"
                    }
                )]
            )

        # 默认：纯文本回复
        return MockResponse(content="我明白了", tool_calls=[])
```

## 18.5 集成测试：测试组件协作

集成测试验证多个组件一起工作：

```python
# tests/test_integration.py

@pytest.mark.asyncio
async def test_agent_with_context():
    """测试 Agent 的上下文管理"""
    llm = MockLLM()
    llm.add_response(content="收到第一条消息")
    llm.add_response(content="收到第二条消息，我记得之前的对话")

    kaos = MemoryKaos()
    agent = Agent(llm=llm, kaos=kaos)

    # 第一轮对话
    await agent.run("你好")

    # 第二轮对话
    await agent.run("还记得我吗？")

    # 验证上下文被保留
    assert len(agent.context.messages) == 4  # 2 user + 2 assistant

    # 验证 LLM 收到了完整历史
    last_call_messages = llm.messages_history[-1]
    assert len(last_call_messages) >= 3  # system + 第一轮 + 第二轮

@pytest.mark.asyncio
async def test_agent_with_max_steps():
    """测试 Agent 的步数限制"""
    llm = MockLLM()

    # 让 Agent 一直调用工具（无限循环）
    for _ in range(10):
        llm.add_response(tool_calls=[
            MockToolCall(name="read_file", arguments={"path": "test.txt"})
        ])

    kaos = MemoryKaos()
    kaos.writetext("test.txt", "content")

    agent = Agent(llm=llm, kaos=kaos, max_steps=5)

    # 应该在 5 步后停止
    with pytest.raises(MaxStepsExceeded):
        await agent.run("读取文件")

    assert llm.call_count == 5

@pytest.mark.asyncio
async def test_agent_error_handling():
    """测试 Agent 的错误处理"""
    llm = MockLLM()

    # Agent 尝试读取不存在的文件
    llm.add_response(tool_calls=[
        MockToolCall(name="read_file", arguments={"path": "nonexist.txt"})
    ])

    # Agent 收到错误后的恢复
    llm.add_response(content="抱歉，文件不存在")

    kaos = MemoryKaos()
    agent = Agent(llm=llm, kaos=kaos)

    response = await agent.run("读取 nonexist.txt")

    # Agent 应该优雅地处理错误
    assert "抱歉" in response or "不存在" in response
```

## 18.6 快照测试：验证提示词

提示词很重要，但难以测试。快照测试来救场！

### 什么是快照测试？

第一次运行时，保存输出为"快照"。以后的运行，对比新输出和快照。

```python
# tests/test_prompts.py

import pytest
import json
from pathlib import Path

def snapshot_path(test_name: str) -> Path:
    """获取快照文件路径"""
    return Path(__file__).parent / "snapshots" / f"{test_name}.json"

def assert_snapshot(data: dict, test_name: str):
    """断言数据匹配快照"""
    snap_file = snapshot_path(test_name)

    if snap_file.exists():
        # 对比模式
        expected = json.loads(snap_file.read_text())
        assert data == expected, f"快照不匹配！预期：{expected}\n实际：{data}"
    else:
        # 记录模式
        snap_file.parent.mkdir(parents=True, exist_ok=True)
        snap_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"✅ 快照已保存: {snap_file}")

def test_system_prompt_snapshot():
    """测试系统提示词"""
    from agent import build_system_prompt

    prompt = build_system_prompt(tools=["read_file", "write_file"])

    assert_snapshot({
        "prompt": prompt,
        "length": len(prompt),
        "tools_mentioned": ["read_file" in prompt, "write_file" in prompt]
    }, "system_prompt")

def test_tool_description_snapshot():
    """测试工具描述"""
    from tools.read_file import ReadFileTool

    tool = ReadFileTool(None)

    assert_snapshot({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters_schema
    }, "read_file_tool")
```

### 运行快照测试

```bash
# 第一次运行：生成快照
pytest tests/test_prompts.py
# ✅ 快照已保存: tests/snapshots/system_prompt.json

# 修改提示词后再运行：验证变化
pytest tests/test_prompts.py
# ❌ AssertionError: 快照不匹配！

# 如果变化是预期的，删除旧快照重新生成
rm tests/snapshots/system_prompt.json
pytest tests/test_prompts.py
```

## 18.7 E2E 测试：真实场景

少量 E2E 测试使用真实 LLM，验证整体流程：

```python
# tests/test_e2e.py

import os
import pytest

# 只在有 API key 时运行
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="需要 OPENAI_API_KEY"
)

@pytest.mark.asyncio
@pytest.mark.slow  # 标记为慢速测试
async def test_real_agent_read_file():
    """E2E: 真实 Agent 读取文件"""
    from openai import AsyncOpenAI
    from agent import Agent
    from kaos.memory import MemoryKaos

    # 真实 LLM
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 准备测试环境
    kaos = MemoryKaos()
    kaos.writetext("README.md", "# Test Project\n\nThis is a test.")

    agent = Agent(llm_client=client, kaos=kaos)

    # 执行任务
    response = await agent.run("读取 README.md 文件的内容")

    # 验证结果（宽松的断言）
    assert "Test Project" in response or "test" in response.lower()
    # LLM 应该理解并执行了任务

@pytest.mark.asyncio
@pytest.mark.slow
async def test_real_agent_multi_step():
    """E2E: 真实 Agent 多步任务"""
    # 更复杂的任务：读取文件 → 修改 → 写入新文件
    # ...实现略
```

运行 E2E 测试：

```bash
# 跳过慢速测试
pytest -m "not slow"

# 只运行 E2E 测试
pytest -m "slow"

# 运行所有测试
pytest
```

## 18.8 测试 Fixture 和辅助工具

复用测试代码：

```python
# tests/conftest.py

import pytest
from kaos.memory import MemoryKaos
from tests.mocks import MockLLM

@pytest.fixture
def kaos():
    """提供干净的内存文件系统"""
    return MemoryKaos()

@pytest.fixture
def mock_llm():
    """提供 Mock LLM"""
    return MockLLM()

@pytest.fixture
def sample_project(kaos):
    """提供示例项目结构"""
    kaos.writetext("README.md", "# Sample Project")
    kaos.writetext("src/main.py", "print('hello')")
    kaos.writetext("tests/test_main.py", "def test(): pass")
    return kaos

# 现在测试可以直接使用这些 fixture
def test_with_sample_project(sample_project):
    assert sample_project.exists("README.md")
    assert sample_project.is_dir("src")
```

## 18.9 性能和成本测试

测试 Agent 的性能特征：

```python
# tests/test_performance.py

import pytest
import time

@pytest.mark.asyncio
async def test_agent_response_time():
    """测试响应时间"""
    llm = MockLLM()
    llm.add_response(content="快速回复")

    kaos = MemoryKaos()
    agent = Agent(llm=llm, kaos=kaos)

    start = time.time()
    await agent.run("你好")
    elapsed = time.time() - start

    # Mock LLM 应该很快
    assert elapsed < 0.1  # 100ms 内

@pytest.mark.asyncio
async def test_agent_token_usage():
    """测试 token 使用量"""
    llm = MockLLM()

    # Mock 返回 token 统计
    llm.add_response(content="回复")

    kaos = MemoryKaos()
    agent = Agent(llm=llm, kaos=kaos)

    await agent.run("简短问题")

    # 验证 token 使用在预期范围内
    assert agent.total_tokens < 1000  # 简单对话不应该用太多 token

def test_prompt_size():
    """测试提示词大小"""
    from agent import build_system_prompt

    prompt = build_system_prompt(tools=all_tools)

    # 粗略估算 token 数（1 token ≈ 4 字符）
    estimated_tokens = len(prompt) / 4

    # 确保提示词不会太大
    assert estimated_tokens < 2000, "System prompt too large!"
```

## 18.10 CI/CD 集成

在 GitHub Actions 中运行测试：

```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-asyncio pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/ -m "not slow" --cov=agent --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

    # E2E 测试（需要 API key，只在 main 分支运行）
    - name: Run E2E tests
      if: github.ref == 'refs/heads/main'
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/ -m "slow"
```

## 18.11 最佳实践

### 1. 使用 MemoryKaos 测试文件操作

```python
# ✅ 好：快速、隔离
def test_file_ops():
    kaos = MemoryKaos()
    # 测试...

# ❌ 坏：慢、有副作用
def test_file_ops():
    # 创建真实文件
    Path("/tmp/test.txt").write_text("test")
    # 测试...
    # 需要清理
```

### 2. Mock 外部调用

```python
# ✅ 好：使用 Mock
def test_agent():
    agent = Agent(llm=MockLLM())

# ❌ 坏：调用真实 API
def test_agent():
    agent = Agent(llm=RealLLM())  # 慢、贵、不稳定
```

### 3. 一个测试一个断言（当可能时）

```python
# ✅ 好：关注点明确
def test_read_file_success(kaos):
    # 准备
    kaos.writetext("test.txt", "content")
    tool = ReadFileTool(kaos)

    # 执行
    result = await tool.execute({"path": "test.txt"})

    # 验证
    assert "content" in result

def test_read_file_not_found(kaos):
    tool = ReadFileTool(kaos)
    result = await tool.execute({"path": "missing.txt"})
    assert "错误" in result

# ❌ 坏：一个测试多个场景
def test_read_file(kaos):
    # 太多断言，失败时难以定位
    ...
```

### 4. 使用描述性的测试名称

```python
# ✅ 好
def test_agent_handles_file_not_found_error():
    ...

def test_agent_stops_after_max_steps_reached():
    ...

# ❌ 坏
def test_agent_1():
    ...

def test_agent_2():
    ...
```

## 18.12 常见测试模式

### 模式 1：Given-When-Then

```python
async def test_agent_creates_file():
    # Given: 准备测试环境
    kaos = MemoryKaos()
    llm = MockLLM()
    llm.add_response(tool_calls=[...])
    agent = Agent(llm=llm, kaos=kaos)

    # When: 执行操作
    await agent.run("创建 hello.txt")

    # Then: 验证结果
    assert kaos.exists("hello.txt")
```

### 模式 2：参数化测试

```python
@pytest.mark.parametrize("filename,expected", [
    ("test.txt", True),
    ("test.py", True),
    ("test.md", True),
    ("missing.txt", False),
])
async def test_file_exists(kaos, filename, expected):
    if expected:
        kaos.writetext(filename, "content")

    tool = ReadFileTool(kaos)
    result = await tool.execute({"path": filename})

    if expected:
        assert "content" in result
    else:
        assert "错误" in result
```

### 模式 3：测试异常

```python
async def test_agent_handles_tool_error():
    llm = MockLLM()
    kaos = BrokenKaos()  # 总是抛出异常的 KAOS

    agent = Agent(llm=llm, kaos=kaos)

    # 应该捕获异常并优雅处理
    response = await agent.run("读取文件")
    assert "错误" in response  # 不应该崩溃
```

## 18.13 FAQ

**Q: 要测试到什么程度？**

A: 目标是 80%+ 代码覆盖率。重点测试：
- 所有工具
- Agent 核心逻辑
- 错误处理路径

**Q: 真的需要 E2E 测试吗？**

A: 少量即可（5-10 个）。用于：
- 验证整体流程
- 捕获意外的集成问题
- 在发布前进行冒烟测试

**Q: 如何测试提示词的质量？**

A: 组合方式：
- 快照测试（结构不变）
- 少量 E2E 测试（真实效果）
- 人工 review（定期检查）

**Q: Mock 会不会太假？**

A: 分层测试解决这个问题：
- 单元测试：Mock 一切
- 集成测试：Mock LLM，真实工具
- E2E 测试：全部真实

## 18.14 练习

### 练习 1：为搜索工具写测试

```python
# TODO: 实现测试
async def test_search_files_with_pattern():
    """测试文件搜索"""
    kaos = MemoryKaos()
    # 创建一些测试文件
    # 测试 glob 模式匹配
    pass
```

### 练习 2：测试 Agent 的循环检测

```python
async def test_agent_detects_infinite_loop():
    """测试 Agent 检测无限循环"""
    # Agent 一直调用同一个工具
    # 应该被检测并停止
    pass
```

### 练习 3：实现快照测试工具

改进我们的 `assert_snapshot`，支持：
- 更新模式（`--update-snapshots`）
- 忽略某些字段
- 更友好的 diff 输出

## 18.15 小结

Agent 测试策略：

- ✅ **单元测试**：测试工具和函数（使用 MemoryKaos + Mock）
- ✅ **集成测试**：测试组件协作（Mock LLM，真实工具）
- ✅ **快照测试**：验证提示词结构
- ✅ **E2E 测试**：少量真实场景测试
- ✅ **CI/CD**：自动化运行所有测试

记住：
- 🚀 **快速反馈**：大部分测试应该在几秒内完成
- 💰 **控制成本**：避免在测试中浪费 API 调用
- 🔒 **隔离**：测试之间不应该相互影响
- 📊 **覆盖率**：追求 80%+，但不要为了 100% 而过度测试

下一章，我们将学习如何调试 Agent 问题。

---

**上一章**：[第 17 章：KAOS 抽象](./17-kaos-abstraction.md) ←
**下一章**：[第 19 章：调试技巧](./19-debugging.md) →
