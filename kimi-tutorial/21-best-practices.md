# 第 21 章：最佳实践

从头到尾构建 Agent 后，让我们总结最佳实践。本章汇集了构建生产级 Coding Agent 的经验教训。

## 21.1 架构设计

### ✅ DO: 模块化设计

**为什么**：模块化让代码易于理解、测试和维护。

```python
# ✅ 好：每个模块职责清晰
agent/
├── tools/           # 工具定义
│   ├── base.py
│   ├── file.py
│   └── shell.py
├── soul/            # 执行引擎
│   ├── engine.py
│   └── context.py
├── ui/              # 界面层
│   ├── shell.py
│   └── print.py
├── config/          # 配置管理
│   └── loader.py
└── session/         # 会话管理
    └── manager.py
```

**实践要点**：
- 每个模块聚焦单一职责
- 模块间依赖清晰
- 通过接口/协议通信

### ❌ DON'T: 上帝类

```python
# ❌ 坏：一个类做所有事
class Agent:
    def __init__(self):
        self.llm = ...
        self.tools = ...
        self.ui = ...
        self.config = ...
        self.session = ...

    def run(self):
        # 1000+ 行代码...
        # 做所有事情
```

**问题**：
- 难以理解
- 难以测试
- 难以扩展
- 职责不清

### ✅ DO: 依赖注入

```python
# ✅ 好：通过依赖注入组装
class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: List[Tool],
        ui: UI,
        config: Config
    ):
        self.llm = llm
        self.tools = tools
        self.ui = ui
        self.config = config

# 使用
agent = Agent(
    llm=OpenAILLM(api_key=...),
    tools=[ReadFile(), WriteFile()],
    ui=ShellUI(),
    config=Config.load()
)
```

**优点**：
- 依赖明确
- 易于测试（可注入 mock）
- 灵活组合

## 21.2 提示词工程

### ✅ DO: 清晰的指令

```markdown
## 工具使用指南

### 文件操作
1. **读取文件前先检查**：使用 `list_files` 确认文件存在
2. **写入文件前先读取**：了解现有内容，避免覆盖
3. **大文件分批处理**：超过 1000 行的文件，分段读取

### Shell 命令
1. **危险命令需要批准**：rm、mv、git push 等
2. **命令失败要解释**：告诉用户为什么失败，如何修复
3. **长时间命令显示进度**：让用户知道还在运行

### 错误处理
1. **遇到错误时**：
   - 解释错误原因
   - 提出解决方案
   - 询问用户意见
2. **不确定时**：主动询问，不要猜测
```

### ❌ DON'T: 模糊的指令

```markdown
# ❌ 坏：模糊不清
Be helpful and do good things.
Try your best.
```

### ✅ DO: 结构化输出

```markdown
## 输出格式

### 代码修改
当修改代码时，使用以下格式：

```
文件: src/main.py
修改: 添加错误处理
原因: 防止未捕获的异常

[代码块]
```

### 任务完成
完成任务后，提供摘要：

```
✅ 已完成:
- 重构函数 X
- 添加测试
- 更新文档

📊 统计:
- 修改文件: 3
- 添加测试: 5
- 代码行数: +120, -50
```
```

## 21.3 错误处理

### ✅ DO: 优雅降级

```python
# ✅ 好：优雅处理错误
async def execute_tool(tool: Tool, params: dict) -> str:
    """执行工具，处理错误"""
    try:
        result = await tool.execute(params)
        return result
    except PermissionError as e:
        return f"权限不足: {e}\n建议: 检查文件权限或使用 sudo"
    except FileNotFoundError as e:
        return f"文件不存在: {e}\n建议: 使用 list_files 查看可用文件"
    except TimeoutError as e:
        return f"操作超时: {e}\n建议: 增加超时时间或分批处理"
    except Exception as e:
        logger.exception("工具执行失败")
        return f"工具失败: {e}\n请尝试其他方法或寻求帮助"
```

**关键点**：
- 捕获具体异常
- 提供有用的错误信息
- 建议解决方案
- 记录日志用于调试

### ❌ DON'T: 崩溃

```python
# ❌ 坏：让程序崩溃
result = await tool.execute(params)  # 可能抛出异常，程序终止
```

### ✅ DO: 重试机制

```python
# ✅ 好：智能重试
async def call_llm_with_retry(
    client: AsyncOpenAI,
    messages: list,
    max_retries: int = 3
) -> str:
    """调用 LLM，带重试"""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content

        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                logger.warning(f"速率限制，等待 {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except APIError as e:
            logger.error(f"API 错误: {e}")
            if attempt == max_retries - 1:
                raise
```

## 21.4 安全性

### ✅ DO: 权限最小化

```python
# ✅ 好：限制工作目录
class FileTools:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir.resolve()

    def read_file(self, path: str) -> str:
        """读取文件（仅限工作目录）"""
        file_path = (self.work_dir / path).resolve()

        # 检查路径是否在工作目录内
        if not str(file_path).startswith(str(self.work_dir)):
            raise PermissionError(
                f"无法访问工作目录外的文件: {path}"
            )

        return file_path.read_text()
```

### ❌ DON'T: 无限权限

```python
# ❌ 坏：允许访问任何文件
def read_file(path: str) -> str:
    return Path(path).read_text()  # 可以读取 /etc/passwd！
```

### ✅ DO: 命令白名单

```python
# ✅ 好：危险命令需要批准
DANGEROUS_COMMANDS = ["rm", "mv", "dd", "mkfs", "reboot"]

class ShellTool:
    def __init__(self, approval_required: bool = True):
        self.approval_required = approval_required

    async def execute(self, command: str) -> str:
        # 检查是否为危险命令
        cmd_name = command.split()[0]
        if cmd_name in DANGEROUS_COMMANDS:
            if self.approval_required:
                approved = await self.request_approval(command)
                if not approved:
                    return "命令已被拒绝"

        # 执行命令...
```

### ✅ DO: 输入验证

```python
# ✅ 好：验证所有输入
def validate_tool_params(tool: Tool, params: dict):
    """验证工具参数"""
    schema = tool.get_parameter_schema()

    # 检查必需参数
    for param in schema.get("required", []):
        if param not in params:
            raise ValueError(f"缺少必需参数: {param}")

    # 检查参数类型
    for param, value in params.items():
        expected_type = schema["properties"][param]["type"]
        if not isinstance(value, type_mapping[expected_type]):
            raise TypeError(
                f"参数 {param} 类型错误，期望 {expected_type}"
            )

    # 检查参数范围
    if "path" in params:
        if ".." in params["path"]:
            raise ValueError("路径不能包含 ..")
```

## 21.5 性能优化

### ✅ DO: 缓存

```python
# ✅ 好：缓存文件内容
from functools import lru_cache

class FileCache:
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size

    def read_file(self, path: Path) -> str:
        """读取文件（带缓存）"""
        # 检查缓存
        mtime = path.stat().st_mtime
        cache_key = (str(path), mtime)

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 读取文件
        content = path.read_text()

        # 更新缓存
        if len(self.cache) >= self.max_size:
            # 移除最旧的
            oldest = min(self.cache.keys(), key=lambda k: k[1])
            del self.cache[oldest]

        self.cache[cache_key] = content
        return content
```

### ✅ DO: 批量操作

```python
# ✅ 好：批量执行工具
async def execute_tools_parallel(tool_calls: List[ToolCall]) -> List[str]:
    """并行执行多个工具"""
    tasks = [
        execute_tool(tc.tool, tc.params)
        for tc in tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    outputs = []
    for result in results:
        if isinstance(result, Exception):
            outputs.append(f"错误: {result}")
        else:
            outputs.append(result)

    return outputs
```

### ✅ DO: 流式处理

```python
# ✅ 好：流式处理大文件
async def process_large_file(path: Path, chunk_size: int = 1000):
    """分块处理大文件"""
    with open(path) as f:
        chunk = []
        for i, line in enumerate(f):
            chunk.append(line)

            if len(chunk) >= chunk_size:
                # 处理一个块
                await process_chunk(chunk)
                chunk = []

        # 处理最后一块
        if chunk:
            await process_chunk(chunk)
```

## 21.6 可观测性

### ✅ DO: 结构化日志

```python
# ✅ 好：使用结构化日志
import structlog

logger = structlog.get_logger()

class Agent:
    async def run_step(self, step: int):
        logger.info(
            "agent.step.start",
            step=step,
            session_id=self.session.id,
            model=self.config.model
        )

        try:
            result = await self.execute_step(step)

            logger.info(
                "agent.step.complete",
                step=step,
                tokens=result.tokens,
                cost=result.cost
            )

        except Exception as e:
            logger.error(
                "agent.step.failed",
                step=step,
                error=str(e),
                exc_info=True
            )
```

### ✅ DO: 指标收集

```python
# ✅ 好：收集性能指标
from dataclasses import dataclass
from typing import List

@dataclass
class Metrics:
    """性能指标"""
    total_runs: int = 0
    total_steps: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    tool_usage: dict = None

    def __post_init__(self):
        self.tool_usage = {}

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def avg_cost_per_run(self) -> float:
        return self.total_cost / self.total_runs if self.total_runs > 0 else 0.0

    def record_tool_use(self, tool_name: str):
        """记录工具使用"""
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1

    def generate_report(self) -> str:
        """生成报告"""
        lines = [
            "# Agent 性能报告",
            f"\n总运行次数: {self.total_runs}",
            f"成功率: {self.success_rate:.1%}",
            f"平均成本: ${self.avg_cost_per_run:.4f}",
            f"\n总步数: {self.total_steps}",
            f"总 Token: {self.total_tokens:,}",
            f"总成本: ${self.total_cost:.2f}",
            "\n## 工具使用统计",
        ]

        for tool, count in sorted(
            self.tool_usage.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"- {tool}: {count}")

        return "\n".join(lines)
```

### ✅ DO: 追踪和调试

```python
# ✅ 好：添加追踪信息
import uuid

class TraceContext:
    """追踪上下文"""
    def __init__(self):
        self.trace_id = str(uuid.uuid4())[:8]
        self.spans = []

    def start_span(self, name: str):
        """开始一个 span"""
        span = {
            "name": name,
            "start": time.time(),
            "trace_id": self.trace_id
        }
        self.spans.append(span)
        return span

    def end_span(self, span: dict):
        """结束 span"""
        span["end"] = time.time()
        span["duration"] = span["end"] - span["start"]

# 使用
trace = TraceContext()

span = trace.start_span("agent.run")
try:
    result = await agent.run()
finally:
    trace.end_span(span)
    logger.info("trace", spans=trace.spans)
```

## 21.7 测试策略

### ✅ DO: 单元测试

```python
# ✅ 好：测试核心逻辑
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_execute_tool_success():
    """测试工具成功执行"""
    tool = Mock()
    tool.execute = AsyncMock(return_value="success")

    result = await execute_tool(tool, {"param": "value"})
    assert result == "success"
    tool.execute.assert_called_once_with({"param": "value"})

@pytest.mark.asyncio
async def test_execute_tool_failure():
    """测试工具执行失败"""
    tool = Mock()
    tool.execute = AsyncMock(side_effect=ValueError("bad param"))

    result = await execute_tool(tool, {"param": "bad"})
    assert "错误" in result
```

### ✅ DO: 集成测试

```python
# ✅ 好：测试完整流程
@pytest.mark.asyncio
async def test_agent_end_to_end(tmp_path):
    """端到端测试"""
    # 准备
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    (work_dir / "test.txt").write_text("hello")

    # 创建 Agent
    agent = Agent(
        llm=MockLLM(),
        tools=[ReadFile(work_dir), WriteFile(work_dir)],
        ui=MockUI()
    )

    # 执行
    await agent.run("读取 test.txt 并转换为大写")

    # 验证
    content = (work_dir / "test.txt").read_text()
    assert content == "HELLO"
```

## 21.8 代码组织模式

### ✅ DO: 使用协议/接口

```python
# ✅ 好：定义清晰的接口
from typing import Protocol

class Tool(Protocol):
    """工具接口"""
    name: str
    description: str

    def get_parameter_schema(self) -> dict:
        """获取参数 schema"""
        ...

    async def execute(self, params: dict) -> str:
        """执行工具"""
        ...

# 实现
class ReadFile:
    name = "read_file"
    description = "读取文件内容"

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }

    async def execute(self, params: dict) -> str:
        path = params["path"]
        return Path(path).read_text()
```

### ✅ DO: 配置与代码分离

```python
# ✅ 好：配置文件
# config.yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 4000

tools:
  - name: read_file
    enabled: true
  - name: write_file
    enabled: true
    require_approval: true

# 代码
config = Config.from_yaml("config.yaml")
agent = Agent.from_config(config)
```

## 21.9 生产环境检查清单

### 部署前检查

```markdown
## 🔒 安全性
- [ ] 所有 API Key 通过环境变量或密钥管理服务
- [ ] 文件操作限制在工作目录内
- [ ] 危险命令需要批准
- [ ] 输入验证和清理
- [ ] 错误信息不泄露敏感数据

## ⚡ 性能
- [ ] 实现缓存机制
- [ ] 大文件分块处理
- [ ] 并行执行工具
- [ ] 设置合理的超时
- [ ] 限制最大步数

## 📊 可观测性
- [ ] 结构化日志
- [ ] 性能指标收集
- [ ] 错误追踪
- [ ] 健康检查接口
- [ ] 监控告警

## 🧪 测试
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试覆盖核心流程
- [ ] 性能测试
- [ ] 压力测试
- [ ] 错误场景测试

## 📝 文档
- [ ] API 文档
- [ ] 部署文档
- [ ] 故障排查指南
- [ ] 用户手册
- [ ] 贡献指南

## 🔧 运维
- [ ] 优雅关闭
- [ ] 会话恢复
- [ ] 自动清理
- [ ] 日志轮转
- [ ] 备份策略
```

## 21.10 反模式警告

### ❌ 反模式 1：过度工程

```python
# ❌ 坏：过度抽象
class AbstractToolFactoryBuilder:
    def build_factory(self):
        return ToolFactory(
            builder=ToolBuilder(
                validator=ToolValidator(
                    schema=SchemaBuilder().build()
                )
            )
        )

# ✅ 好：保持简单
tools = [ReadFile(), WriteFile(), ShellTool()]
```

### ❌ 反模式 2：忽略错误

```python
# ❌ 坏：吞掉异常
try:
    result = await tool.execute(params)
except:
    pass  # 静默失败

# ✅ 好：处理错误
try:
    result = await tool.execute(params)
except Exception as e:
    logger.error(f"工具执行失败: {e}")
    return f"错误: {e}"
```

### ❌ 反模式 3：硬编码

```python
# ❌ 坏：硬编码
api_key = "sk-xxx"
model = "gpt-4"
max_steps = 100

# ✅ 好：配置化
config = Config.load()
api_key = config.get_api_key()
model = config.model
max_steps = config.max_steps
```

## 21.11 FAQ

**Q: 如何平衡功能和复杂度？**

A: 遵循 YAGNI（You Aren't Gonna Need It）原则：
- 先实现核心功能
- 根据实际需求迭代
- 避免过度设计

**Q: 何时需要重构？**

A: 当出现以下情况时：
- 代码重复 > 3 次
- 函数超过 50 行
- 类超过 300 行
- 测试困难
- 频繁出 bug

**Q: 如何提高代码质量？**

A: 使用工具和流程：
- 代码审查
- 自动化测试
- Linter（如 ruff）
- Type checker（如 mypy）
- 持续集成

## 21.12 练习

### 练习 1: 实现健康检查

添加健康检查接口：

```python
class HealthCheck:
    async def check(self) -> dict:
        """检查系统健康状态"""
        # TODO: 检查 LLM 连接
        # TODO: 检查文件系统
        # TODO: 检查会话存储
        pass
```

### 练习 2: 实现速率限制

添加请求速率限制：

```python
class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        # TODO: 实现令牌桶或滑动窗口算法
        pass

    async def acquire(self):
        # TODO: 获取令牌，超过限制则等待
        pass
```

### 练习 3: 实现审计日志

记录所有重要操作：

```python
class AuditLog:
    def log_action(
        self,
        action: str,
        user: str,
        resource: str,
        result: str
    ):
        # TODO: 记录审计日志
        pass
```

## 21.13 小结

本章总结了构建 Agent 的最佳实践：

- 🏗️ **架构设计**：模块化、依赖注入
- 📝 **提示词工程**：清晰指令、结构化输出
- 🛡️ **安全性**：最小权限、输入验证
- ⚡ **性能优化**：缓存、批量、流式
- 📊 **可观测性**：日志、指标、追踪
- 🧪 **测试策略**：单元测试、集成测试
- 🚀 **生产就绪**：完整的检查清单

**核心原则**：

1. **简单优于复杂**：保持代码简洁
2. **安全第一**：不信任任何输入
3. **可观测**：记录一切重要信息
4. **可测试**：编写测试就是最好的文档
5. **渐进式**：先让它工作，再让它优雅

记住：好的代码不是一次写成的，而是不断重构出来的。

---

**上一章**：[第 20 章：部署和分发](./20-deployment.md) ←
**下一章**：[第 22 章：未来展望](./22-future.md) →
