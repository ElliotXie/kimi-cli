# 第 20 章：部署和分发

你的 Agent 开发完成了！现在需要让用户能够使用它。

从开发到生产，有很多事情要做：

```
开发环境                     生产环境
──────────                   ──────────
python main.py              → pip install my-agent
                            → docker run my-agent
                            → 云端服务
                            → CI/CD 自动部署
```

本章教你如何专业地部署和分发 Agent。

## 20.1 打包策略

### 现代 Python 打包

使用 `pyproject.toml` (PEP 518):

```toml
# pyproject.toml

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-agent"
version = "1.0.0"
description = "An AI coding agent"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]

# 依赖
dependencies = [
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "rich>=13.0.0",
    "click>=8.0.0",  # CLI 框架
]

# 可选依赖
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
test = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
]

# 命令行入口点
[project.scripts]
my-agent = "my_agent.cli:main"
my-agent-debug = "my_agent.cli:debug_main"

# 项目 URL
[project.urls]
Homepage = "https://github.com/user/my-agent"
Documentation = "https://my-agent.readthedocs.io"
Repository = "https://github.com/user/my-agent"
Issues = "https://github.com/user/my-agent/issues"
```

### 目录结构

```
my-agent/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── my_agent/
│   ├── __init__.py
│   ├── __version__.py      # 版本号
│   ├── cli.py              # CLI 入口
│   ├── agent.py            # Agent 核心
│   ├── tools/
│   │   ├── __init__.py
│   │   └── ...
│   ├── kaos/
│   │   └── ...
│   └── config/
│       └── default.json
├── tests/
│   ├── __init__.py
│   └── test_*.py
└── docs/
    └── ...
```

### 版本管理

```python
# my_agent/__version__.py

__version__ = "1.0.0"
__version_info__ = tuple(int(i) for i in __version__.split("."))
```

```python
# my_agent/__init__.py

from my_agent.__version__ import __version__

__all__ = ["__version__", "Agent"]
```

### 构建和发布

```bash
# 安装构建工具
pip install build twine

# 构建
python -m build
# 生成:
#   dist/my-agent-1.0.0.tar.gz
#   dist/my-agent-1.0.0-py3-none-any.whl

# 检查包
twine check dist/*

# 上传到 TestPyPI（测试）
twine upload --repository testpypi dist/*

# 测试安装
pip install --index-url https://test.pypi.org/simple/ my-agent

# 上传到 PyPI（生产）
twine upload dist/*
```

用户安装：

```bash
pip install my-agent
my-agent --help
```

## 20.2 Docker 部署

### 基础 Dockerfile

```dockerfile
# Dockerfile

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# 复制代码
COPY my_agent/ ./my_agent/

# 创建非 root 用户
RUN useradd -m -u 1000 agent && \
    chown -R agent:agent /app
USER agent

# 默认命令
CMD ["my-agent"]
```

### 多阶段构建（优化镜像大小）

```dockerfile
# Dockerfile.multi-stage

# 阶段 1: 构建
FROM python:3.11-slim as builder

WORKDIR /build

# 安装构建依赖
RUN pip install --no-cache-dir build

# 复制源代码
COPY . .

# 构建 wheel
RUN python -m build --wheel

# 阶段 2: 运行时
FROM python:3.11-slim

WORKDIR /app

# 从构建阶段复制 wheel
COPY --from=builder /build/dist/*.whl .

# 安装
RUN pip install --no-cache-dir *.whl && \
    rm *.whl

# 创建用户
RUN useradd -m -u 1000 agent
USER agent

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s \
  CMD my-agent --version || exit 1

CMD ["my-agent"]
```

### 使用 Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  agent:
    build: .
    image: my-agent:latest
    container_name: my-agent
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AGENT_LOG_LEVEL=INFO
      - AGENT_MAX_STEPS=100
    volumes:
      # 挂载工作目录
      - ./workspace:/workspace
      # 挂载配置
      - ./config.json:/app/config.json:ro
    working_dir: /workspace
    restart: unless-stopped

  # 可选：监控服务
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### 构建和运行

```bash
# 构建镜像
docker build -t my-agent:1.0.0 .

# 运行
docker run -it \
  -e OPENAI_API_KEY="sk-..." \
  -v $(pwd)/workspace:/workspace \
  my-agent:1.0.0

# 使用 docker-compose
docker-compose up -d

# 查看日志
docker-compose logs -f agent
```

## 20.3 无服务器部署

### AWS Lambda

```python
# lambda_handler.py

import json
import os
from my_agent import Agent
from kaos.memory import MemoryKaos

def lambda_handler(event, context):
    """AWS Lambda 处理函数"""

    # 从事件获取用户输入
    user_input = event.get("input", "")

    # 创建 Agent
    kaos = MemoryKaos()
    agent = Agent(
        llm_api_key=os.environ["OPENAI_API_KEY"],
        kaos=kaos
    )

    # 执行
    try:
        result = asyncio.run(agent.run(user_input))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "result": result,
                "tokens": agent.total_tokens
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }
```

```dockerfile
# Dockerfile.lambda (AWS Lambda 容器镜像)

FROM public.ecr.aws/lambda/python:3.11

# 复制依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY my_agent/ ${LAMBDA_TASK_ROOT}/my_agent/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# 设置处理函数
CMD ["lambda_handler.lambda_handler"]
```

### Google Cloud Functions

```python
# main.py

import functions_framework
from my_agent import Agent

@functions_framework.http
def agent_endpoint(request):
    """HTTP Cloud Function"""

    # 获取请求
    request_json = request.get_json(silent=True)

    if not request_json or "input" not in request_json:
        return {"error": "Missing 'input' field"}, 400

    # 运行 Agent
    agent = Agent(...)
    result = asyncio.run(agent.run(request_json["input"]))

    return {
        "result": result,
        "status": "success"
    }
```

```yaml
# cloudbuild.yaml

steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - functions
      - deploy
      - agent-function
      - --runtime=python311
      - --trigger-http
      - --entry-point=agent_endpoint
      - --set-env-vars=OPENAI_API_KEY=${_OPENAI_API_KEY}
```

## 20.4 配置管理

### 环境变量

```python
# my_agent/config/env.py

import os
from typing import Optional

class EnvConfig:
    """环境变量配置"""

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4-turbo")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Agent
    AGENT_MAX_STEPS: int = int(os.getenv("AGENT_MAX_STEPS", "100"))
    AGENT_DEBUG: bool = os.getenv("AGENT_DEBUG", "false").lower() == "true"

    # 日志
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

    # 安全
    WORK_DIR: str = os.getenv("WORK_DIR", "/workspace")
    READONLY_MODE: bool = os.getenv("READONLY_MODE", "false").lower() == "true"

    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

        if cls.AGENT_MAX_STEPS <= 0:
            raise ValueError("AGENT_MAX_STEPS must be > 0")
```

### .env 文件

```bash
# .env.example (检入版本控制)

# LLM 配置
OPENAI_API_KEY=your-api-key-here
LLM_MODEL=gpt-4-turbo
LLM_TEMPERATURE=0.7

# Agent 配置
AGENT_MAX_STEPS=100
AGENT_DEBUG=false

# 日志
LOG_LEVEL=INFO
LOG_FILE=

# 工作目录
WORK_DIR=/workspace
READONLY_MODE=false
```

```bash
# .env (不检入版本控制，复制 .env.example)
OPENAI_API_KEY=sk-actual-key-here
```

加载 .env 文件：

```python
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

# 现在可以使用环境变量
from my_agent.config.env import EnvConfig

EnvConfig.validate()
```

### 分环境配置

```python
# my_agent/config/__init__.py

import os
from pathlib import Path
import json

def load_config():
    """根据环境加载配置"""

    env = os.getenv("ENV", "development")

    # 配置文件路径
    config_dir = Path(__file__).parent
    config_file = config_dir / f"{env}.json"

    if not config_file.exists():
        config_file = config_dir / "default.json"

    with open(config_file) as f:
        return json.load(f)

# 配置文件:
# config/default.json
# config/development.json
# config/production.json
```

## 20.5 监控和可观测性

### 指标收集

```python
# my_agent/metrics.py

from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class Metrics:
    """Agent 指标"""

    # 计数器
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    # LLM
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # 工具
    tool_calls: Dict[str, int] = None

    # 性能
    total_duration: float = 0.0

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = {}

    def record_run(self, success: bool, duration: float):
        """记录一次运行"""
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        self.total_duration += duration

    def record_llm_call(self, tokens: int, cost: float):
        """记录 LLM 调用"""
        self.total_llm_calls += 1
        self.total_tokens += tokens
        self.total_cost += cost

    def record_tool_call(self, tool_name: str):
        """记录工具调用"""
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.successful_runs / max(self.total_runs, 1),
            "llm": {
                "total_calls": self.total_llm_calls,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "avg_tokens": self.total_tokens // max(self.total_llm_calls, 1),
            },
            "tools": self.tool_calls,
            "performance": {
                "total_duration": self.total_duration,
                "avg_duration": self.total_duration / max(self.total_runs, 1),
            }
        }
```

### Prometheus 导出

```python
# my_agent/prometheus.py

from prometheus_client import Counter, Histogram, Gauge, generate_latest

# 定义指标
agent_runs_total = Counter(
    "agent_runs_total",
    "Total number of agent runs",
    ["status"]  # success/failure
)

agent_llm_calls_total = Counter(
    "agent_llm_calls_total",
    "Total number of LLM calls"
)

agent_tokens_total = Counter(
    "agent_tokens_total",
    "Total tokens used"
)

agent_cost_total = Counter(
    "agent_cost_total",
    "Total cost in USD"
)

agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Agent execution duration",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

agent_tool_calls_total = Counter(
    "agent_tool_calls_total",
    "Total tool calls",
    ["tool"]
)

# 在 Agent 中使用
class Agent:
    async def run(self, user_input: str):
        start = time.time()

        try:
            result = await self._run_impl(user_input)

            # 记录成功
            agent_runs_total.labels(status="success").inc()

            return result

        except Exception as e:
            # 记录失败
            agent_runs_total.labels(status="failure").inc()
            raise

        finally:
            # 记录耗时
            duration = time.time() - start
            agent_duration_seconds.observe(duration)

    async def execute_tool(self, tool_call):
        # 记录工具调用
        agent_tool_calls_total.labels(tool=tool_call.name).inc()
        return await super().execute_tool(tool_call)

# HTTP 端点导出指标
from flask import Flask, Response

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")
```

### 结构化日志

```python
# 使用 structlog

import structlog

logger = structlog.get_logger()

class Agent:
    async def run(self, user_input: str):
        logger.info(
            "agent_run_started",
            user_input=user_input,
            session_id=self.session_id
        )

        # ...

        logger.info(
            "agent_run_completed",
            session_id=self.session_id,
            tokens_used=self.total_tokens,
            duration=duration
        )
```

输出（JSON 格式）：

```json
{
  "event": "agent_run_started",
  "user_input": "read README.md",
  "session_id": "a1b2c3d4",
  "timestamp": "2025-01-15T10:30:00.123Z"
}
{
  "event": "agent_run_completed",
  "session_id": "a1b2c3d4",
  "tokens_used": 1234,
  "duration": 2.5,
  "timestamp": "2025-01-15T10:30:02.623Z"
}
```

## 20.6 CI/CD 流水线

### GitHub Actions 完整示例

```yaml
# .github/workflows/ci.yml

name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  PYTHON_VERSION: '3.11'

jobs:
  # 测试
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"

    - name: Run linters
      run: |
        black --check my_agent/
        ruff check my_agent/

    - name: Run tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/ -v --cov=my_agent --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  # 构建
  build:
    needs: test
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Build package
      run: |
        pip install build
        python -m build

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  # Docker 镜像
  docker:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: username/my-agent
        tags: |
          type=ref,event=branch
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # 发布到 PyPI
  publish:
    needs: [test, build]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'

    steps:
    - uses: actions/checkout@v4

    - name: Download artifacts
      uses: actions/download-artifact@v3
      with:
        name: dist
        path: dist/

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## 20.7 安全最佳实践

### 1. 密钥管理

```python
# ❌ 坏：硬编码
api_key = "sk-1234567890abcdef"

# ✅ 好：环境变量
import os
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
```

### 2. 输入验证

```python
# my_agent/security.py

def validate_user_input(user_input: str) -> str:
    """验证和清理用户输入"""

    # 长度限制
    if len(user_input) > 10000:
        raise ValueError("Input too long")

    # 禁止的模式
    forbidden_patterns = [
        r"rm\s+-rf",  # 危险命令
        r"eval\(",    # 代码注入
        # ...
    ]

    import re
    for pattern in forbidden_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValueError(f"Forbidden pattern detected: {pattern}")

    return user_input
```

### 3. 沙箱隔离

```python
# 使用 KAOS 限制文件访问
from kaos.local import LocalKaos
from pathlib import Path

# ✅ 好：限制在项目目录
kaos = LocalKaos(
    work_dir=Path("/workspace/project"),
    readonly=False  # 根据需要设置
)

# ❌ 坏：允许访问整个文件系统
kaos = LocalKaos(work_dir=Path("/"))
```

### 4. 速率限制

```python
# my_agent/ratelimit.py

from functools import wraps
import time

class RateLimiter:
    """简单的速率限制器"""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()

            # 清理过期记录
            self.calls = [t for t in self.calls if now - t < self.period]

            # 检查限制
            if len(self.calls) >= self.max_calls:
                wait_time = self.period - (now - self.calls[0])
                raise RateLimitExceeded(f"Rate limit exceeded. Retry in {wait_time:.1f}s")

            # 记录调用
            self.calls.append(now)

            return await func(*args, **kwargs)

        return wrapper

# 使用
@RateLimiter(max_calls=10, period=60.0)  # 每分钟最多 10 次
async def agent_run(user_input: str):
    ...
```

## 20.8 性能优化

### 1. 连接池

```python
# 复用 HTTP 连接
from openai import AsyncOpenAI

# ✅ 好：单例客户端
class LLMClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                max_retries=3,
                timeout=30.0
            )
        return cls._instance
```

### 2. 缓存

```python
# my_agent/cache.py

from functools import lru_cache
import hashlib
import json

class LLMCache:
    """LLM 响应缓存"""

    def __init__(self):
        self.cache = {}

    def get_key(self, messages, **kwargs) -> str:
        """生成缓存键"""
        data = {
            "messages": messages,
            **kwargs
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    async def get_or_call(self, llm_func, messages, **kwargs):
        """获取缓存或调用 LLM"""
        key = self.get_key(messages, **kwargs)

        if key in self.cache:
            return self.cache[key]

        result = await llm_func(messages, **kwargs)
        self.cache[key] = result

        return result
```

### 3. 异步并发

```python
# 并发执行工具
import asyncio

async def execute_tools_parallel(tool_calls):
    """并发执行多个工具"""
    tasks = [execute_tool(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## 20.9 FAQ

**Q: 如何选择部署方式？**

A:
- **PyPI**: 命令行工具，用户自己运行
- **Docker**: 标准化环境，易于部署
- **Serverless**: 按需运行，无需管理服务器
- **云 VM**: 完全控制，适合复杂场景

**Q: 生产环境应该用什么日志级别？**

A: INFO 级别。WARNING 可能错过重要信息，DEBUG 太详细且影响性能。

**Q: 如何处理 API 密钥泄露？**

A:
1. 立即撤销泄露的密钥
2. 生成新密钥
3. 检查是否有异常使用
4. 使用密钥扫描工具（git-secrets, truffleHog）

**Q: Docker 镜像太大怎么办？**

A:
- 使用 `slim` 基础镜像
- 多阶段构建
- `.dockerignore` 排除不需要的文件
- 只安装生产依赖

## 20.10 练习

### 练习 1：健康检查端点

实现一个健康检查 HTTP 端点：

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    # TODO: 检查 Agent 状态
    # - LLM API 可达性
    # - 文件系统可访问性
    # - 内存使用
    return jsonify({"status": "healthy"})
```

### 练习 2：蓝绿部署

设计一个蓝绿部署策略，允许无停机升级。

### 练习 3：自动化回滚

当部署后错误率超过阈值时，自动回滚到前一个版本。

## 20.11 小结

部署 Agent 的关键要素：

- 📦 **打包**: 使用现代 Python 工具（pyproject.toml）
- 🐳 **容器化**: Docker 确保环境一致性
- 🔧 **配置**: 环境变量 + 配置文件
- 📊 **监控**: 日志 + 指标 + 追踪
- 🚀 **CI/CD**: 自动化测试和部署
- 🔒 **安全**: 密钥管理 + 输入验证 + 沙箱
- ⚡ **性能**: 缓存 + 连接池 + 异步

记住：
- 🎯 **自动化一切**：从测试到部署
- 🔍 **可观测性第一**：你看不到的就无法改进
- 🛡️ **安全优先**：永远不要信任用户输入
- 📈 **渐进式发布**：金丝雀 → 蓝绿 → 全量

下一章，我们将总结最佳实践。

---

**上一章**：[第 19 章：调试技巧](./19-debugging.md) ←
**下一章**：[第 21 章：最佳实践](./21-best-practices.md) →
