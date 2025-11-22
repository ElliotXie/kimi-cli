# 第 17 章：KAOS 抽象层

KAOS = **K**imi **A**gent **O**perating **S**ystem

## 17.1 为什么需要操作系统抽象？

想象一个场景：你开发了一个出色的 Coding Agent，它能在你的笔记本电脑上完美运行。但是：

```
开发者 Alice: "我想在 Docker 容器里跑这个 Agent"
开发者 Bob: "我需要在云端服务器上运行"
开发者 Carol: "我要在 CI/CD 管道中使用"
测试工程师: "我需要在测试中模拟文件系统"
```

每种场景都需要不同的文件操作方式。如果你的工具直接使用 `open()` 和 `Path().read_text()`，你就麻烦了：

```python
# ❌ 问题代码：工具直接操作文件系统
class ReadFileTool:
    async def execute(self, params):
        # 这只能在本地文件系统工作！
        with open(params.path, 'r') as f:
            return f.read()
```

这个工具：
- 🚫 无法在远程服务器运行
- 🚫 无法在 Docker 容器中使用
- 🚫 测试时无法 mock
- 🚫 无法添加安全限制

**解决方案**：引入操作系统抽象层 - KAOS。

### 抽象层的类比

想象一下电脑的操作系统：

```
应用程序
    ↓
操作系统接口（Windows API / POSIX）
    ↓
实际硬件（硬盘、内存、CPU）
```

应用程序不直接操作硬盘，而是通过操作系统提供的统一接口。这样：
- ✅ 同样的程序可以在不同硬盘上运行
- ✅ 可以在虚拟机中运行
- ✅ 操作系统可以添加权限控制
- ✅ 可以切换存储后端（SSD、HDD、网络存储）

KAOS 为 Agent 做同样的事情：

```
Agent 工具
    ↓
KAOS 接口（统一的文件操作）
    ↓
具体实现（本地、Docker、云端、Mock）
```

## 17.2 KAOS 协议设计

让我们设计完整的 KAOS 协议：

```python
# kaos/__init__.py

from typing import Protocol, Iterator
from pathlib import Path

class Kaos(Protocol):
    """Kimi Agent Operating System 协议

    定义 Agent 需要的所有文件系统操作。
    任何实现都必须提供这些方法。
    """

    # 目录操作
    def getcwd(self) -> Path:
        """获取当前工作目录"""
        ...

    def chdir(self, path: str) -> None:
        """切换工作目录"""
        ...

    def listdir(self, path: str = ".") -> list[str]:
        """列出目录内容"""
        ...

    def mkdir(self, path: str, parents: bool = False) -> None:
        """创建目录"""
        ...

    # 文件读写
    def readtext(self, path: str, encoding: str = "utf-8") -> str:
        """读取文本文件"""
        ...

    def writetext(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """写入文本文件"""
        ...

    def readbytes(self, path: str) -> bytes:
        """读取二进制文件"""
        ...

    def writebytes(self, path: str, content: bytes) -> None:
        """写入二进制文件"""
        ...

    # 文件查询
    def exists(self, path: str) -> bool:
        """检查路径是否存在"""
        ...

    def is_file(self, path: str) -> bool:
        """检查是否为文件"""
        ...

    def is_dir(self, path: str) -> bool:
        """检查是否为目录"""
        ...

    def glob(self, pattern: str) -> list[Path]:
        """文件模式匹配"""
        ...

    # 文件操作
    def remove(self, path: str) -> None:
        """删除文件"""
        ...

    def rename(self, old: str, new: str) -> None:
        """重命名文件"""
        ...

    def copy(self, src: str, dst: str) -> None:
        """复制文件"""
        ...

    # 元数据
    def stat(self, path: str) -> dict:
        """获取文件元数据（大小、修改时间等）"""
        ...
```

### 为什么用 Protocol？

Python 的 `Protocol` 是一种"结构化类型"（structural typing）：

```python
# 不需要继承 Kaos，只要实现了这些方法就行
class MyKaos:
    def readtext(self, path: str) -> str:
        return "..."

    def writetext(self, path: str, content: str):
        pass

    # ... 其他方法

# 类型检查通过！
kaos: Kaos = MyKaos()  # ✅ OK
```

这比传统的抽象基类更灵活。

## 17.3 本地文件系统实现

最基础的实现 - 直接操作本地文件：

```python
# kaos/local.py

import shutil
from pathlib import Path
from typing import Iterator

class LocalKaos:
    """本地文件系统实现

    直接操作本地磁盘上的文件。
    这是最常用的实现。
    """

    def __init__(self, work_dir: Path, readonly: bool = False):
        """初始化本地 KAOS

        Args:
            work_dir: 工作目录（Agent 的"根目录"）
            readonly: 是否只读模式（安全起见）
        """
        self.work_dir = work_dir.resolve()  # 转换为绝对路径
        self.readonly = readonly

        # 确保工作目录存在
        if not self.work_dir.exists():
            raise ValueError(f"工作目录不存在: {self.work_dir}")

    def _resolve_path(self, path: str) -> Path:
        """解析路径，确保在工作目录内

        这是安全关键！防止路径遍历攻击：
        - "../../../etc/passwd" ❌
        - "/etc/passwd" ❌
        - "project/src/main.py" ✅
        """
        target = (self.work_dir / path).resolve()

        # 检查是否在工作目录内
        try:
            target.relative_to(self.work_dir)
        except ValueError:
            raise PermissionError(
                f"路径 {path} 在工作目录外，拒绝访问！"
            )

        return target

    def _check_write(self):
        """检查是否允许写入"""
        if self.readonly:
            raise PermissionError("只读模式，禁止写入")

    # 目录操作
    def getcwd(self) -> Path:
        return self.work_dir

    def chdir(self, path: str):
        new_dir = self._resolve_path(path)
        if not new_dir.is_dir():
            raise NotADirectoryError(f"{path} 不是目录")
        self.work_dir = new_dir

    def listdir(self, path: str = ".") -> list[str]:
        dir_path = self._resolve_path(path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"{path} 不是目录")
        return [item.name for item in dir_path.iterdir()]

    def mkdir(self, path: str, parents: bool = False):
        self._check_write()
        dir_path = self._resolve_path(path)
        dir_path.mkdir(parents=parents, exist_ok=False)

    # 文件读写
    def readtext(self, path: str, encoding: str = "utf-8") -> str:
        file_path = self._resolve_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            raise ValueError(f"文件 {path} 不是有效的 {encoding} 文本")

    def writetext(self, path: str, content: str, encoding: str = "utf-8"):
        self._check_write()
        file_path = self._resolve_path(path)

        # 确保父目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)

    def readbytes(self, path: str) -> bytes:
        file_path = self._resolve_path(path)
        return file_path.read_bytes()

    def writebytes(self, path: str, content: bytes):
        self._check_write()
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    # 文件查询
    def exists(self, path: str) -> bool:
        try:
            return self._resolve_path(path).exists()
        except PermissionError:
            return False  # 路径在工作目录外

    def is_file(self, path: str) -> bool:
        try:
            return self._resolve_path(path).is_file()
        except PermissionError:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            return self._resolve_path(path).is_dir()
        except PermissionError:
            return False

    def glob(self, pattern: str) -> list[Path]:
        """文件模式匹配

        例如：
        - "*.py" - 所有 Python 文件
        - "**/*.md" - 递归查找所有 Markdown 文件
        - "src/**/*.ts" - src 目录下所有 TypeScript 文件
        """
        results = []
        for match in self.work_dir.glob(pattern):
            # 返回相对路径
            try:
                rel_path = match.relative_to(self.work_dir)
                results.append(rel_path)
            except ValueError:
                pass  # 忽略工作目录外的匹配
        return results

    # 文件操作
    def remove(self, path: str):
        self._check_write()
        file_path = self._resolve_path(path)
        if file_path.is_dir():
            shutil.rmtree(file_path)
        else:
            file_path.unlink()

    def rename(self, old: str, new: str):
        self._check_write()
        old_path = self._resolve_path(old)
        new_path = self._resolve_path(new)
        old_path.rename(new_path)

    def copy(self, src: str, dst: str):
        self._check_write()
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    # 元数据
    def stat(self, path: str) -> dict:
        file_path = self._resolve_path(path)
        stat_info = file_path.stat()

        return {
            "size": stat_info.st_size,
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
        }
```

## 17.4 内存文件系统实现

测试时，你不想真的创建文件。用内存实现：

```python
# kaos/memory.py

from pathlib import Path
from typing import Dict
import time

class MemoryKaos:
    """内存文件系统

    所有文件都存在内存中，重启后消失。
    非常适合测试！
    """

    def __init__(self):
        # 用字典模拟文件系统
        self.files: Dict[str, str | bytes] = {}
        self.dirs: set[str] = {"."}  # 根目录总是存在
        self.cwd = "."

        # 元数据
        self.metadata: Dict[str, dict] = {}

    def _normalize_path(self, path: str) -> str:
        """规范化路径"""
        if path.startswith("/"):
            path = path[1:]

        # 处理当前目录
        if self.cwd != ".":
            path = f"{self.cwd}/{path}"

        # 简化路径（去掉 "./" 和 "../"）
        parts = []
        for part in path.split("/"):
            if part == "." or part == "":
                continue
            elif part == "..":
                if parts:
                    parts.pop()
            else:
                parts.append(part)

        return "/".join(parts) if parts else "."

    def getcwd(self) -> Path:
        return Path(self.cwd)

    def chdir(self, path: str):
        norm_path = self._normalize_path(path)
        if norm_path not in self.dirs:
            raise NotADirectoryError(f"{path} 不是目录")
        self.cwd = norm_path

    def listdir(self, path: str = ".") -> list[str]:
        norm_path = self._normalize_path(path)
        if norm_path not in self.dirs:
            raise NotADirectoryError(f"{path} 不是目录")

        # 查找此目录下的直接子项
        prefix = norm_path + "/" if norm_path != "." else ""
        items = set()

        for file_path in self.files:
            if file_path.startswith(prefix):
                relative = file_path[len(prefix):]
                if "/" not in relative:  # 直接子文件
                    items.add(relative)

        for dir_path in self.dirs:
            if dir_path.startswith(prefix) and dir_path != norm_path:
                relative = dir_path[len(prefix):]
                first_part = relative.split("/")[0]
                items.add(first_part)

        return sorted(items)

    def mkdir(self, path: str, parents: bool = False):
        norm_path = self._normalize_path(path)

        if norm_path in self.dirs:
            raise FileExistsError(f"目录已存在: {path}")

        # 检查父目录
        parent = "/".join(norm_path.split("/")[:-1]) or "."
        if not parents and parent not in self.dirs:
            raise FileNotFoundError(f"父目录不存在: {parent}")

        # 创建目录（以及必要的父目录）
        if parents:
            parts = norm_path.split("/")
            for i in range(len(parts)):
                dir_path = "/".join(parts[:i+1])
                self.dirs.add(dir_path)
        else:
            self.dirs.add(norm_path)

    def readtext(self, path: str, encoding: str = "utf-8") -> str:
        norm_path = self._normalize_path(path)

        if norm_path not in self.files:
            raise FileNotFoundError(f"文件不存在: {path}")

        content = self.files[norm_path]
        if isinstance(content, bytes):
            return content.decode(encoding)
        return content

    def writetext(self, path: str, content: str, encoding: str = "utf-8"):
        norm_path = self._normalize_path(path)

        # 确保父目录存在
        parent = "/".join(norm_path.split("/")[:-1]) or "."
        if parent not in self.dirs:
            self.mkdir(parent, parents=True)

        self.files[norm_path] = content
        self.metadata[norm_path] = {
            "created": time.time(),
            "modified": time.time(),
            "size": len(content),
        }

    def exists(self, path: str) -> bool:
        norm_path = self._normalize_path(path)
        return norm_path in self.files or norm_path in self.dirs

    def is_file(self, path: str) -> bool:
        norm_path = self._normalize_path(path)
        return norm_path in self.files

    def is_dir(self, path: str) -> bool:
        norm_path = self._normalize_path(path)
        return norm_path in self.dirs

    def glob(self, pattern: str) -> list[Path]:
        """简化的 glob 实现"""
        import fnmatch

        results = []
        for file_path in self.files:
            if fnmatch.fnmatch(file_path, pattern):
                results.append(Path(file_path))

        return results
```

## 17.5 在工具中使用 KAOS

现在改造我们的工具来使用 KAOS：

```python
# tools/read_file.py

from kaos import Kaos

class ReadFileTool:
    """读取文件工具 - KAOS 版本"""

    def __init__(self, kaos: Kaos):
        self.kaos = kaos  # 依赖抽象，不是具体实现

    async def execute(self, params: dict) -> str:
        """执行文件读取

        现在这个工具可以：
        - 在本地文件系统运行 ✅
        - 在内存文件系统运行（测试）✅
        - 在远程文件系统运行 ✅
        - 在 Docker 容器中运行 ✅
        """
        path = params["path"]

        # 使用 KAOS 接口，而不是直接 open()
        try:
            content = self.kaos.readtext(path)
            return f"文件内容：\n{content}"
        except FileNotFoundError:
            return f"错误：文件 {path} 不存在"
        except PermissionError:
            return f"错误：没有权限读取 {path}"


# tools/write_file.py

class WriteFileTool:
    """写入文件工具 - KAOS 版本"""

    def __init__(self, kaos: Kaos):
        self.kaos = kaos

    async def execute(self, params: dict) -> str:
        path = params["path"]
        content = params["content"]

        try:
            self.kaos.writetext(path, content)
            return f"成功写入文件: {path}"
        except PermissionError:
            return f"错误：没有权限写入 {path}"


# tools/search_files.py

class SearchFilesTool:
    """搜索文件工具 - KAOS 版本"""

    def __init__(self, kaos: Kaos):
        self.kaos = kaos

    async def execute(self, params: dict) -> str:
        pattern = params["pattern"]

        # 使用 KAOS 的 glob
        matches = self.kaos.glob(pattern)

        if not matches:
            return f"没有找到匹配 {pattern} 的文件"

        files_list = "\n".join(f"- {m}" for m in matches)
        return f"找到 {len(matches)} 个文件：\n{files_list}"
```

## 17.6 完整使用示例

```python
# 示例 1：本地开发
from kaos.local import LocalKaos
from pathlib import Path

# 创建本地 KAOS，限制在项目目录
kaos = LocalKaos(work_dir=Path("/home/user/my-project"))

# 创建工具
read_tool = ReadFileTool(kaos)
write_tool = WriteFileTool(kaos)

# Agent 使用工具
await read_tool.execute({"path": "README.md"})  # ✅ 可以读
await write_tool.execute({"path": "../../../etc/passwd", "content": "hack"})  # ❌ 被拒绝！


# 示例 2：测试环境
from kaos.memory import MemoryKaos

# 创建内存文件系统
kaos = MemoryKaos()

# 准备测试数据
kaos.writetext("test.txt", "test content")

# 创建工具
read_tool = ReadFileTool(kaos)

# 测试
result = await read_tool.execute({"path": "test.txt"})
assert "test content" in result  # ✅ 测试通过

# 测试结束，内存自动清理，没有留下任何文件


# 示例 3：只读模式（安全分析）
kaos = LocalKaos(work_dir=Path("/home/user/code"), readonly=True)

read_tool = ReadFileTool(kaos)
write_tool = WriteFileTool(kaos)

await read_tool.execute({"path": "main.py"})  # ✅ 可以读
await write_tool.execute({"path": "main.py", "content": "..."})  # ❌ 只读模式，拒绝
```

## 17.7 常见陷阱与解决方案

### 陷阱 1：忘记路径验证

```python
# ❌ 危险！
def readtext(self, path: str) -> str:
    return Path(path).read_text()  # 可以读任何文件！

# ✅ 安全
def readtext(self, path: str) -> str:
    safe_path = self._resolve_path(path)  # 验证路径
    return safe_path.read_text()
```

### 陷阱 2：硬编码文件操作

```python
# ❌ 问题：工具直接操作文件
class MyTool:
    async def execute(self, params):
        with open(params.path) as f:  # 无法测试！
            return f.read()

# ✅ 解决：依赖 KAOS
class MyTool:
    def __init__(self, kaos: Kaos):
        self.kaos = kaos

    async def execute(self, params):
        return self.kaos.readtext(params.path)  # 可测试！
```

### 陷阱 3：假设本地文件系统

```python
# ❌ 问题：假设可以直接访问
def get_file_size(path: str) -> int:
    return os.path.getsize(path)  # 在内存 KAOS 中不工作

# ✅ 解决：使用 KAOS 接口
def get_file_size(kaos: Kaos, path: str) -> int:
    stat = kaos.stat(path)
    return stat["size"]
```

## 17.8 最佳实践

### 1. 始终通过 KAOS 操作文件

```python
# ✅ 好
content = kaos.readtext("file.txt")

# ❌ 坏
content = open("file.txt").read()
```

### 2. 工具接受 KAOS 作为依赖

```python
# ✅ 好：依赖注入
class Tool:
    def __init__(self, kaos: Kaos):
        self.kaos = kaos

# ❌ 坏：创建自己的 KAOS
class Tool:
    def __init__(self):
        self.kaos = LocalKaos(Path.cwd())  # 无法替换！
```

### 3. 在测试中使用 MemoryKaos

```python
# ✅ 好：快速、隔离的测试
def test_read_file():
    kaos = MemoryKaos()
    kaos.writetext("test.txt", "hello")

    tool = ReadFileTool(kaos)
    result = await tool.execute({"path": "test.txt"})

    assert "hello" in result
```

## 17.9 FAQ

**Q: KAOS 和真实 OS 有什么区别？**

A: KAOS 是一个轻量级抽象，只包含 Agent 需要的文件操作。真实 OS 提供进程管理、网络、硬件访问等。

**Q: 为什么不直接用 `pathlib`？**

A: `pathlib` 总是操作真实文件系统。KAOS 允许我们切换后端（内存、远程等）。

**Q: 性能如何？**

A: `LocalKaos` 只是薄薄一层包装，性能开销极小（<1%）。`MemoryKaos` 实际上更快，因为不涉及磁盘 I/O。

**Q: 可以嵌套 KAOS 吗？**

A: 可以！例如，在 `LocalKaos` 外包装一个日志层：

```python
class LoggingKaos:
    def __init__(self, inner: Kaos):
        self.inner = inner

    def readtext(self, path: str) -> str:
        logger.info(f"Reading {path}")
        return self.inner.readtext(path)
```

## 17.10 练习

### 练习 1：实现只读 KAOS

创建一个 `ReadOnlyKaos` 包装器，拒绝所有写操作：

```python
class ReadOnlyKaos:
    def __init__(self, inner: Kaos):
        self.inner = inner

    def readtext(self, path: str) -> str:
        # TODO: 委托给 inner
        pass

    def writetext(self, path: str, content: str):
        # TODO: 抛出异常
        pass
```

### 练习 2：添加缓存

实现一个 `CachedKaos`，缓存文件读取结果：

```python
class CachedKaos:
    def __init__(self, inner: Kaos):
        self.inner = inner
        self.cache = {}

    def readtext(self, path: str) -> str:
        if path in self.cache:
            return self.cache[path]

        content = self.inner.readtext(path)
        self.cache[path] = content
        return content
```

### 练习 3：文件监控

扩展 KAOS 协议，添加文件变更监控：

```python
class WatchableKaos(Kaos):
    def watch(self, pattern: str, callback):
        """监控文件变更"""
        pass
```

## 17.11 小结

KAOS 抽象层让你的 Agent：

- ✅ **可移植**：在任何环境运行（本地、云、容器）
- ✅ **可测试**：使用内存文件系统快速测试
- ✅ **安全**：路径验证防止越界访问
- ✅ **灵活**：轻松切换文件系统后端

记住：**永远通过 KAOS 操作文件，永远不要直接使用 `open()` 或 `Path.read_text()`。**

下一章，我们将探讨如何测试使用 KAOS 的 Agent。

---

**上一章**：[第 16 章：会话管理](./16-session-management.md) ←
**下一章**：[第 18 章：测试策略](./18-testing.md) →
