# 第 16 章：会话管理

每次对话都是一个"会话"（Session）。好的会话管理让用户能够：

- 📝 继续上次的对话
- 🔍 查看历史会话
- 🗑️ 清理旧会话
- 💾 恢复崩溃的会话
- 🔄 在多个会话间切换

## 16.1 为什么需要会话管理？

### 场景 1：继续未完成的工作

昨天你让 Agent 重构代码，进行到一半：

```bash
$ kimi
> 你: 重构 user.py，提取数据库逻辑
Agent: 好的，我已经提取了 3 个函数...
> 你: ^C (意外中断)

# 第二天
$ kimi --continue
Agent: 继续昨天的重构任务...
已完成: ✓ 提取数据库逻辑
待完成: ⏳ 更新测试、⏳ 更新文档
```

### 场景 2：多项目切换

你同时参与多个项目：

```bash
$ kimi --session project-a
> 处理 project-a 的任务...

$ kimi --session project-b
> 处理 project-b 的任务...

$ kimi --session project-a
> 继续 project-a，上下文仍然存在
```

### 场景 3：崩溃恢复

Agent 崩溃了，但工作不应该丢失：

```bash
$ kimi
> 你: 分析这 1000 个文件
Agent: 正在分析... (程序崩溃)

# 重启后
$ kimi
检测到未完成的会话，是否恢复？ [y/n]
> y
Agent: 继续分析，已完成 450/1000...
```

## 16.2 会话数据结构

### 会话元数据

```python
# session/metadata.py

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import uuid

@dataclass
class SessionMetadata:
    """会话元数据"""
    # 基本信息
    id: str
    created_at: datetime
    last_active: datetime
    status: str = "active"  # active, paused, completed, crashed

    # 工作环境
    work_dir: Path
    model: str
    total_steps: int = 0

    # 统计信息
    message_count: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # 标签和描述
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None

    # 进度信息
    current_task: Optional[str] = None
    progress: float = 0.0

def create_session_id() -> str:
    """创建唯一的会话 ID"""
    return str(uuid.uuid4())[:8]
```

### 会话存储结构

```
~/.kimi/sessions/
├── abc123/                    # 会话 ID
│   ├── metadata.json         # 元数据
│   ├── messages.jsonl        # 消息历史
│   ├── snapshots/            # 状态快照
│   │   ├── step_1.json
│   │   ├── step_2.json
│   │   └── ...
│   └── artifacts/            # 生成的文件
│       ├── output.txt
│       └── ...
├── def456/
│   └── ...
└── index.json                # 会话索引
```

## 16.3 会话生命周期

### 创建会话

```python
# session/manager.py

import json
from pathlib import Path
from datetime import datetime

class SessionManager:
    """会话管理器"""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = sessions_dir / "index.json"
        self._ensure_index()

    def _ensure_index(self):
        """确保索引文件存在"""
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps({"sessions": []}))

    def create_session(
        self,
        work_dir: Path,
        model: str,
        description: Optional[str] = None,
        tags: List[str] = None
    ) -> SessionMetadata:
        """创建新会话"""
        session = SessionMetadata(
            id=create_session_id(),
            created_at=datetime.now(),
            last_active=datetime.now(),
            work_dir=work_dir,
            model=model,
            description=description,
            tags=tags or []
        )

        # 创建会话目录
        session_dir = self._get_session_dir(session.id)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "snapshots").mkdir(exist_ok=True)
        (session_dir / "artifacts").mkdir(exist_ok=True)

        # 保存元数据
        self._save_metadata(session)

        # 更新索引
        self._add_to_index(session)

        return session

    def _get_session_dir(self, session_id: str) -> Path:
        """获取会话目录"""
        return self.sessions_dir / session_id

    def _save_metadata(self, session: SessionMetadata):
        """保存会话元数据"""
        metadata_file = self._get_session_dir(session.id) / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self._serialize_session(session), f, indent=2)

    def _serialize_session(self, session: SessionMetadata) -> dict:
        """序列化会话对象"""
        return {
            "id": session.id,
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
            "status": session.status,
            "work_dir": str(session.work_dir),
            "model": session.model,
            "total_steps": session.total_steps,
            "message_count": session.message_count,
            "tool_calls": session.tool_calls,
            "total_tokens": session.total_tokens,
            "total_cost": session.total_cost,
            "tags": session.tags,
            "description": session.description,
            "current_task": session.current_task,
            "progress": session.progress
        }

    def _add_to_index(self, session: SessionMetadata):
        """添加到索引"""
        index = self._load_index()
        index["sessions"].append({
            "id": session.id,
            "created_at": session.created_at.isoformat(),
            "description": session.description,
            "tags": session.tags
        })
        self._save_index(index)

    def _load_index(self) -> dict:
        """加载索引"""
        with open(self.index_file) as f:
            return json.load(f)

    def _save_index(self, index: dict):
        """保存索引"""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
```

### 加载会话

```python
class SessionManager:
    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """获取会话"""
        metadata_file = self._get_session_dir(session_id) / "metadata.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            data = json.load(f)
            return self._deserialize_session(data)

    def _deserialize_session(self, data: dict) -> SessionMetadata:
        """反序列化会话对象"""
        return SessionMetadata(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            status=data["status"],
            work_dir=Path(data["work_dir"]),
            model=data["model"],
            total_steps=data["total_steps"],
            message_count=data["message_count"],
            tool_calls=data["tool_calls"],
            total_tokens=data["total_tokens"],
            total_cost=data["total_cost"],
            tags=data.get("tags", []),
            description=data.get("description"),
            current_task=data.get("current_task"),
            progress=data.get("progress", 0.0)
        )

    def list_sessions(
        self,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> List[SessionMetadata]:
        """列出会话"""
        sessions = []

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir() or session_dir.name == "index.json":
                continue

            metadata_file = session_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                data = json.load(f)
                session = self._deserialize_session(data)

                # 过滤
                if tags and not any(t in session.tags for t in tags):
                    continue
                if status and session.status != status:
                    continue

                sessions.append(session)

        # 按最后活跃时间排序
        sessions.sort(key=lambda s: s.last_active, reverse=True)
        return sessions[:limit]

    def get_latest_session(self) -> Optional[SessionMetadata]:
        """获取最近的会话"""
        sessions = self.list_sessions(limit=1)
        return sessions[0] if sessions else None
```

### 更新会话

```python
class SessionManager:
    def update_session(self, session: SessionMetadata):
        """更新会话"""
        session.last_active = datetime.now()
        self._save_metadata(session)

    def mark_completed(self, session_id: str):
        """标记会话完成"""
        session = self.get_session(session_id)
        if session:
            session.status = "completed"
            session.progress = 1.0
            self._save_metadata(session)

    def mark_crashed(self, session_id: str):
        """标记会话崩溃"""
        session = self.get_session(session_id)
        if session:
            session.status = "crashed"
            self._save_metadata(session)
```

## 16.4 消息历史管理

### 存储消息

```python
# session/messages.py

import json
from typing import List, Dict, Any

class MessageStore:
    """消息存储"""

    def __init__(self, session_dir: Path):
        self.messages_file = session_dir / "messages.jsonl"

    def append_message(self, role: str, content: str, metadata: dict = None):
        """追加消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        with open(self.messages_file, 'a') as f:
            f.write(json.dumps(message) + '\n')

    def get_messages(self, limit: int = None) -> List[Dict[str, Any]]:
        """获取消息历史"""
        if not self.messages_file.exists():
            return []

        messages = []
        with open(self.messages_file) as f:
            for line in f:
                messages.append(json.loads(line))

        if limit:
            return messages[-limit:]
        return messages

    def clear_messages(self):
        """清空消息"""
        if self.messages_file.exists():
            self.messages_file.unlink()
```

## 16.5 状态快照

### 创建快照

```python
# session/snapshot.py

class SnapshotManager:
    """快照管理器"""

    def __init__(self, session_dir: Path):
        self.snapshots_dir = session_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

    def create_snapshot(self, step: int, state: dict):
        """创建状态快照"""
        snapshot_file = self.snapshots_dir / f"step_{step}.json"
        with open(snapshot_file, 'w') as f:
            json.dump({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "state": state
            }, f, indent=2)

    def get_snapshot(self, step: int) -> Optional[dict]:
        """获取快照"""
        snapshot_file = self.snapshots_dir / f"step_{step}.json"
        if not snapshot_file.exists():
            return None

        with open(snapshot_file) as f:
            return json.load(f)

    def get_latest_snapshot(self) -> Optional[dict]:
        """获取最新快照"""
        snapshots = sorted(self.snapshots_dir.glob("step_*.json"))
        if not snapshots:
            return None

        with open(snapshots[-1]) as f:
            return json.load(f)

    def list_snapshots(self) -> List[int]:
        """列出所有快照"""
        snapshots = []
        for file in self.snapshots_dir.glob("step_*.json"):
            step = int(file.stem.split('_')[1])
            snapshots.append(step)
        return sorted(snapshots)
```

## 16.6 会话恢复

### 检测未完成会话

```python
# session/recovery.py

class SessionRecovery:
    """会话恢复"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def find_recoverable_sessions(self) -> List[SessionMetadata]:
        """查找可恢复的会话"""
        return self.session_manager.list_sessions(status="active")

    def recover_session(self, session_id: str) -> bool:
        """恢复会话"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return False

        # 加载最新快照
        snapshot_mgr = SnapshotManager(
            self.session_manager._get_session_dir(session_id)
        )
        latest_snapshot = snapshot_mgr.get_latest_snapshot()

        if not latest_snapshot:
            return False

        # 恢复状态
        print(f"恢复会话 {session_id}")
        print(f"上次进度: {session.progress * 100:.1f}%")
        print(f"当前任务: {session.current_task}")

        return True

    def auto_recover(self) -> Optional[SessionMetadata]:
        """自动恢复最近的会话"""
        sessions = self.find_recoverable_sessions()
        if not sessions:
            return None

        latest = sessions[0]
        print(f"检测到未完成的会话: {latest.id}")
        print(f"创建时间: {latest.created_at}")
        print(f"描述: {latest.description or '无'}")

        response = input("是否恢复此会话？ [y/n] ")
        if response.lower() == 'y':
            if self.recover_session(latest.id):
                return latest

        return None
```

## 16.7 并发会话管理

### 会话锁

```python
# session/lock.py

import fcntl
from contextlib import contextmanager

class SessionLock:
    """会话锁，防止并发访问"""

    def __init__(self, session_dir: Path):
        self.lock_file = session_dir / ".lock"

    @contextmanager
    def acquire(self, timeout: int = 5):
        """获取锁"""
        lock_fd = None
        try:
            # 创建锁文件
            lock_fd = open(self.lock_file, 'w')

            # 尝试获取锁
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"无法获取会话锁，超时 {timeout}s")
                    time.sleep(0.1)

            yield

        finally:
            if lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
```

## 16.8 会话清理

### 自动清理策略

```python
# session/cleanup.py

class SessionCleanup:
    """会话清理"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def cleanup_old_sessions(self, days: int = 30):
        """清理旧会话"""
        cutoff = datetime.now() - timedelta(days=days)
        sessions = self.session_manager.list_sessions(limit=1000)

        removed = 0
        for session in sessions:
            if session.last_active < cutoff and session.status == "completed":
                self._remove_session(session.id)
                removed += 1

        print(f"清理了 {removed} 个旧会话")

    def cleanup_crashed_sessions(self):
        """清理崩溃的会话"""
        sessions = self.session_manager.list_sessions(status="crashed")

        for session in sessions:
            print(f"发现崩溃的会话: {session.id}")
            print(f"描述: {session.description}")
            response = input("删除此会话？ [y/n] ")
            if response.lower() == 'y':
                self._remove_session(session.id)

    def _remove_session(self, session_id: str):
        """删除会话"""
        import shutil
        session_dir = self.session_manager._get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

        # 从索引中移除
        index = self.session_manager._load_index()
        index["sessions"] = [
            s for s in index["sessions"] if s["id"] != session_id
        ]
        self.session_manager._save_index(index)
```

## 16.9 会话分析

### 统计信息

```python
# session/analytics.py

class SessionAnalytics:
    """会话分析"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def get_statistics(self) -> dict:
        """获取统计信息"""
        sessions = self.session_manager.list_sessions(limit=1000)

        total_sessions = len(sessions)
        total_messages = sum(s.message_count for s in sessions)
        total_tokens = sum(s.total_tokens for s in sessions)
        total_cost = sum(s.total_cost for s in sessions)

        active_sessions = [s for s in sessions if s.status == "active"]
        completed_sessions = [s for s in sessions if s.status == "completed"]

        return {
            "total_sessions": total_sessions,
            "active_sessions": len(active_sessions),
            "completed_sessions": len(completed_sessions),
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_session": total_cost / total_sessions if total_sessions > 0 else 0
        }

    def generate_report(self) -> str:
        """生成报告"""
        stats = self.get_statistics()

        report = []
        report.append("# 会话统计报告\n")
        report.append(f"总会话数: {stats['total_sessions']}")
        report.append(f"活跃会话: {stats['active_sessions']}")
        report.append(f"已完成会话: {stats['completed_sessions']}")
        report.append(f"\n总消息数: {stats['total_messages']}")
        report.append(f"总 Token 数: {stats['total_tokens']:,}")
        report.append(f"总成本: ${stats['total_cost']:.2f}")
        report.append(f"平均每会话成本: ${stats['avg_cost_per_session']:.2f}")

        return "\n".join(report)
```

## 16.10 常见陷阱

### 陷阱 1：忘记保存状态

```python
# ❌ 错误：不保存中间状态
def run_task(session):
    for i in range(100):
        do_work(i)
    # 如果崩溃，所有工作都丢失！

# ✅ 正确：定期保存快照
def run_task(session, snapshot_mgr):
    for i in range(100):
        do_work(i)
        if i % 10 == 0:
            snapshot_mgr.create_snapshot(i, {"progress": i})
```

### 陷阱 2：会话泄漏

```python
# ❌ 错误：创建会话但从不清理
session = session_manager.create_session(...)
# ... 使用会话 ...
# 忘记标记完成！

# ✅ 正确：使用上下文管理器
@contextmanager
def managed_session(session_manager, ...):
    session = session_manager.create_session(...)
    try:
        yield session
    finally:
        session_manager.mark_completed(session.id)
```

### 陷阱 3：并发冲突

```python
# ❌ 错误：多个进程同时修改会话
# 进程 1
session = session_manager.get_session(id)
session.message_count += 1  # 竞争条件！

# ✅ 正确：使用锁
lock = SessionLock(session_dir)
with lock.acquire():
    session = session_manager.get_session(id)
    session.message_count += 1
    session_manager.update_session(session)
```

## 16.11 最佳实践

### 1. 会话命名

```python
# 创建会话时提供有意义的描述
session = session_manager.create_session(
    work_dir=Path.cwd(),
    model="gpt-4",
    description="重构用户认证模块",
    tags=["refactoring", "auth"]
)
```

### 2. 定期快照

```python
# 每 N 步保存一次快照
class Agent:
    def run(self):
        for step in range(max_steps):
            result = self.execute_step(step)

            # 每 10 步快照一次
            if step % 10 == 0:
                self.snapshot_manager.create_snapshot(step, {
                    "step": step,
                    "context": self.context,
                    "history": self.history
                })
```

### 3. 优雅关闭

```python
# 注册信号处理器
import signal

def signal_handler(sig, frame):
    print("\n正在保存会话...")
    session_manager.update_session(current_session)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
```

## 16.12 FAQ

**Q: 会话占用多少磁盘空间？**

A: 取决于消息数量和快照频率。典型的会话：
- 元数据: ~1 KB
- 消息历史 (100条): ~50 KB
- 快照 (10个): ~100 KB
- 总计: ~150 KB

**Q: 如何在多台机器间同步会话？**

A: 可以将 sessions 目录放在云存储（如 Dropbox）：
```bash
ln -s ~/Dropbox/kimi-sessions ~/.kimi/sessions
```

**Q: 会话能保存多久？**

A: 默认永久保存，但可以配置自动清理：
```python
cleanup = SessionCleanup(session_manager)
cleanup.cleanup_old_sessions(days=30)  # 清理 30 天前的
```

## 16.13 练习

### 练习 1: 实现会话导出

实现导出会话到 zip 文件：

```python
class SessionExporter:
    def export_session(self, session_id: str, output_path: Path):
        """导出会话到 zip"""
        # TODO: 打包会话目录
        pass

    def import_session(self, zip_path: Path) -> str:
        """从 zip 导入会话"""
        # TODO: 解压并恢复会话
        pass
```

### 练习 2: 实现会话搜索

实现按内容搜索会话：

```python
class SessionSearch:
    def search_messages(self, query: str) -> List[SessionMetadata]:
        """在所有会话的消息中搜索"""
        # TODO: 实现全文搜索
        pass
```

### 练习 3: 实现会话合并

合并多个会话：

```python
class SessionMerger:
    def merge_sessions(
        self,
        session_ids: List[str],
        output_description: str
    ) -> SessionMetadata:
        """合并多个会话"""
        # TODO: 合并消息历史和元数据
        pass
```

## 16.14 小结

本章学习了：

- ✅ **会话生命周期**：创建、加载、更新、删除
- ✅ **消息历史**：JSONL 格式存储
- ✅ **状态快照**：支持恢复和回滚
- ✅ **会话恢复**：崩溃后自动恢复
- ✅ **并发控制**：文件锁防止冲突
- ✅ **会话清理**：自动清理旧会话
- ✅ **会话分析**：统计和报告

**关键要点**:

1. 每个会话有唯一 ID 和完整元数据
2. 定期保存快照，支持恢复
3. 使用锁机制处理并发
4. 及时清理完成的会话
5. 提供丰富的查询和分析功能

会话管理提供：

- 💾 **持久化**：工作永不丢失
- 🔄 **可恢复**：崩溃后继续
- 🔍 **可追溯**：完整历史记录
- 📊 **可分析**：统计和报告

---

**上一章**：[第 15 章：配置系统](./15-config-system.md) ←
**下一章**：[第 17 章：KAOS 抽象](./17-kaos-abstraction.md) →
