# 第 15 章：配置系统

不同用户有不同需求：

- 🌍 国内用户：用 Moonshot Kimi
- 🌐 国际用户：用 OpenAI GPT-4
- 💰 成本敏感：用便宜的模型
- 🚀 性能优先：用最强的模型

**配置系统**让 Agent 灵活适应各种环境。

## 15.1 为什么需要配置系统？

### 场景 1：团队协作

你的团队有不同的工作环境：

```
开发环境 (dev)
├─ 使用本地 LLM (Ollama)
├─ 低成本模型
└─ 详细日志

生产环境 (prod)
├─ 使用 GPT-4
├─ 高性能模型
└─ 简洁日志
```

### 场景 2：多账号管理

你有多个 API 账号，需要分别配置：

```bash
# 个人账号
OPENAI_API_KEY=sk-personal-xxx

# 公司账号
OPENAI_API_KEY=sk-company-xxx

# 测试账号
OPENAI_API_KEY=sk-test-xxx
```

### 场景 3：敏感信息保护

配置中包含敏感信息，不能提交到 Git：

```json
{
  "api_key": "sk-xxx",  // ❌ 不能提交！
  "database_password": "xxx"  // ❌ 不能提交！
}
```

## 15.2 配置层级

kimi-cli 使用三层配置，优先级从低到高：

```
系统配置 (System)
  ↓ 被覆盖
用户配置 (User)
  ↓ 被覆盖
项目配置 (Project)
  ↓ 被覆盖
环境变量 (Environment)
```

### 层级 1：系统配置

所有用户共享的默认配置：

```bash
# Linux/macOS
/etc/kimi/config.json

# Windows
C:\ProgramData\kimi\config.json
```

```json
{
  "llm_providers": {
    "openai": {
      "base_url": "https://api.openai.com/v1"
    }
  },
  "max_steps": 100,
  "timeout": 300
}
```

### 层级 2：用户配置

个人配置，存储在用户主目录：

```bash
# Linux/macOS
~/.kimi/config.json

# Windows
%USERPROFILE%\.kimi\config.json
```

```json
{
  "default_model": "gpt-4",
  "llm_providers": {
    "openai": {
      "api_key_env": "OPENAI_API_KEY"
    }
  },
  "ui_mode": "shell"
}
```

### 层级 3：项目配置

项目特定配置，存储在项目根目录：

```bash
# 项目根目录
./kimi.json
```

```json
{
  "default_model": "gpt-3.5-turbo",  // 项目用便宜的模型
  "max_steps": 50,  // 限制步数
  "work_dir": "./workspace",  // 工作目录
  "prompt_file": "./custom_prompt.md"  // 自定义提示词
}
```

### 层级 4：环境变量

最高优先级，运行时覆盖：

```bash
# 临时切换模型
KIMI_MODEL=gpt-4-turbo kimi

# 临时切换 API Key
OPENAI_API_KEY=sk-xxx kimi

# 临时开启调试
KIMI_DEBUG=1 kimi
```

## 15.3 完整配置实现

### 配置数据结构

```python
# config/schema.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

@dataclass
class LLMProvider:
    """LLM 提供商配置"""
    base_url: str
    api_key_env: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3

@dataclass
class LLMModel:
    """LLM 模型配置"""
    provider: str
    name: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    temperature: float = 0.7
    supports_streaming: bool = True
    supports_vision: bool = False

@dataclass
class Config:
    """完整配置"""
    # LLM 配置
    llm_providers: Dict[str, LLMProvider] = field(default_factory=dict)
    llm_models: Dict[str, LLMModel] = field(default_factory=dict)
    default_model: str = "gpt-4"

    # Agent 配置
    max_steps: int = 100
    timeout: int = 300
    approval_required: bool = True

    # UI 配置
    ui_mode: str = "shell"
    ui_format: str = "text"
    verbose: bool = False

    # 路径配置
    work_dir: Optional[Path] = None
    prompt_file: Optional[Path] = None
    sessions_dir: Optional[Path] = None

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[Path] = None
```

### 配置加载器

```python
# config/loader.py

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """配置加载器"""

    def __init__(self):
        self.system_config_path = self._get_system_config_path()
        self.user_config_path = self._get_user_config_path()

    def _get_system_config_path(self) -> Path:
        """获取系统配置路径"""
        if os.name == 'nt':  # Windows
            return Path(os.getenv('PROGRAMDATA', 'C:/ProgramData')) / 'kimi' / 'config.json'
        else:  # Linux/macOS
            return Path('/etc/kimi/config.json')

    def _get_user_config_path(self) -> Path:
        """获取用户配置路径"""
        return Path.home() / '.kimi' / 'config.json'

    def _get_project_config_path(self, work_dir: Path) -> Optional[Path]:
        """获取项目配置路径"""
        # 在当前目录及父目录查找
        current = work_dir.absolute()
        while current != current.parent:
            config_file = current / 'kimi.json'
            if config_file.exists():
                return config_file
            current = current.parent
        return None

    def load(self, work_dir: Optional[Path] = None) -> Config:
        """加载完整配置（合并所有层级）"""
        work_dir = work_dir or Path.cwd()

        # 1. 加载系统配置
        system_config = self._load_file(self.system_config_path)

        # 2. 加载用户配置
        user_config = self._load_file(self.user_config_path)

        # 3. 加载项目配置
        project_config_path = self._get_project_config_path(work_dir)
        project_config = self._load_file(project_config_path) if project_config_path else {}

        # 4. 合并配置（优先级：项目 > 用户 > 系统）
        merged = self._merge_configs(
            system_config,
            user_config,
            project_config
        )

        # 5. 应用环境变量
        merged = self._apply_env_vars(merged)

        # 6. 验证配置
        self._validate(merged)

        # 7. 转换为 Config 对象
        return self._to_config_object(merged)

    def _load_file(self, path: Optional[Path]) -> Dict[str, Any]:
        """加载配置文件"""
        if not path or not path.exists():
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {path}\n{e}")

    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并多个配置"""
        result = {}
        for config in configs:
            self._deep_merge(result, config)
        return result

    def _deep_merge(self, base: Dict, overlay: Dict):
        """深度合并两个字典"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        # KIMI_MODEL -> default_model
        if os.getenv('KIMI_MODEL'):
            config['default_model'] = os.getenv('KIMI_MODEL')

        # KIMI_DEBUG -> log_level
        if os.getenv('KIMI_DEBUG'):
            config['log_level'] = 'DEBUG'

        # KIMI_MODE -> ui_mode
        if os.getenv('KIMI_MODE'):
            config['ui_mode'] = os.getenv('KIMI_MODE')

        # KIMI_MAX_STEPS -> max_steps
        if os.getenv('KIMI_MAX_STEPS'):
            config['max_steps'] = int(os.getenv('KIMI_MAX_STEPS'))

        return config

    def _validate(self, config: Dict[str, Any]):
        """验证配置"""
        # 检查必需字段
        if 'llm_providers' not in config or not config['llm_providers']:
            raise ValueError("配置中必须包含至少一个 LLM 提供商")

        # 检查默认模型存在
        default_model = config.get('default_model')
        if default_model and default_model not in config.get('llm_models', {}):
            raise ValueError(f"默认模型 '{default_model}' 不存在于配置中")

        # 检查数值范围
        if config.get('max_steps', 0) <= 0:
            raise ValueError("max_steps 必须大于 0")

    def _to_config_object(self, data: Dict[str, Any]) -> Config:
        """转换为 Config 对象"""
        # 转换 providers
        providers = {
            name: LLMProvider(**provider_data)
            for name, provider_data in data.get('llm_providers', {}).items()
        }

        # 转换 models
        models = {
            name: LLMModel(**model_data)
            for name, model_data in data.get('llm_models', {}).items()
        }

        # 转换路径
        work_dir = Path(data['work_dir']) if data.get('work_dir') else None
        prompt_file = Path(data['prompt_file']) if data.get('prompt_file') else None
        sessions_dir = Path(data['sessions_dir']) if data.get('sessions_dir') else None
        log_file = Path(data['log_file']) if data.get('log_file') else None

        return Config(
            llm_providers=providers,
            llm_models=models,
            default_model=data.get('default_model', 'gpt-4'),
            max_steps=data.get('max_steps', 100),
            timeout=data.get('timeout', 300),
            approval_required=data.get('approval_required', True),
            ui_mode=data.get('ui_mode', 'shell'),
            ui_format=data.get('ui_format', 'text'),
            verbose=data.get('verbose', False),
            work_dir=work_dir,
            prompt_file=prompt_file,
            sessions_dir=sessions_dir,
            log_level=data.get('log_level', 'INFO'),
            log_file=log_file
        )
```

## 15.4 秘密管理

### 方案 1：环境变量

```bash
# .env 文件（不提交到 Git）
OPENAI_API_KEY=sk-xxx
MOONSHOT_API_KEY=sk-yyy
DATABASE_PASSWORD=zzz
```

```python
# 加载 .env
from dotenv import load_dotenv
load_dotenv()

# 从环境变量读取
api_key = os.getenv('OPENAI_API_KEY')
```

### 方案 2：密钥管理服务

```python
# config/secrets.py

class SecretManager:
    """密钥管理器"""

    def __init__(self, backend: str = "env"):
        self.backend = backend

    def get_secret(self, key: str) -> str:
        """获取密钥"""
        if self.backend == "env":
            return self._get_from_env(key)
        elif self.backend == "keyring":
            return self._get_from_keyring(key)
        elif self.backend == "vault":
            return self._get_from_vault(key)

    def _get_from_env(self, key: str) -> str:
        """从环境变量获取"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"环境变量 {key} 未设置")
        return value

    def _get_from_keyring(self, key: str) -> str:
        """从系统密钥链获取"""
        import keyring
        value = keyring.get_password("kimi", key)
        if not value:
            raise ValueError(f"密钥 {key} 不存在")
        return value

    def _get_from_vault(self, key: str) -> str:
        """从 Vault 获取"""
        import hvac
        client = hvac.Client(url=os.getenv('VAULT_URL'))
        client.token = os.getenv('VAULT_TOKEN')
        secret = client.secrets.kv.v2.read_secret_version(path=key)
        return secret['data']['data']['value']
```

### 方案 3：加密配置

```python
# 加密敏感配置
from cryptography.fernet import Fernet

class EncryptedConfig:
    """加密配置"""

    def __init__(self, key_file: Path):
        # 从文件读取加密密钥
        with open(key_file, 'rb') as f:
            self.key = f.read()
        self.fernet = Fernet(self.key)

    def encrypt_value(self, value: str) -> str:
        """加密值"""
        return self.fernet.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted: str) -> str:
        """解密值"""
        return self.fernet.decrypt(encrypted.encode()).decode()

    def save_encrypted(self, config: dict, path: Path):
        """保存加密配置"""
        encrypted = {}
        for key, value in config.items():
            if key.endswith('_key') or key.endswith('_password'):
                encrypted[key] = self.encrypt_value(value)
            else:
                encrypted[key] = value

        with open(path, 'w') as f:
            json.dump(encrypted, f)
```

## 15.5 多环境支持

### 环境配置文件

```bash
.kimi/
├── config.json          # 基础配置
├── config.dev.json      # 开发环境
├── config.staging.json  # 预发环境
└── config.prod.json     # 生产环境
```

```python
# config/environment.py

class EnvironmentConfig:
    """环境配置管理"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def load_for_env(self, env: str = None) -> Config:
        """加载指定环境的配置"""
        # 1. 确定环境
        env = env or os.getenv('KIMI_ENV', 'dev')

        # 2. 加载基础配置
        base_config = self._load_file(self.base_dir / 'config.json')

        # 3. 加载环境配置
        env_config = self._load_file(self.base_dir / f'config.{env}.json')

        # 4. 合并
        merged = {**base_config, **env_config}

        return Config(**merged)
```

### 环境配置示例

```json
// config.dev.json
{
  "default_model": "gpt-3.5-turbo",  // 开发用便宜模型
  "log_level": "DEBUG",
  "approval_required": false,  // 开发不需要审批
  "llm_providers": {
    "openai": {
      "base_url": "http://localhost:8000/v1"  // 本地代理
    }
  }
}

// config.prod.json
{
  "default_model": "gpt-4-turbo",  // 生产用最强模型
  "log_level": "INFO",
  "approval_required": true,  // 生产需要审批
  "max_steps": 50,  // 限制步数
  "timeout": 600
}
```

## 15.6 配置热更新

```python
# config/watcher.py

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigWatcher(FileSystemEventHandler):
    """配置文件监听器"""

    def __init__(self, config_file: Path, on_reload):
        self.config_file = config_file
        self.on_reload = on_reload
        self.last_modified = 0

    def on_modified(self, event):
        """文件修改回调"""
        if event.src_path != str(self.config_file):
            return

        # 防止重复触发
        current_time = time.time()
        if current_time - self.last_modified < 1:
            return
        self.last_modified = current_time

        # 重新加载配置
        try:
            new_config = ConfigLoader().load()
            self.on_reload(new_config)
            print("✅ 配置已重新加载")
        except Exception as e:
            print(f"❌ 配置重新加载失败: {e}")

def watch_config(config_file: Path, on_reload):
    """监听配置文件变化"""
    observer = Observer()
    handler = ConfigWatcher(config_file, on_reload)
    observer.schedule(handler, str(config_file.parent), recursive=False)
    observer.start()
    return observer

# 使用示例
def reload_agent_config(new_config: Config):
    """重新加载 Agent 配置"""
    agent.update_config(new_config)

observer = watch_config(
    Path.home() / '.kimi' / 'config.json',
    reload_agent_config
)
```

## 15.7 配置迁移

### 版本检测

```python
# config/migration.py

class ConfigMigrator:
    """配置迁移器"""

    def migrate(self, config_data: dict) -> dict:
        """迁移配置到最新版本"""
        version = config_data.get('version', 1)

        # 应用所有必要的迁移
        if version < 2:
            config_data = self._migrate_v1_to_v2(config_data)
        if version < 3:
            config_data = self._migrate_v2_to_v3(config_data)

        config_data['version'] = 3
        return config_data

    def _migrate_v1_to_v2(self, config: dict) -> dict:
        """v1 -> v2: 添加 provider 支持"""
        # 将旧的 api_key 移到 provider 配置
        if 'api_key' in config:
            config['llm_providers'] = {
                'openai': {
                    'api_key_env': 'OPENAI_API_KEY'
                }
            }
            del config['api_key']
        return config

    def _migrate_v2_to_v3(self, config: dict) -> dict:
        """v2 -> v3: 添加成本追踪"""
        # 为所有模型添加成本信息
        for model_name, model_config in config.get('llm_models', {}).items():
            if 'cost_per_1k_input' not in model_config:
                model_config['cost_per_1k_input'] = 0.01
                model_config['cost_per_1k_output'] = 0.03
        return config
```

## 15.8 常见陷阱

### 陷阱 1：硬编码配置

```python
# ❌ 错误：硬编码
api_key = "sk-xxx"  # 泄露风险！
base_url = "https://api.openai.com/v1"  # 不灵活

# ✅ 正确：从配置读取
config = ConfigLoader().load()
provider = config.get_provider('openai')
api_key = provider.api_key
```

### 陷阱 2：忽略优先级

```python
# ❌ 错误：只读取一个配置文件
config = json.load(open('config.json'))

# ✅ 正确：合并所有层级
config = ConfigLoader().load()  # 自动合并所有层级
```

### 陷阱 3：明文存储密钥

```python
# ❌ 错误：配置文件中明文存储
{
  "api_key": "sk-xxx"  // 提交到 Git！
}

# ✅ 正确：使用环境变量引用
{
  "api_key_env": "OPENAI_API_KEY"  // 引用环境变量
}
```

## 15.9 最佳实践

### 1. 配置模板

提供配置模板供用户复制：

```bash
.kimi/
├── config.template.json  # 模板
└── .gitignore            # 忽略 config.json
```

```json
// config.template.json
{
  "llm_providers": {
    "openai": {
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY"
    }
  },
  "default_model": "gpt-4",
  "max_steps": 100
}
```

### 2. 配置验证

启动时验证配置：

```python
def validate_config(config: Config):
    """验证配置"""
    errors = []

    # 检查 API Key
    for provider_name, provider in config.llm_providers.items():
        if provider.api_key_env:
            if not os.getenv(provider.api_key_env):
                errors.append(f"环境变量 {provider.api_key_env} 未设置")

    # 检查路径
    if config.work_dir and not config.work_dir.exists():
        errors.append(f"工作目录不存在: {config.work_dir}")

    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(f"- {e}" for e in errors))
```

### 3. 配置文档

生成配置文档：

```python
def generate_config_docs(config: Config) -> str:
    """生成配置文档"""
    docs = []
    docs.append("# 当前配置\n")
    docs.append(f"默认模型: {config.default_model}")
    docs.append(f"最大步数: {config.max_steps}")
    docs.append(f"UI 模式: {config.ui_mode}")

    docs.append("\n## LLM 提供商\n")
    for name, provider in config.llm_providers.items():
        docs.append(f"- {name}: {provider.base_url}")

    return "\n".join(docs)
```

## 15.10 FAQ

**Q: 如何切换不同的 API Key？**

A: 使用环境变量覆盖：
```bash
OPENAI_API_KEY=sk-personal kimi  # 使用个人 Key
OPENAI_API_KEY=sk-company kimi   # 使用公司 Key
```

**Q: 配置文件放在哪里最好？**

A: 推荐顺序：
1. 项目特定配置 → `./kimi.json`
2. 用户配置 → `~/.kimi/config.json`
3. 系统配置 → `/etc/kimi/config.json`

**Q: 如何在 CI/CD 中使用配置？**

A: 使用环境变量：
```yaml
# .github/workflows/test.yml
env:
  KIMI_MODEL: gpt-3.5-turbo
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  KIMI_MAX_STEPS: 20
```

## 15.11 练习

### 练习 1: 实现配置命令

添加 CLI 命令查看和修改配置：

```python
@click.group()
def config():
    """配置管理"""
    pass

@config.command()
def show():
    """显示当前配置"""
    # TODO: 实现显示配置
    pass

@config.command()
@click.argument('key')
@click.argument('value')
def set(key, value):
    """设置配置项"""
    # TODO: 实现设置配置
    pass
```

### 练习 2: 实现配置校验

添加配置 schema 校验：

```python
from jsonschema import validate

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "default_model": {"type": "string"},
        "max_steps": {"type": "number", "minimum": 1}
    },
    "required": ["default_model"]
}

def validate_config(config_data: dict):
    """校验配置格式"""
    # TODO: 使用 jsonschema 校验
    pass
```

### 练习 3: 实现配置导入导出

支持导入导出配置：

```python
class ConfigManager:
    def export_config(self, output_path: Path):
        """导出配置到文件"""
        # TODO: 导出配置（排除敏感信息）
        pass

    def import_config(self, input_path: Path):
        """从文件导入配置"""
        # TODO: 导入并验证配置
        pass
```

## 15.12 小结

本章学习了：

- ✅ **配置层级**：系统、用户、项目、环境变量
- ✅ **秘密管理**：环境变量、密钥服务、加密存储
- ✅ **多环境支持**：dev、staging、prod
- ✅ **配置热更新**：监听文件变化
- ✅ **配置迁移**：版本管理和升级

**关键要点**:

1. 使用分层配置，提供灵活性
2. 永远不要硬编码敏感信息
3. 支持环境变量覆盖
4. 提供配置验证和默认值
5. 文档化所有配置选项

配置系统提供：

- 🔧 **灵活性**：适应不同环境
- 🔒 **安全性**：保护敏感信息
- 🎯 **便捷性**：多层级覆盖
- 📝 **可维护性**：清晰的配置结构

---

**上一章**：[第 14 章：UI 模式](./14-ui-modes.md) ←
**下一章**：[第 16 章：会话管理](./16-session-management.md) →
