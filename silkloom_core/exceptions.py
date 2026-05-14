from __future__ import annotations


class SilkLoomError(Exception):
    """Base exception type for SilkLoom Core."""


class ConfigurationError(SilkLoomError, ValueError):
    """Raised when runtime configuration values are invalid.
    
    解决方案：检查 max_retries >= 1, workers >= 1, max_concurrent >= 1 的约束。
    """


class InvalidInputError(SilkLoomError, TypeError):
    """Raised when provided input payload has an unsupported shape.
    
    解决方案：
    - map() 期望序列(列表/DataFrame)，不在单个字符串上调用 map()，用 run_one() 代替。
    - 输入项必须是字符串或映射(dict)。检查 .required_variables 了解模板需要的字段。
    """


class AsyncClientNotConfiguredError(SilkLoomError, RuntimeError):
    """Raised when async API is used without a configured async client.
    
    解决方案：
    - 调用 amap() / arun_one() 前，确保已安装 openai>=1.40 并在构建 PromptMapper 时传递 async_client。
    - 或设置环境变量 OPENAI_API_KEY 并让 PromptMapper 自动初始化 AsyncOpenAI。
    """


class TemplateRenderError(SilkLoomError):
    """Raised when Jinja2 template rendering fails.
    
    解决方案：
    - 检查输入数据是否包含模板中所有必需的变量。
    - 使用 mapper.required_variables 查看模板需要的字段。
    - 如启用 simple_format_mode=True，用 {field} 替代 {{ field }}。
    """


class ResponseParseError(SilkLoomError):
    """Raised when model output cannot be parsed into response_model.
    
    解决方案：
    - 检查 response_model 的字段定义与模型输出格式是否匹配。
    - 在 system_prompt 中明确指示模型输出 JSON 或特定结构。
    - 查看 raw_output 字段/reasoning 关键字了解实际模型回答内容。
    """


class LLMRequestError(SilkLoomError):
    """Raised for LLM request failures not covered by other categories.
    
    解决方案：
    - 检查 API 密钥 (OPENAI_API_KEY) 和网络连接。
    - 查看 TaskResult.raw_output 了解 API 返回的具体错误。
    - 检查模型名称 (gpt-4, gpt-3.5-turbo 等) 是否正确；max_retries >= 3 以重试。
    """
