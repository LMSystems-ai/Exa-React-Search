import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Type, Dict, Union, Literal

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""
    # Optional configuration
    filter_domains: list[str] = field(default_factory=lambda: ["langchain.com", "langchain-ai.github.io"])

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        if not config or "configurable" not in config:
            raise ValueError("Configuration required")

        configurable = config["configurable"]
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

