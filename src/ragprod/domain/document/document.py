from dataclasses import dataclass, field
from typing import Dict, Any, Union
from rich.console import Console
from rich.panel import Panel
from .base import BaseDocument

@dataclass
class Document(BaseDocument):
    id: str = None
    raw_text: str = ""
    source: str = "Unknown"
    title: str = "Untitled"
    _metadata: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    distance: Union[float, None] = None
    score: Union[float, None] = None

    def __init__(
        self,
        id: str = None,
        raw_text: str = "",
        source: str = "Unknown",
        title: str = "Untitled",
        metadata: Dict[str, Any] = None,
        distance: Union[float, None] = None,
        score: Union[float, None] = None,
    ):
        """Initialize Document with metadata parameter."""
        self.id = id
        self.raw_text = raw_text
        self.source = source
        self.title = title
        self.distance = distance
        self.score = score
        self._metadata = metadata.copy() if metadata else {}

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def content(self) -> str:
        return self.raw_text

    def __repr__(self) -> str:
        console = Console()

        # Ensure metadata is a dict
        meta = self.metadata or {}

        # Merge source, title, and metadata
        meta_dict = {"source": self.source, "title": self.title, **meta}

        # Prepare metadata text
        meta_text = "\n".join(f"[bold yellow]{k}[/]: [magenta]{v}[/]" for k, v in meta_dict.items())

        # Prepare content preview
        preview = self.content[:200] + ("..." if len(self.content) > 200 else "")

        # Combine content and metadata
        panel_text = f"[bold green]Content:[/]\n{preview}\n\n[bold green]Metadata:[/]\n{meta_text}"

        panel = Panel(panel_text, border_style="green", expand=False)

        console.print(panel)

        return ""  # Avoid default BaseModel repr

    def __str__(self) -> str:
        """Implement abstract method from BaseDocument."""
        return self.content  # Or any string representation you want
