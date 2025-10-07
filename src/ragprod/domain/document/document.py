from pydantic import BaseModel
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


from .base import BaseDocument

class Document(BaseModel, BaseDocument):
    id: str = None
    raw_text: str
    source: str = "Unknown"
    title: str = "Untitled"
    metadata: Dict[str, Any] = {}

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


