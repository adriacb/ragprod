from typing import List, Optional, Callable, Union
from .base import BaseTextSplitter


class TokenTextSplitter(BaseTextSplitter):
    """Memory-efficient token-based text splitter with custom tokenizer support."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[Callable[[str], List[Union[int, str]]]] = None,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if tokenizer is None:
            try:
                import tiktoken
                self.encoding = tiktoken.get_encoding(encoding_name)
                self.tokenizer = lambda t: self.encoding.encode(t)
                self.decode = self.encoding.decode
            except ImportError:
                self.encoding = None
                self.tokenizer = lambda t: list(t)
                self.decode = lambda toks: "".join(toks)
        else:
            self.encoding = None
            self.tokenizer = tokenizer

            # custom decode: join strings
            def _decode(toks):
                if all(isinstance(t, str) for t in toks):
                    return " ".join(toks)
                return "".join(str(t) for t in toks)
            self.decode = _decode

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        tokens = self.tokenizer(text)
        total = len(tokens)
        if total <= self.chunk_size:
            return [text]

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks = []
        start = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            chunk_tokens = tokens[start:end]
            chunks.append(self.decode(chunk_tokens))

            start += self.chunk_size - self.chunk_overlap

        return chunks