from typing import List, Optional, Callable
from .base import BaseTextSplitter


class TokenTextSplitter(BaseTextSplitter):
    """
    Token-based text splitter.
    
    Splits text into chunks based on token count, useful for LLM context limits.
    Requires a tokenizer function to count tokens.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[Callable[[str], int]] = None,
        encoding_name: str = "cl100k_base",  # tiktoken default
    ):
        """
        Initialize the token text splitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
            tokenizer: Optional function to count tokens. If None, uses tiktoken.
            encoding_name: Encoding name for tiktoken (default: cl100k_base for GPT models).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        if tokenizer is None:
            try:
                import tiktoken

                encoding = tiktoken.get_encoding(encoding_name)
                self.tokenizer = lambda text: len(encoding.encode(text))
            except ImportError:
                # Fallback to character-based if tiktoken not available
                self.tokenizer = lambda text: len(text)
        else:
            self.tokenizer = tokenizer

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: The text to split.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # Count tokens in entire text
        total_tokens = self.tokenizer(text)

        if total_tokens <= self.chunk_size:
            return [text]

        # Use recursive character splitter as base, then merge based on tokens
        from .recursive_character import RecursiveCharacterTextSplitter

        # First split by sentences/paragraphs
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # Large size to get natural splits
            chunk_overlap=0,
        )
        splits = base_splitter.split_text(text)

        # Check if recursive splitter removed spaces (common issue)
        # If splits don't have spaces but original text does, use original for token splitting
        original_has_spaces = " " in text
        splits_have_spaces = any(" " in s for s in splits) if splits else False
        
        # If recursive splitter removed spaces, split original text directly by tokens
        if original_has_spaces and not splits_have_spaces:
            return self._split_by_tokens_directly(text)
        
        # If recursive splitter didn't split (single large chunk), split by tokens directly
        if len(splits) == 1:
            split_tokens = self.tokenizer(splits[0])
            if split_tokens > self.chunk_size:
                # Split the original text by tokens directly
                return self._split_by_tokens_directly(text)
            else:
                # Single chunk that fits, return as is
                return splits

        # Merge splits based on token count
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for split in splits:
            split_tokens = self.tokenizer(split)

            # If split itself is too large, split it further
            if split_tokens > self.chunk_size:
                sub_chunks = self._split_by_tokens_directly(split)
                for sub_chunk in sub_chunks:
                    sub_tokens = self.tokenizer(sub_chunk)
                    if current_tokens + sub_tokens > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        # Start new chunk with overlap
                        if self.chunk_overlap > 0 and current_chunk:
                            overlap_text = self._get_overlap_text(
                                current_chunk, self.chunk_overlap
                            )
                            current_chunk = overlap_text + sub_chunk
                            current_tokens = self.tokenizer(current_chunk)
                        else:
                            current_chunk = sub_chunk
                            current_tokens = sub_tokens
                    else:
                        if current_chunk:
                            current_chunk += sub_chunk
                        else:
                            current_chunk = sub_chunk
                        current_tokens = self.tokenizer(current_chunk)
            # If adding this split would exceed chunk size
            elif current_tokens + split_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Get last N tokens for overlap
                    overlap_text = self._get_overlap_text(
                        current_chunk, self.chunk_overlap
                    )
                    current_chunk = overlap_text + split
                    current_tokens = self.tokenizer(current_chunk)
                else:
                    current_chunk = split
                    current_tokens = split_tokens
            else:
                if current_chunk:
                    current_chunk += split
                else:
                    current_chunk = split
                current_tokens = self.tokenizer(current_chunk)

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_tokens_directly(self, text: str) -> List[str]:
        """Split text directly by token count when no natural breaks exist."""
        chunks = []
        has_spaces = " " in text
        words = text.split() if has_spaces else list(text)  # Fallback to chars if no spaces
        current_chunk = []
        current_tokens = 0
        separator = " " if has_spaces else ""

        for word in words:
            # Calculate tokens for current word
            word_tokens = self.tokenizer(word)
            
            # Calculate tokens if we add this word
            test_chunk = current_chunk + [word]
            test_text = separator.join(test_chunk)
            test_tokens = self.tokenizer(test_text)
            
            if test_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(separator.join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Get overlap words/chars
                    overlap_count = max(1, min(len(current_chunk), self.chunk_overlap))
                    current_chunk = current_chunk[-overlap_count:]
                    current_tokens = self.tokenizer(separator.join(current_chunk))
                else:
                    current_chunk = []
                    current_tokens = 0
            
            # Add word to current chunk
            current_chunk.append(word)
            current_tokens = self.tokenizer(separator.join(current_chunk))

        # Add final chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def _get_overlap_text(self, text: str, target_tokens: int) -> str:
        """Get the last N tokens of text as a string."""
        # Simple approach: take last N characters (approximation)
        # For more accuracy, would need to decode tokens
        try:
            import tiktoken

            encoding = tiktoken.get_encoding(self.encoding_name)
            tokens = encoding.encode(text)
            if len(tokens) <= target_tokens:
                return text
            overlap_tokens = tokens[-target_tokens:]
            return encoding.decode(overlap_tokens)
        except (ImportError, Exception):
            # Fallback: use character-based approximation
            char_per_token = len(text) / max(self.tokenizer(text), 1)
            overlap_chars = int(target_tokens * char_per_token)
            return text[-overlap_chars:] if len(text) > overlap_chars else text

