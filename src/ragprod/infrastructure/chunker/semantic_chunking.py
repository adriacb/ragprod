import math
from typing import List, Optional, Literal
from .base import BaseTextSplitter
from ragprod.domain.embedding import EmbeddingModel


class SemanticChunker(BaseTextSplitter):
    """
    Split text based on semantic similarity.
    
    Similar to LangChain's SemanticChunker, this splitter:
    1. Splits text into sentences
    2. Groups sentences into windows (default: 3 sentences)
    3. Calculates embeddings for each window
    4. Merges windows that are similar in embedding space
    
    Based on Greg Kamradt's approach from FullStackRetrieval tutorials.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            embedding_model: Embedding model to use for semantic similarity.
            buffer_size: Number of sentences to include in each window.
            add_start_index: Whether to add start index to metadata.
            breakpoint_threshold_type: Method to determine similarity threshold.
            breakpoint_threshold_amount: Threshold amount (None = auto-calculate).
            number_of_chunks: Target number of chunks (None = auto-determine).
            sentence_split_regex: Regex pattern to split sentences.
        """
        import re

        self.embedding_model = embedding_model
        self.buffer_size = buffer_size
        self.add_start_index = add_start_index
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = re.compile(sentence_split_regex)

    async def split_text_async(self, text: str) -> List[str]:
        """
        Asynchronously split text based on semantic similarity.
        
        Args:
            text: The text to split.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= self.buffer_size:
            return [text]

        # Create sentence windows
        windows = self._create_windows(sentences)

        # Get embeddings for windows
        window_texts = [" ".join(window) for window in windows]
        
        # Check if embed_documents is async
        if hasattr(self.embedding_model.embed_documents, '__call__'):
            import inspect
            if inspect.iscoroutinefunction(self.embedding_model.embed_documents):
                embeddings = await self.embedding_model.embed_documents(window_texts)
            else:
                embeddings = self.embedding_model.embed_documents(window_texts)
        else:
            # Fallback: try async first
            try:
                embeddings = await self.embedding_model.embed_documents(window_texts)
            except TypeError:
                # If not async, call directly
                embeddings = self.embedding_model.embed_documents(window_texts)

        # Calculate cosine similarities between consecutive windows
        similarities = self._calculate_similarities(embeddings)

        # Determine breakpoints based on threshold
        breakpoints = self._find_breakpoints(similarities)

        # Split text at breakpoints
        chunks = self._create_chunks(sentences, breakpoints)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split text based on semantic similarity (synchronous wrapper).
        
        Args:
            text: The text to split.
            
        Returns:
            List of text chunks.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, we need to use a different approach
                # For now, fall back to simple sentence-based splitting
                return self._fallback_split(text)
            return loop.run_until_complete(self.split_text_async(text))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.split_text_async(text))

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_split_regex.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_windows(self, sentences: List[str]) -> List[List[str]]:
        """Create sliding windows of sentences."""
        windows = []
        for i in range(len(sentences) - self.buffer_size + 1):
            window = sentences[i : i + self.buffer_size]
            windows.append(window)
        return windows

    def _calculate_similarities(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarity between consecutive embeddings."""
        if len(embeddings) < 2:
            return []

        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(
                embeddings[i], embeddings[i + 1]
            )
            similarities.append(similarity)

        return similarities

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """Find breakpoints where similarity drops below threshold."""
        if not similarities:
            return []

        threshold = self._calculate_threshold(similarities)

        breakpoints = []
        for i, similarity in enumerate(similarities):
            if similarity < threshold:
                breakpoints.append(i + 1)  # +1 because breakpoint is after window i

        return breakpoints

    def _calculate_threshold(self, similarities: List[float]) -> float:
        """Calculate similarity threshold based on method."""
        if self.breakpoint_threshold_amount is not None:
            return self.breakpoint_threshold_amount

        if self.breakpoint_threshold_type == "percentile":
            # Use percentile (default: 10th percentile)
            sorted_sims = sorted(similarities)
            percentile_index = int(len(sorted_sims) * 0.1)
            return sorted_sims[percentile_index] if sorted_sims else 0.5

        elif self.breakpoint_threshold_type == "standard_deviation":
            # Mean minus one standard deviation
            mean = sum(similarities) / len(similarities)
            variance = sum((x - mean) ** 2 for x in similarities) / len(
                similarities
            )
            std_dev = math.sqrt(variance)
            return mean - std_dev

        elif self.breakpoint_threshold_type == "interquartile":
            # First quartile
            sorted_sims = sorted(similarities)
            q1_index = len(sorted_sims) // 4
            return sorted_sims[q1_index] if sorted_sims else 0.5

        elif self.breakpoint_threshold_type == "gradient":
            # Use gradient-based approach
            if len(similarities) < 2:
                return 0.5
            # Find largest drop
            drops = [
                similarities[i] - similarities[i + 1]
                for i in range(len(similarities) - 1)
            ]
            if not drops:
                return 0.5
            max_drop_index = drops.index(max(drops))
            return similarities[max_drop_index]

        return 0.5  # Default threshold

    def _create_chunks(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        """Create chunks from sentences and breakpoints."""
        if not breakpoints:
            return [" ".join(sentences)]

        chunks = []
        start = 0

        for breakpoint in breakpoints:
            chunk_sentences = sentences[start:breakpoint]
            chunks.append(" ".join(chunk_sentences))
            start = breakpoint

        # Add final chunk
        if start < len(sentences):
            chunk_sentences = sentences[start:]
            chunks.append(" ".join(chunk_sentences))

        return [chunk for chunk in chunks if chunk.strip()]

    def _fallback_split(self, text: str) -> List[str]:
        """Fallback splitting method when async is not available."""
        sentences = self._split_sentences(text)
        # Simple chunking: group sentences into windows
        chunks = []
        for i in range(0, len(sentences), self.buffer_size * 3):
            chunk_sentences = sentences[i : i + self.buffer_size * 3]
            chunks.append(" ".join(chunk_sentences))
        return chunks
