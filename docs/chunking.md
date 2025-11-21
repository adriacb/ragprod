# Document Chunking

RAGProd provides a comprehensive set of text splitting/chunking strategies to break down documents into smaller, manageable pieces. This is crucial for RAG systems as it affects retrieval quality and context window management.

## Overview

Chunking is the process of dividing large documents into smaller segments (chunks) that can be:
- Embedded efficiently
- Retrieved more accurately
- Fitted into LLM context windows
- Processed in parallel

RAGProd offers multiple chunking strategies, each optimized for different use cases and document types.

## Base Interface

All chunkers inherit from `BaseTextSplitter` and implement a consistent interface:

```python
from ragprod.infrastructure.chunker import BaseTextSplitter

# Main methods available on all chunkers:
splitter.split_text(text: str) -> List[str]
splitter.split_documents(documents: List[Document]) -> List[Document]
splitter.create_documents(texts: List[str], metadatas: Optional[List[Dict]]) -> List[Document]
```

## Chunking Strategies

### 1. RecursiveCharacterTextSplitter

**Best for**: General-purpose text splitting with intelligent separator handling.

The `RecursiveCharacterTextSplitter` tries to split text using a hierarchy of separators, falling back to the next separator if chunks are still too large. This ensures chunks respect natural boundaries (paragraphs, sentences, words) when possible.

#### Features

- **Hierarchical splitting**: Tries separators in order (paragraphs → lines → words → characters)
- **Automatic fallback**: If one separator doesn't work, tries the next
- **Configurable separators**: Customize the splitting hierarchy
- **Overlap support**: Maintains context between chunks

#### Usage

```python
from ragprod.infrastructure.chunker import RecursiveCharacterTextSplitter
from ragprod.domain.document import Document

# Basic usage
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum characters per chunk
    chunk_overlap=200,      # Characters to overlap between chunks
)

text = """
This is a long document with multiple paragraphs.

Each paragraph contains multiple sentences. 
The splitter will try to keep paragraphs together.

If paragraphs are too large, it will split by sentences.
If sentences are too large, it will split by words.
"""

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:100]}...")
```

#### Custom Separators

```python
# Custom separator hierarchy
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],  # Custom order
    keep_separator=True,  # Keep separators in chunks
)

chunks = splitter.split_text(text)
```

#### With Documents

```python
doc = Document(
    raw_text="Your long document text here...",
    source="document.pdf",
    title="My Document"
)

chunked_docs = splitter.split_documents([doc])

for chunk in chunked_docs:
    print(f"Source: {chunk.source}")
    print(f"Chunk {chunk.metadata['chunk_index']}/{chunk.metadata['total_chunks']}")
    print(f"Content: {chunk.content[:100]}...")
```

#### Parameters

- `chunk_size` (int): Maximum size of chunks (default: 1000)
- `chunk_overlap` (int): Overlap between chunks (default: 200)
- `separators` (List[str]): List of separators to try in order (default: `["\n\n", "\n", " ", ""]`)
- `length_function` (callable): Function to measure text length (default: `len()`)
- `keep_separator` (bool): Whether to keep separator in chunks (default: False)

---

### 2. MarkdownHeaderTextSplitter

**Best for**: Markdown documents, documentation, structured content with headers.

The `MarkdownHeaderTextSplitter` divides Markdown documents based on header hierarchy, preserving the document structure and header metadata in each chunk.

#### Features

- **Header-aware splitting**: Splits at header boundaries
- **Metadata preservation**: Each chunk includes header hierarchy information
- **Configurable header levels**: Choose which header levels to split on
- **Structure preservation**: Maintains document organization

#### Usage

```python
from ragprod.infrastructure.chunker import MarkdownHeaderTextSplitter
from ragprod.domain.document import Document

# Basic usage
splitter = MarkdownHeaderTextSplitter(
    strip_headers=True,  # Remove header text from chunk content
)

markdown_text = """
# Introduction

This is the introduction section.

## Getting Started

Here's how to get started.

### Installation

Install the package using pip.

## Advanced Usage

For advanced users.

### Configuration

Configure the system.
"""

chunks = splitter.split_text(markdown_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n---")
```

#### With Header Metadata

```python
doc = Document(
    raw_text=markdown_text,
    source="README.md",
    title="Project Documentation"
)

chunked_docs = splitter.split_documents([doc])

for chunk in chunked_docs:
    print(f"Headers: {chunk.metadata}")
    print(f"Content: {chunk.content[:100]}...\n")
    # Metadata will include: header_1, header_2, etc.
```

#### Custom Header Levels

```python
# Only split on H1 and H2 headers
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
    ],
    strip_headers=False,  # Keep headers in content
)

chunks = splitter.split_text(markdown_text)
```

#### Parameters

- `headers_to_split_on` (List[Tuple[str, str]]): Header levels to split on (default: all levels # through ######)
- `strip_headers` (bool): Whether to remove headers from chunk content (default: True)

#### Use Cases

- Documentation websites
- Technical documentation
- Structured markdown files
- Wiki content
- README files

---

### 3. CharacterTextSplitter

**Best for**: Simple, fast splitting when structure doesn't matter.

The `CharacterTextSplitter` is a straightforward character-based splitter that divides text into fixed-size chunks with optional overlap.

#### Usage

```python
from ragprod.infrastructure.chunker import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50,    # Overlap between chunks
    separator="",        # Optional separator (empty = no separator)
)

text = "Your long text here that needs to be split into smaller pieces..."

chunks = splitter.split_text(text)
```

#### With Separator

```python
# Split at sentence boundaries when possible
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separator=". ",  # Try to split at sentence boundaries
)

chunks = splitter.split_text(text)
```

#### Parameters

- `chunk_size` (int): Maximum characters per chunk (default: 1000)
- `chunk_overlap` (int): Character overlap between chunks (default: 200)
- `separator` (str): Optional separator to prefer when splitting (default: "")

#### Use Cases

- Simple text processing
- When document structure is not important
- Fast, lightweight splitting
- Binary or unstructured text

---

### 4. TokenTextSplitter

**Best for**: LLM context management, token-limited scenarios.

The `TokenTextSplitter` splits text based on token count rather than characters, which is essential when working with LLMs that have token-based context limits.

#### Features

- **Token-aware**: Uses actual token counts, not character estimates
- **LLM-optimized**: Designed for context window management
- **Multiple encodings**: Supports different tokenization schemes
- **Smart merging**: Combines natural splits based on token limits

#### Usage

```python
from ragprod.infrastructure.chunker import TokenTextSplitter

# Basic usage (uses tiktoken with cl100k_base encoding for GPT models)
splitter = TokenTextSplitter(
    chunk_size=1000,      # Maximum tokens per chunk
    chunk_overlap=200,    # Token overlap between chunks
)

text = "Your long text here..."

chunks = splitter.split_text(text)
```

#### Custom Tokenizer

```python
# Custom tokenizer function
def my_tokenizer(text: str) -> int:
    # Your custom token counting logic
    return len(text.split())  # Simple word count

splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    tokenizer=my_tokenizer,
)

chunks = splitter.split_text(text)
```

#### Different Encodings

```python
# For different models (requires tiktoken)
splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name="p50k_base",  # For older GPT models
)

chunks = splitter.split_text(text)
```

#### Parameters

- `chunk_size` (int): Maximum tokens per chunk (default: 1000)
- `chunk_overlap` (int): Token overlap between chunks (default: 200)
- `tokenizer` (Callable): Optional custom tokenizer function (default: uses tiktoken)
- `encoding_name` (str): Encoding name for tiktoken (default: "cl100k_base")

#### Use Cases

- GPT model context management
- Token-budgeted applications
- API cost optimization
- Precise context window control

#### Note

Requires `tiktoken` package for default tokenization. Falls back to character-based splitting if not available.

---

### 5. SemanticChunker

**Best for**: Preserving semantic coherence, content-aware splitting.

The `SemanticChunker` uses embeddings to identify semantically similar content and creates chunks that maintain semantic coherence. It's based on Greg Kamradt's approach and similar to LangChain's SemanticChunker.

#### Features

- **Semantic awareness**: Uses embeddings to identify natural boundaries
- **Multiple threshold methods**: Percentile, standard deviation, interquartile, gradient
- **Sentence-based**: Works with sentence windows
- **Async support**: Supports async embedding models

#### Usage

```python
from ragprod.infrastructure.chunker import SemanticChunker
from ragprod.application.use_cases import get_embeddings

# Get an embedding model
embedder = get_embeddings(
    model_type="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create semantic chunker
splitter = SemanticChunker(
    embedding_model=embedder,
    buffer_size=3,  # Sentences per window
    breakpoint_threshold_type="percentile",  # Threshold calculation method
)

# Async usage (recommended)
chunks = await splitter.split_text_async(text)

# Or synchronous (with fallback)
chunks = splitter.split_text(text)
```

#### Threshold Methods

```python
# Percentile-based (default)
splitter = SemanticChunker(
    embedding_model=embedder,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=None,  # Auto-calculate (10th percentile)
)

# Standard deviation
splitter = SemanticChunker(
    embedding_model=embedder,
    breakpoint_threshold_type="standard_deviation",
)

# Interquartile range
splitter = SemanticChunker(
    embedding_model=embedder,
    breakpoint_threshold_type="interquartile",
)

# Gradient-based (finds largest similarity drop)
splitter = SemanticChunker(
    embedding_model=embedder,
    breakpoint_threshold_type="gradient",
)
```

#### Custom Threshold

```python
# Manual threshold
splitter = SemanticChunker(
    embedding_model=embedder,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=0.3,  # Fixed threshold
)
```

#### Custom Sentence Splitting

```python
# Custom regex for sentence splitting
splitter = SemanticChunker(
    embedding_model=embedder,
    sentence_split_regex=r"(?<=[.!?])\s+",  # Custom pattern
)
```

#### Parameters

- `embedding_model` (EmbeddingModel): Embedding model for semantic similarity (required)
- `buffer_size` (int): Number of sentences per window (default: 1)
- `add_start_index` (bool): Add start index to metadata (default: False)
- `breakpoint_threshold_type` (str): Method for threshold calculation (default: "percentile")
- `breakpoint_threshold_amount` (float): Manual threshold value (default: None = auto)
- `number_of_chunks` (int): Target number of chunks (default: None = auto)
- `sentence_split_regex` (str): Regex for sentence splitting (default: `r"(?<=[.?!])\s+"`)

#### Use Cases

- Long-form content
- Articles and blog posts
- Research papers
- When semantic coherence is critical
- Content with varying topics

#### How It Works

1. Splits text into sentences
2. Creates sliding windows of sentences (default: 3 sentences)
3. Generates embeddings for each window
4. Calculates cosine similarity between consecutive windows
5. Identifies breakpoints where similarity drops below threshold
6. Creates chunks at breakpoints

---

## Comparison Guide

### When to Use Each Chunker

| Chunker | Best For | Speed | Quality | Complexity |
|---------|----------|-------|---------|------------|
| **RecursiveCharacterTextSplitter** | General purpose, most documents | Fast | High | Low |
| **MarkdownHeaderTextSplitter** | Markdown, documentation | Fast | Very High | Low |
| **CharacterTextSplitter** | Simple, unstructured text | Very Fast | Medium | Very Low |
| **TokenTextSplitter** | LLM context management | Fast | High | Medium |
| **SemanticChunker** | Semantic coherence, long-form | Slow | Very High | High |

### Performance Considerations

- **Fastest**: CharacterTextSplitter
- **Most Accurate**: SemanticChunker (for semantic coherence)
- **Best Structure Preservation**: MarkdownHeaderTextSplitter
- **Best for LLMs**: TokenTextSplitter
- **Most Versatile**: RecursiveCharacterTextSplitter

## Best Practices

### 1. Choose the Right Chunker

- **Structured documents (Markdown, HTML)**: Use `MarkdownHeaderTextSplitter`
- **General text**: Use `RecursiveCharacterTextSplitter`
- **LLM applications**: Use `TokenTextSplitter`
- **Semantic coherence important**: Use `SemanticChunker`
- **Simple, fast splitting**: Use `CharacterTextSplitter`

### 2. Chunk Size Guidelines

- **Small chunks (100-500 chars/tokens)**: Better precision, more chunks
- **Medium chunks (500-2000 chars/tokens)**: Balanced approach
- **Large chunks (2000+ chars/tokens)**: Better context, fewer chunks

### 3. Overlap Considerations

- **No overlap (0)**: Fastest, but may lose context at boundaries
- **Small overlap (50-200)**: Good balance for most cases
- **Large overlap (200-500)**: Better context preservation, more redundancy

### 4. Metadata Preservation

All chunkers automatically preserve:
- Original document metadata (source, title, etc.)
- Chunk index and total chunks
- Chunk length
- Header information (for MarkdownHeaderTextSplitter)

### 5. Combining Strategies

You can chain chunkers for complex scenarios:

```python
# First split by headers, then recursively split large chunks
md_splitter = MarkdownHeaderTextSplitter()
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# Split by headers first
header_chunks = md_splitter.split_documents([document])

# Then recursively split any large chunks
final_chunks = []
for chunk in header_chunks:
    if len(chunk.content) > 1000:
        sub_chunks = recursive_splitter.split_documents([chunk])
        final_chunks.extend(sub_chunks)
    else:
        final_chunks.append(chunk)
```

## Complete Example

```python
from ragprod.infrastructure.chunker import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)
from ragprod.domain.document import Document

# Example: Processing a technical documentation file
doc = Document(
    raw_text="""
# API Documentation

## Authentication

To authenticate, use the API key.

### Getting an API Key

Contact support to get your API key.

## Endpoints

### GET /users

Returns a list of users.

### POST /users

Creates a new user.
""",
    source="api_docs.md",
    title="API Documentation"
)

# Option 1: Split by headers (preserves structure)
md_splitter = MarkdownHeaderTextSplitter()
header_chunks = md_splitter.split_documents([doc])

for chunk in header_chunks:
    print(f"Header: {chunk.metadata.get('header_1', 'N/A')}")
    print(f"Content: {chunk.content[:100]}...\n")

# Option 2: Recursive character splitting (general purpose)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
recursive_chunks = recursive_splitter.split_documents([doc])

# Option 3: Token-based (for LLM)
token_splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
token_chunks = token_splitter.split_documents([doc])

# Use chunks with your retrieval system
from ragprod.application.use_cases import get_client_service

client = await get_client_service(mode="local_persistent")
await client.add_documents(header_chunks)
```

## Integration with RAG Pipeline

```python
from ragprod.infrastructure.chunker import RecursiveCharacterTextSplitter
from ragprod.application.use_cases import get_client_service, get_embeddings

# 1. Load document
doc = Document(raw_text="...", source="document.pdf")

# 2. Chunk document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents([doc])

# 3. Add to vector database
client = await get_client_service(mode="local_persistent")
await client.add_documents(chunks)

# 4. Retrieve relevant chunks
results = await client.retrieve("query", k=5)
```

## Troubleshooting

### Chunks Too Large

- Reduce `chunk_size`
- Use `TokenTextSplitter` for precise control
- Check if separators are working correctly

### Chunks Too Small

- Increase `chunk_size`
- Adjust `chunk_overlap` if needed
- Check separator configuration

### Poor Retrieval Quality

- Try `SemanticChunker` for better semantic coherence
- Use `MarkdownHeaderTextSplitter` for structured documents
- Adjust overlap to preserve more context

### Performance Issues

- Use `CharacterTextSplitter` for fastest processing
- Avoid `SemanticChunker` for large-scale processing
- Consider batch processing

## Related Documentation

- [Architecture Documentation](architecture.md) - Overall system architecture
- [Evaluator Documentation](evaluator.md) - Evaluating retrieval performance
- [Embeddings Documentation](../README.md) - Embedding models

