import pytest
import torch
from unittest.mock import MagicMock

from ragprod.infrastructure.embeddings import (
    HuggingFaceEmbeddings,
    ColBERTEmbeddings,
    OpenAIEmbeddings,
)

# ---------------------------------------------------------------------------
# HUGGINGFACE EMBEDDINGS TESTS
# ---------------------------------------------------------------------------

def test_hf_lazy_model_load(mocker):
    mock_sentence_transformer = MagicMock()
    mocker.patch(
        "ragprod.infrastructure.embeddings.huggingface_embeddings.SentenceTransformer",
        return_value=mock_sentence_transformer,
    )

    emb = HuggingFaceEmbeddings(model_name="fake-model")
    assert emb._model is None

    _ = emb.model
    assert emb._model is mock_sentence_transformer


@pytest.mark.asyncio
async def test_hf_embed_query_async(mocker):
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1, 0.2]

    mocker.patch(
        "ragprod.infrastructure.embeddings.huggingface_embeddings.SentenceTransformer",
        return_value=mock_model,
    )

    emb = HuggingFaceEmbeddings(model_name="fake")
    result = await emb.embed_query("hello")
    assert result == [0.1, 0.2]


# ---------------------------------------------------------------------------
# COLBERT EMBEDDINGS TESTS
# ---------------------------------------------------------------------------

def test_colbert_lazy_load(mocker):
    fake_model = MagicMock()
    fake_model.dim = 128

    # Patch ColBERTConfig to avoid real constructor signature
    mocker.patch(
        "ragprod.infrastructure.embeddings.colbert_embeddings.ColBERTConfig",
        return_value=MagicMock()
    )
    # Patch ColBERT class itself
    mocker.patch(
        "ragprod.infrastructure.embeddings.colbert_embeddings.ColBERT",
        return_value=fake_model
    )

    emb = ColBERTEmbeddings(model_name="fake-colbert")
    assert emb._model is None

    _ = emb.model  # triggers instantiation
    assert emb._model is fake_model
    assert emb.get_dimension() == 128


@pytest.mark.asyncio
async def test_colbert_async_embeddings(mocker):
    fake_model = MagicMock()
    fake_model.query.return_value = torch.tensor([[1.0, 2.0]])
    fake_model.doc.return_value = torch.tensor([[1.0, 2.0]])

    mocker.patch(
        "ragprod.infrastructure.embeddings.colbert_embeddings.ColBERTConfig",
        return_value=MagicMock()
    )
    mocker.patch(
        "ragprod.infrastructure.embeddings.colbert_embeddings.ColBERT",
        return_value=fake_model
    )

    emb = ColBERTEmbeddings()

    q = await emb.embed_query("hello")
    d = await emb.embed_documents(["hello"])

    assert torch.equal(q, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(d, torch.tensor([[1.0, 2.0]]))


def test_colbert_similarity_gpu_optimized(mocker):
    fake_model = MagicMock()
    fake_model.query.return_value = torch.tensor([[1.0, 0.0]])
    fake_model.doc.return_value = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    fake_model.dim = 2

    mocker.patch(
        "ragprod.infrastructure.embeddings.colbert_embeddings.ColBERTConfig",
        return_value=MagicMock()
    )
    mocker.patch(
        "ragprod.infrastructure.embeddings.colbert_embeddings.ColBERT",
        return_value=fake_model
    )

    emb = ColBERTEmbeddings()

    scores = emb.similarity("hi", ["A", "B"])
    assert scores == pytest.approx([1.0, 0.0])


# ---------------------------------------------------------------------------
# OPENAI EMBEDDINGS TESTS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_embed_query(mocker):
    mocker.patch(
        "ragprod.infrastructure.embeddings.openai_embeddings.openai.Embedding.create",
        return_value={"data": [{"embedding": [0.1, 0.2]}]},
    )

    emb = OpenAIEmbeddings()
    result = await emb.embed_query("hello")
    assert result == [0.1, 0.2]


@pytest.mark.asyncio
async def test_openai_embed_documents_batching(mocker):
    def fake_create(input, model):
        return {"data": [{"embedding": [1, 2]} for _ in input]}

    mock_create = mocker.patch(
        "ragprod.infrastructure.embeddings.openai_embeddings.openai.Embedding.create",
        side_effect=fake_create
    )

    emb = OpenAIEmbeddings(batch_size=2)
    texts = ["a", "b", "c", "d"]
    result = await emb.embed_documents(texts)

    assert mock_create.call_count == 2
    assert len(result) == 4
    assert all(r == [1, 2] for r in result)


def test_openai_dimension():
    emb = OpenAIEmbeddings(model_name="text-embedding-3-large")
    assert emb.get_dimension() == 3072
