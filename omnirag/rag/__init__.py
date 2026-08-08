"""The RAG engine: chunk -> embed -> index -> retrieve -> rerank -> answer -> cite.

This package is completely independent of Streamlit and of any single vendor;
it is the part that migrates to FastAPI unchanged.
"""

from omnirag.rag.chunking import Chunker, chunk_documents
from omnirag.rag.citations import (
    CitationBundle,
    build_citations,
    verify_and_clean,
)
from omnirag.rag.embeddings import EmbeddingPipeline, build_embedding_pipeline
from omnirag.rag.generation import (
    AnswerGenerator,
    GenerationRequest,
    build_generator,
)
from omnirag.rag.hybrid import build_bm25_index, reciprocal_rank_fusion
from omnirag.rag.query_rewrite import QueryPlan, parse_query
from omnirag.rag.reranker import RerankingService, build_reranking_service
from omnirag.rag.retrieval import RetrievalRequest, Retriever, build_retriever
from omnirag.rag.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    QdrantVectorStore,
    SearchFilter,
    get_vector_store,
)

__all__ = [
    "AnswerGenerator",
    "BaseVectorStore",
    "Chunker",
    "CitationBundle",
    "EmbeddingPipeline",
    "GenerationRequest",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "QueryPlan",
    "RerankingService",
    "RetrievalRequest",
    "Retriever",
    "SearchFilter",
    "build_bm25_index",
    "build_citations",
    "build_embedding_pipeline",
    "build_generator",
    "build_reranking_service",
    "build_retriever",
    "chunk_documents",
    "get_vector_store",
    "parse_query",
    "reciprocal_rank_fusion",
    "verify_and_clean",
]
