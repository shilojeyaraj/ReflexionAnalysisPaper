"""
Memory backends for the Reflexion memory study.

Provides three implementations of MemoryBackend:
- SlidingWindowMemory: recency-based baseline (mirrors original Reflexion paper)
- SQLMemory: structured retrieval via SQLite (domain + error_type filters)
- VectorMemory: semantic retrieval via ChromaDB + sentence-transformers
"""

from memory.base import MemoryBackend
from memory.sliding_window import SlidingWindowMemory
from memory.sql_memory import SQLMemory
from memory.vector_memory import VectorMemory

__all__ = ["MemoryBackend", "SlidingWindowMemory", "SQLMemory", "VectorMemory"]
