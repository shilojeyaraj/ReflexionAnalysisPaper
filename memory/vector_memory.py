"""
Vector memory backend using ChromaDB + sentence-transformers for the Reflexion memory study.

Retrieval hypothesis: semantically similar tasks share lessons even across different
task IDs or error types. This is the key advantage over sliding window (recency)
and SQL (structure).
"""

import logging
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer
from memory.base import MemoryBackend

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "reflexion_episodes"


class VectorMemory(MemoryBackend):
    """
    Chroma-backed vector memory with semantic similarity retrieval.

    Episodes are embedded as: "{domain}: {action_summary} -> {reflection}"
    Retrieval embeds the current task description and finds the most
    semantically similar past episodes via cosine similarity.

    A configurable min_similarity threshold gates low-quality retrievals:
    if the top-k similarity score falls below this threshold, the episode
    is excluded rather than padding with irrelevant reflections.
    This is methodologically important and should be reported in the paper.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        min_similarity: float = 0.55,
    ) -> None:
        """
        Args:
            persist_dir: Directory for ChromaDB persistent storage.
            embedding_model: Sentence-transformers model name for embeddings.
            min_similarity: Minimum cosine similarity (0-1) for an episode to be returned.
                            Episodes below this threshold are excluded even if top-k.
                            Default 0.55 (configurable via config/vector.yaml).
            """
        self._min_similarity = min_similarity
        self._encoder = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_episode(self, episode: dict) -> list[float]:
        """Embed episode as '{domain}: {action_summary} -> {reflection}'."""
        text = f"{episode['domain']}: {episode['action_summary']} -> {episode['reflection']}"
        return self._encoder.encode(text).tolist()

    def _embed_query(self, description: str) -> list[float]:
        """Embed a task description for similarity search."""
        return self._encoder.encode(description).tolist()

    def store(self, episode: dict) -> None:
        """Embed and store the episode in Chroma."""
        embedding = self._embed_episode(episode)
        episode_id = f"{episode['task_id']}_{episode['attempt']}_{episode['timestamp']}"
        # Chroma metadata values must be str/int/float/bool
        metadata = {
            "task_id": str(episode["task_id"]),
            "domain": str(episode["domain"]),
            "attempt": int(episode["attempt"]),
            "success": bool(episode["success"]),
            "reward": float(episode["reward"]),
            "action_summary": str(episode["action_summary"]),
            "reflection": str(episode["reflection"]),
            "error_type": str(episode["error_type"]),
            "tokens_used": int(episode["tokens_used"]),
            "timestamp": str(episode["timestamp"]),
        }
        self._collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def retrieve(self, query: dict, k: int) -> list[dict]:
        """
        Retrieve up to k semantically similar episodes.

        Episodes with cosine similarity below min_similarity are excluded.
        Logs a warning when fewer than k episodes pass the threshold.
        """
        total = self.count()
        if total == 0:
            return []

        n_results = min(k, total)
        query_embedding = self._embed_query(query["current_task_description"])

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine distance in [0, 2]

        filtered: list[dict] = []
        for meta, dist in zip(metadatas, distances):
            # Convert cosine distance to similarity: similarity = 1 - (distance / 2)
            similarity = 1.0 - (dist / 2.0)
            if similarity >= self._min_similarity:
                episode = dict(meta)
                episode["_similarity"] = similarity
                filtered.append(episode)

        if len(filtered) < k:
            logger.warning(
                "VectorMemory: requested k=%d but only %d episodes passed "
                "min_similarity=%.2f threshold (total stored=%d). "
                "Warm-up phase or low-similarity pool.",
                k, len(filtered), self._min_similarity, total,
            )

        return filtered

    def reset(self) -> None:
        """Delete and recreate the Chroma collection."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return total number of stored episodes."""
        return self._collection.count()
