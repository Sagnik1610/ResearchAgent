"""
Enhanced Knowledge Store with Per-Entity Hybrid Scoring

Provides entity retrieval using hybrid scoring that combines:
1. Sparse co-occurrence evidence (with Laplace smoothing)
2. Dense embedding signals (embedding-based fallback)

Supports both old method (co-occurrence only) and new hybrid method.
"""

import math
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

from utils.data_io import load_jsonl
from knowledge.hybrid_scorer import HybridScorer


class KnowledgeStore(object):
    """
    Knowledge store managing entities and their relationships.
    
    Supports two retrieval modes:
    - Legacy: Pure co-occurrence based (backward compatible)
    - Hybrid: Combines co-occurrence and embeddings (recommended)
    """
    
    def __init__(
        self,
        file_path: str,
        entity_embeddings: Optional[Dict[str, np.ndarray]] = None,
        use_hybrid_scoring: bool = True,
        alpha: float = 0.01,
        tau: float = 0.5,
        epsilon: float = 1e-9,
        beta: float = 0.3,
        threshold: int = 1,
        r: Optional[int] = None,
    ):
        """
        Initialize knowledge store.
        
        Args:
            file_path: Path to knowledge base JSONL file
            entity_embeddings: Optional dict mapping entity names to embedding vectors
            use_hybrid_scoring: If True, use hybrid scoring; else use legacy method
            alpha: Laplace smoothing constant
            tau: Embedding-to-probability scale factor
            epsilon: Probability floor
            beta: Mix weight between co-occurrence and embedding
            threshold: Minimum co-occurrence count
            r: Row normalizer for smoothing
        """
        super(KnowledgeStore, self).__init__()
        
        # Load knowledge base
        self.knowledge_base = load_jsonl(file_path)
        self.paper2entities = self.build_paper2entities()
        self.entity_counter, self.entity_cooccurrence = self.build_entity_statistics()
        self.entity_embeddings = entity_embeddings or {}
        
        # Configuration
        self.use_hybrid_scoring = use_hybrid_scoring
        
        # Initialize hybrid scorer if enabled
        if self.use_hybrid_scoring:
            self.hybrid_scorer = HybridScorer(
                entity_cooccurrence=self.entity_cooccurrence,
                entity_counter=self.entity_counter,
                entity_embeddings=self.entity_embeddings,
                alpha=alpha,
                tau=tau,
                epsilon=epsilon,
                beta=beta,
                r=r,
                threshold=threshold,
            )
        else:
            self.hybrid_scorer = None

    def build_paper2entities(self) -> Dict[int, Dict[str, int]]:
        """
        Build mapping from paper corpus ID to entities.
        
        Returns:
            Dict mapping corpus_id -> {entity_name -> count}
        """
        return {instance['corpusid']: instance['knowledge'] for instance in self.knowledge_base}

    def build_entity_statistics(self) -> Tuple[Counter, Dict[str, Counter]]:
        """
        Build entity statistics from knowledge base.
        
        Returns:
            Tuple of (entity_counter, entity_cooccurrence)
            - entity_counter: Count of papers each entity appears in
            - entity_cooccurrence: For each entity, counter of co-occurrences
        """
        entity_counter = Counter()
        entity_cooccurrence = defaultdict(Counter)

        for instance in self.knowledge_base:
            entities = instance['knowledge']
            entity_counter.update(entities)

            # Build co-occurrence counts
            for entity_name in entities.keys():
                entity_cooccurrence[entity_name].update(
                    {key: value for key, value in entities.items() if key != entity_name}
                )

        return entity_counter, entity_cooccurrence
    
    def _compute_local_embedding(
        self,
        local_entities: List[str],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Compute aggregated embedding for local entity set.
        
        Args:
            local_entities: List of local entity names
            weights: Optional weights for weighted mean (defaults to uniform)
            
        Returns:
            Mean embedding vector, or empty array if no embeddings available
        """
        embeddings = []
        
        for entity in local_entities:
            if entity in self.entity_embeddings:
                embeddings.append(self.entity_embeddings[entity])
        
        if not embeddings:
            return np.array([])
        
        embeddings_array = np.array(embeddings)
        
        if weights is None:
            return np.mean(embeddings_array, axis=0)
        else:
            # Weighted mean
            weights_array = np.array(weights[:len(embeddings)])
            weights_array = weights_array / np.sum(weights_array)  # Normalize
            return np.average(embeddings_array, axis=0, weights=weights_array)
    
    def get_entity_log_likelihood(self, entity: str, paper_entities: List[str]) -> float:
        """
        Legacy method: Get log likelihood of entity given paper entities.
        
        Used for backward compatibility with original implementation.
        
        Args:
            entity: Candidate entity
            paper_entities: List of entities from paper
            
        Returns:
            Sum of log probabilities
        """
        conditional_log_probabilities = [
            math.log2(
                (self.entity_cooccurrence[entity][paper_entity] + 1e-16) /
                (sum(self.entity_cooccurrence[entity].values()) + 1e-16)
            ) for paper_entity in paper_entities
        ]
        return sum(conditional_log_probabilities)

    def get_entity_probability(self, entity: str) -> float:
        """
        Get prior probability of entity.
        
        Args:
            entity: Entity name
            
        Returns:
            Probability in [0, 1]
        """
        return self.entity_counter[entity] / sum(self.entity_counter.values())

    def get_relevant_entities(
        self,
        paper_ids: List[int],
        top_k: int = 30,
        weights: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Retrieve top-k relevant entities using configured scoring method.
        
        Args:
            paper_ids: List of paper corpus IDs to extract entities from
            top_k: Number of top entities to retrieve
            weights: Optional weights for entities (for weighted local embedding)
            
        Returns:
            List of top-k entity names
        """
        # Extract local entities from papers
        paper_entities = sum(
            [
                Counter(self.paper2entities[paper_id]) 
                for paper_id in paper_ids 
                if paper_id in self.paper2entities.keys()
            ], 
            start=Counter()
        )
        paper_entities_list = list(paper_entities.elements())

        # Find candidate entities (those that co-occur with local entities)
        candidate_entities = sum(
            [self.entity_cooccurrence[entity] for entity in paper_entities_list],
            start=Counter()
        )
        candidate_entities_filtered = [
            entity for entity, count in candidate_entities.items() if count >= 1
        ]

        # Use appropriate retrieval method
        if self.use_hybrid_scoring and self.hybrid_scorer is not None:
            return self._get_relevant_entities_hybrid(
                candidate_entities_filtered,
                paper_entities_list,
                top_k,
                weights,
            )
        else:
            return self._get_relevant_entities_legacy(
                candidate_entities_filtered,
                paper_entities_list,
                top_k,
            )

    def _get_relevant_entities_legacy(
        self,
        candidate_entities: List[str],
        local_entities: List[str],
        top_k: int,
    ) -> List[str]:
        """
        Legacy retrieval method (backward compatible).
        
        Uses original co-occurrence based scoring.
        """
        candidate_entities_probs = [
            (
                self.get_entity_log_likelihood(entity, local_entities) + 
                math.log2(self.get_entity_probability(entity) + 1e-16)
            )
            for entity in candidate_entities
        ]

        # Get top-k indices
        if len(candidate_entities_probs) > 0:
            top_indices = sorted(
                range(len(candidate_entities_probs)),
                key=lambda i: candidate_entities_probs[i],
                reverse=True
            )[:min(top_k, len(candidate_entities))]
            
            return [candidate_entities[index] for index in top_indices]
        
        return []

    def _get_relevant_entities_hybrid(
        self,
        candidate_entities: List[str],
        local_entities: List[str],
        top_k: int,
        weights: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Hybrid retrieval method using Per-Entity Hybrid Scoring.
        
        Combines co-occurrence and embedding signals.
        """
        # Compute local embedding
        local_embedding = self._compute_local_embedding(local_entities, weights)
        
        # Get top-k using hybrid scorer
        return self.hybrid_scorer.get_top_k_entities(
            candidate_entities,
            local_entities,
            local_embedding=local_embedding,
            k=top_k,
        )

    def get_relevant_entities_with_scores(
        self,
        paper_ids: List[int],
        top_k: int = 30,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k entities with their scores.
        
        Args:
            paper_ids: List of paper corpus IDs
            top_k: Number of entities to retrieve
            weights: Optional weights for entities
            
        Returns:
            List of (entity, score) tuples sorted by score
        """
        if not self.use_hybrid_scoring or self.hybrid_scorer is None:
            # Legacy method doesn't expose scores cleanly
            entities = self.get_relevant_entities(paper_ids, top_k, weights)
            return [(e, 0.0) for e in entities]
        
        # Extract local entities from papers
        paper_entities = sum(
            [
                Counter(self.paper2entities[paper_id]) 
                for paper_id in paper_ids 
                if paper_id in self.paper2entities.keys()
            ], 
            start=Counter()
        )
        paper_entities_list = list(paper_entities.elements())

        # Find candidate entities
        candidate_entities = sum(
            [self.entity_cooccurrence[entity] for entity in paper_entities_list],
            start=Counter()
        )
        candidate_entities_filtered = [
            entity for entity, count in candidate_entities.items() if count >= 1
        ]

        # Compute local embedding
        local_embedding = self._compute_local_embedding(paper_entities_list, weights)
        
        # Get scored entities
        return self.hybrid_scorer.get_top_k_entities_with_scores(
            candidate_entities_filtered,
            paper_entities_list,
            local_embedding=local_embedding,
            k=top_k,
        )

    def get_scorer_config(self) -> Dict[str, float]:
        """
        Get current hybrid scorer configuration.
        
        Returns:
            Dict with hyperparameter values
        """
        if self.hybrid_scorer is None:
            return {"method": "legacy"}
        
        return {
            "method": "hybrid",
            "alpha": self.hybrid_scorer.alpha,
            "tau": self.hybrid_scorer.tau,
            "epsilon": self.hybrid_scorer.epsilon,
            "beta": self.hybrid_scorer.beta,
            "threshold": self.hybrid_scorer.threshold,
            "r": self.hybrid_scorer.r,
        }