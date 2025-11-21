"""
Per-Entity Hybrid Scoring Module

Combines sparse co-occurrence evidence with dense embedding signals for robust
entity relevance scoring. Implements the mathematical framework from:
"Per-Entity Hybrid Scoring: Full Explanation, Mathematics, and Implementation"

Key Features:
- Laplace smoothing for co-occurrence probabilities
- Conservative embedding-based fallback
- Log-space aggregation for numerical stability
- Tunable mixing parameter for exploration vs. conservatism
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


class HybridScorer:
    """
    Hybrid scoring mechanism combining co-occurrence and embedding-based signals.
    
    Attributes:
        entity_cooccurrence: Dict mapping entity -> Counter of co-occurrence counts
        entity_counter: Counter of entity occurrences
        entity_embeddings: Dict mapping entity -> embedding vector (optional)
        alpha: Laplace smoothing constant
        tau: Embedding-to-probability scale factor
        epsilon: Probability floor to avoid log(0)
        beta: Mixing weight between co-occurrence and embedding signals
        r: Row normalizer for smoothing (typically vocab size or average neighbors)
        threshold: Minimum co-occurrence count to use smoothed probability
    """
    
    def __init__(
        self,
        entity_cooccurrence: Dict[str, Counter],
        entity_counter: Counter,
        entity_embeddings: Optional[Dict[str, np.ndarray]] = None,
        alpha: float = 0.01,
        tau: float = 0.5,
        epsilon: float = 1e-9,
        beta: float = 0.3,
        r: Optional[int] = None,
        threshold: int = 1,
    ):
        """
        Initialize the hybrid scorer.
        
        Args:
            entity_cooccurrence: Co-occurrence matrix as nested dict/Counter
            entity_counter: Marginal counts for each entity
            entity_embeddings: Optional embeddings dict for fallback signals
            alpha: Laplace smoothing constant (0.01 to 1)
            tau: Embedding-to-probability scale (0 to 1, typically 0.4-0.8)
            epsilon: Floor for probabilities to avoid log(0) (e.g., 1e-9)
            beta: Mix weight: 0=pure co-occurrence, 1=pure embedding
            r: Row normalizer (defaults to total_counts if None)
            threshold: Min co-occurrence count to use smoothed probability
        """
        self.entity_cooccurrence = entity_cooccurrence
        self.entity_counter = entity_counter
        self.entity_embeddings = entity_embeddings or {}
        
        # Hyperparameters
        self.alpha = alpha
        self.tau = tau
        self.epsilon = epsilon
        self.beta = beta
        self.threshold = threshold
        
        # Compute normalizing constants
        self.total_counts = sum(entity_counter.values())
        self.r = r if r is not None else self.total_counts
        
    def _compute_smoothed_cooccurrence_prob(
        self,
        entity: str,
        local_entity: str,
    ) -> float:
        """
        Compute smoothed co-occurrence conditional probability.
        
        Formula: P̂_co(e_j|e) = (K[e,e_j] + α) / (count(e) + αR)
        
        Args:
            entity: Candidate external entity
            local_entity: Local entity to condition on
            
        Returns:
            Smoothed conditional probability in (0, 1]
        """
        cooccurrence_count = self.entity_cooccurrence.get(entity, Counter()).get(local_entity, 0)
        entity_count = self.entity_counter.get(entity, 0)
        
        numerator = cooccurrence_count + self.alpha
        denominator = entity_count + self.alpha * self.r
        
        # Avoid division by zero
        if denominator == 0:
            return self.epsilon
        
        return max(self.epsilon, numerator / denominator)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Embedding vectors
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _compute_embedding_fallback_prob(
        self,
        entity: str,
        local_entity: str,
    ) -> float:
        """
        Compute conservative embedding-based pseudo-probability.
        
        Formula: P̂_emb(e_j|e) = max(ε, τ × (cos(v(e), v(e_j)) + 1) / 2)
        
        Maps cosine similarity in [-1, 1] to probability in [0, 1] conservatively.
        
        Args:
            entity: Candidate external entity
            local_entity: Local entity to condition on
            
        Returns:
            Pseudo-probability in [ε, 1]
        """
        if entity not in self.entity_embeddings or local_entity not in self.entity_embeddings:
            return self.epsilon
        
        vec_entity = self.entity_embeddings[entity]
        vec_local = self.entity_embeddings[local_entity]
        
        cosine_sim = self._cosine_similarity(vec_entity, vec_local)
        
        # Map [-1, 1] to [0, 1] conservatively
        mapped_prob = self.tau * (cosine_sim + 1) / 2
        
        return max(self.epsilon, mapped_prob)
    
    def _select_factor(
        self,
        entity: str,
        local_entity: str,
    ) -> float:
        """
        Select between smoothed co-occurrence or embedding fallback.
        
        Formula:
            p_used(e_j|e) = P̂_co(e_j|e)  if K[e,e_j] ≥ threshold
                           P̂_emb(e_j|e)  otherwise
        
        Args:
            entity: Candidate external entity
            local_entity: Local entity to condition on
            
        Returns:
            Selected conditional probability
        """
        cooccurrence_count = self.entity_cooccurrence.get(entity, Counter()).get(local_entity, 0)
        
        if cooccurrence_count >= self.threshold:
            return self._compute_smoothed_cooccurrence_prob(entity, local_entity)
        else:
            return self._compute_embedding_fallback_prob(entity, local_entity)
    
    def _compute_log_prior(self, entity: str) -> float:
        """
        Compute log prior probability.
        
        Formula: ℓ_prior = log(count(e) / total_counts)
        
        Args:
            entity: Candidate entity
            
        Returns:
            Log probability (can be negative)
        """
        entity_count = self.entity_counter.get(entity, 0)
        
        if entity_count == 0:
            return math.log(self.epsilon)
        
        prob = entity_count / self.total_counts
        return math.log(max(self.epsilon, prob))
    
    def _compute_log_likelihood(self, entity: str, local_entities: List[str]) -> float:
        """
        Compute log likelihood over local entities.
        
        Formula: ℓ_like = Σ_{e_j ∈ E_local} log(p_used(e_j|e))
        
        Args:
            entity: Candidate external entity
            local_entities: List of local entities
            
        Returns:
            Sum of log probabilities
        """
        log_likelihood = 0.0
        
        for local_entity in local_entities:
            prob = self._select_factor(entity, local_entity)
            log_likelihood += math.log(max(self.epsilon, prob))
        
        return log_likelihood
    
    def _compute_log_embedding_relevance(
        self,
        entity: str,
        local_embedding: np.ndarray,
    ) -> float:
        """
        Compute log embedding-based relevance for the full local set.
        
        Formula: ℓ_emb = log(max(ε, (cos(v_local, v(e)) + 1) / 2))
        
        Args:
            entity: Candidate external entity
            local_embedding: Mean or aggregated embedding of local entities
            
        Returns:
            Log pseudo-probability
        """
        if entity not in self.entity_embeddings or len(local_embedding) == 0:
            return math.log(self.epsilon)
        
        vec_entity = self.entity_embeddings[entity]
        cosine_sim = self._cosine_similarity(vec_entity, local_embedding)
        
        mapped_prob = (cosine_sim + 1) / 2
        mapped_prob = max(self.epsilon, mapped_prob)
        
        return math.log(mapped_prob)
    
    def score_entity(
        self,
        entity: str,
        local_entities: List[str],
        local_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute final hybrid score for a candidate entity.
        
        Formula: score(e) = (1-β) × (ℓ_prior + ℓ_like) + β × ℓ_emb
        
        Args:
            entity: Candidate external entity
            local_entities: List of local entities from the core paper
            local_embedding: Optional pre-computed local set embedding
            
        Returns:
            Hybrid score (higher is better)
        """
        # Compute co-occurrence signal
        log_prior = self._compute_log_prior(entity)
        log_likelihood = self._compute_log_likelihood(entity, local_entities)
        co_occurrence_signal = log_prior + log_likelihood
        
        # Compute embedding signal
        if local_embedding is None or len(local_embedding) == 0:
            embedding_signal = math.log(self.epsilon)
        else:
            embedding_signal = self._compute_log_embedding_relevance(entity, local_embedding)
        
        # Mix signals
        score = (1 - self.beta) * co_occurrence_signal + self.beta * embedding_signal
        
        return score
    
    def score_entities_batch(
        self,
        candidate_entities: List[str],
        local_entities: List[str],
        local_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """
        Score multiple candidate entities efficiently.
        
        Args:
            candidate_entities: List of candidate entities to score
            local_entities: List of local entities from core paper
            local_embedding: Optional pre-computed local set embedding
            
        Returns:
            List of (entity, score) tuples sorted by score (descending)
        """
        scores = []
        
        for entity in candidate_entities:
            score = self.score_entity(entity, local_entities, local_embedding)
            scores.append((entity, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def get_top_k_entities(
        self,
        candidate_entities: List[str],
        local_entities: List[str],
        local_embedding: Optional[np.ndarray] = None,
        k: int = 30,
    ) -> List[str]:
        """
        Get top-k entities by hybrid score.
        
        Args:
            candidate_entities: List of candidate entities
            local_entities: List of local entities
            local_embedding: Optional pre-computed local set embedding
            k: Number of top entities to return
            
        Returns:
            Top-k entities sorted by score (descending)
        """
        scored_entities = self.score_entities_batch(
            candidate_entities, local_entities, local_embedding
        )
        
        return [entity for entity, _ in scored_entities[:k]]
    
    def get_top_k_entities_with_scores(
        self,
        candidate_entities: List[str],
        local_entities: List[str],
        local_embedding: Optional[np.ndarray] = None,
        k: int = 30,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k entities with their scores.
        
        Args:
            candidate_entities: List of candidate entities
            local_entities: List of local entities
            local_embedding: Optional pre-computed local set embedding
            k: Number of top entities to return
            
        Returns:
            Top-k (entity, score) tuples sorted by score (descending)
        """
        scored_entities = self.score_entities_batch(
            candidate_entities, local_entities, local_embedding
        )
        
        return scored_entities[:k]