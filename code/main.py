# code/main.py
import argparse
import os
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

from utils import data_io, s2
from knowledge.store import KnowledgeStore
from models.openai import OpenAIClient
from pipelines.research_pipeline import ResearchPipeline

def load_entity_embeddings(embedding_path: str, embedding_dim: int) -> dict:
    """Load pre-computed entity embeddings from file."""
    embeddings = {}
    
    try:
        with open(embedding_path, 'r') as f:
            line_count = 0
            for line in f:
                item = json.loads(line.strip())
                entity = item.get('entity')
                embedding = item.get('embedding')
                
                if entity and embedding:
                    embeddings[entity] = np.array(embedding, dtype=np.float32)
                    line_count += 1
        
        print(f"Loaded embeddings for {len(embeddings)} entities from {embedding_path}")
    except FileNotFoundError:
        print(f"WARNING: Embedding file {embedding_path} not found. Proceeding without embeddings.")
    except Exception as e:
        print(f"ERROR loading embeddings: {e}")
    
    return embeddings

def fetch_resources(paper: dict, knowledge_store: KnowledgeStore):
    references = s2.get_relevant_references(paper)
    entities = knowledge_store.get_relevant_entities(
        [paper['corpusId']] + [reference['corpusId'] for reference in references]
    )
    return references, entities

def get_argument_parser():
    """Enhanced argument parser with hybrid scoring options."""
    parser = argparse.ArgumentParser(description="ResearchAgent: Iterative Research Idea Generation")
    
    # Original arguments (keep these)
    parser.add_argument('--data-path', '-d', type=str, default='./data/papers.jsonl',
                        help='Path to input papers JSONL file')
    parser.add_argument('--knowledge-path', '-k', type=str, default='./data/knowledge.jsonl',
                        help='Path to knowledge base JSONL file')
    parser.add_argument('--model-name', '-m', type=str, default='DeepSeek-R1-Llama-70B',
                        help='LLM model name to use')
    
    # Hybrid Scoring Configuration
    parser.add_argument('--use-hybrid-scoring', type=bool, default=True,
                        help='Enable hybrid scoring (default: True)')
    
    # Hyperparameters for hybrid scoring
    parser.add_argument('--hybrid-alpha', type=float, default=0.01,
                        help='Laplace smoothing constant (0.01 to 1)')
    
    parser.add_argument('--hybrid-tau', type=float, default=0.5,
                        help='Embedding-to-probability scale (0.4 to 0.8)')
    
    parser.add_argument('--hybrid-epsilon', type=float, default=1e-9,
                        help='Probability floor to avoid log(0)')
    
    parser.add_argument('--hybrid-beta', type=float, default=0.3,
                        help='Mix weight: 0=co-occurrence only, 1=embedding only (0 to 1)')
    
    parser.add_argument('--hybrid-threshold', type=int, default=1,
                        help='Minimum co-occurrence count to use smoothed probability')
    
    parser.add_argument('--hybrid-row-normalizer', type=int, default=None,
                        help='Row normalizer for smoothing (default: total entity count)')
    
    parser.add_argument('--entity-embedding-path', type=str, default=None,
                        help='Optional path to pre-computed entity embeddings (JSONL)')
    
    parser.add_argument('--embedding-dim', type=int, default=768,
                        help='Dimension of entity embeddings')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser

def run(
    paper_ids: list,
    knowledge_store: KnowledgeStore,
    openai_client: OpenAIClient
):
    papers = s2.filter_papers(
        s2.get_papers(paper_ids),
        categories=['title', 'abstract', 'embedding']
    )

    results = []

    for paper in tqdm(papers):
        context = {'paper': {key: paper.get(key) for key in ('title', 'abstract')}}
        
        references, entities = fetch_resources(paper, knowledge_store)
        context.update(references=references, entities=entities)

        research_pipeline = ResearchPipeline(api_client=openai_client)
        context = research_pipeline.run(context)

        results.append(context)
        data_io.save_result('./results/ideas.jsonl', context)

    return results

def initialize_knowledge_store(args) -> KnowledgeStore:
    """
    Initialize knowledge store with hybrid scoring.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        KnowledgeStore instance
    """
    # Load entity embeddings if provided
    entity_embeddings = {}
    if args.entity_embedding_path:
        entity_embeddings = load_entity_embeddings(
            args.entity_embedding_path,
            args.embedding_dim
        )
    
    # Initialize knowledge store with hybrid scoring
    knowledge_store = KnowledgeStore(
        file_path=args.knowledge_path,
        entity_embeddings=entity_embeddings,
        use_hybrid_scoring=args.use_hybrid_scoring,
        alpha=args.hybrid_alpha,
        tau=args.hybrid_tau,
        epsilon=args.hybrid_epsilon,
        beta=args.hybrid_beta,
        threshold=args.hybrid_threshold,
        r=args.hybrid_row_normalizer,
    )
    
    # Log configuration
    print("\n=== Knowledge Store Configuration ===")
    config = knowledge_store.get_scorer_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*40 + "\n")
    
    return knowledge_store

if __name__ == "__main__":
    # Use the enhanced argument parser
    argparser = get_argument_parser()
    args = argparser.parse_args()

    # quick env checks to surface missing keys early
    if not os.getenv("KRUTRIM_API_KEY"):
        print("WARNING: KRUTRIM_API_KEY is not set. Export it before running, e.g.:")
        print('  export KRUTRIM_API_KEY="your_krutrim_key_here"')
    if not os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        print("WARNING: SEMANTIC_SCHOLAR_API_KEY is not set. Export it before running, e.g.:")
        print('  export SEMANTIC_SCHOLAR_API_KEY="your_semanticscholar_key_here"')

    paper_ids = data_io.load_paper_ids(args.data_path)
    knowledge_store = initialize_knowledge_store(args)

    openai_client = OpenAIClient(model=args.model_name)

    results = run(paper_ids, knowledge_store, openai_client)