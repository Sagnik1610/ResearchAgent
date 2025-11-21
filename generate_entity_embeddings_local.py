#!/usr/bin/env python3
"""
generate_entity_embeddings_local.py

Reads entities from your local knowledge.jsonl and writes entity_embeddings.jsonl
One JSON object per line: {"entity": "...", "embedding": [...]}

Configured for user path:
  /mnt/c/Users/sagni/Agentic_AI/ResearchAgent/data/knowledge.jsonl
"""

import json
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

# Default config - update here if you want different defaults
DEFAULT_INPUT = Path("/mnt/c/Users/sagni/Agentic_AI/ResearchAgent/data/knowledge.jsonl")
DEFAULT_OUTPUT = Path("/mnt/c/Users/sagni/Agentic_AI/ResearchAgent/data/entity_embeddings.jsonl")
DEFAULT_MODEL = "allenai/specter"          # change to a smaller model if you need speed (e.g. all-MiniLM-L6-v2)
DEFAULT_BATCH = 64

def load_entities_from_knowledge(path: Path):
    entities = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if "knowledge" in obj and isinstance(obj["knowledge"], dict):
                    for k in obj["knowledge"].keys():
                        if isinstance(k, str) and k.strip():
                            entities.add(k.strip())
                # also check common fields
                for key in ("entity", "entities", "name", "title"):
                    if key in obj:
                        v = obj[key]
                        if isinstance(v, str) and v.strip():
                            entities.add(v.strip())
                        elif isinstance(v, list):
                            for it in v:
                                if isinstance(it, str) and it.strip():
                                    entities.add(it.strip())
    return sorted(entities)

def read_already_processed(output_path: Path):
    processed = set()
    if not output_path.exists():
        return processed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and "entity" in rec:
                    processed.add(rec["entity"])
            except Exception:
                continue
    return processed

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_name = args.model
    batch_size = args.batch

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    print("Loading entities from:", input_path)
    entities = load_entities_from_knowledge(input_path)
    print(f"Found {len(entities)} unique entities.")

    processed = read_already_processed(output_path)
    print(f"Already processed (in output file): {len(processed)}")

    to_process = [e for e in entities if e not in processed]
    print(f"Remaining to process: {len(to_process)}")

    # load model
    print("Loading SentenceTransformer model:", model_name)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("Please install sentence-transformers (pip install -U sentence-transformers tqdm).", file=sys.stderr)
        raise

    model = SentenceTransformer(model_name)

    # determine device used by sentence-transformers
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    # model.encode accepts `device` argument
    with output_path.open("a", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
            batch = to_process[i:i+batch_size]
            # generate embeddings (returns numpy array)
            embeddings = model.encode(batch, batch_size=len(batch), show_progress_bar=False, device=device)
            for ent, emb in zip(batch, embeddings):
                # convert to list of floats (JSON serializable)
                try:
                    emb_list = emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
                except Exception:
                    # fallback: convert elementwise
                    emb_list = [float(x) for x in emb]
                out_f.write(json.dumps({"entity": ent, "embedding": emb_list}, ensure_ascii=False) + "\n")
            out_f.flush()

    print("Done. Output written to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity embeddings from knowledge.jsonl")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Path to knowledge.jsonl")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="Path to output JSONL")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--batch", "-b", default=DEFAULT_BATCH, type=int, help="Batch size")
    args = parser.parse_args()
    main(args)
