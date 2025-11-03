# code/utils/data_io.py
import os
import json
from typing import List, Optional

def load_jsonl(file_path: str) -> List[dict]:
    """Load a newline-delimited JSON file and return list of parsed objects.
    Returns empty list if file doesn't exist.
    """
    if not file_path or not os.path.isfile(file_path):
        return []
    items = []
    with open(file_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # If a line is not JSON, keep it as raw string inside an object for safety
                items.append(line)
    return items

def save_result(file_path: str, result: dict) -> None:
    """Append a JSON object as a line to file_path, creating directories if needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as outfile:
        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

def load_paper_ids(file_path: Optional[str] = None, num_papers: int = 300) -> List[str]:
    """
    Load paper identifiers from a file. Accepts multiple formats:
      - JSONL where each line is an object containing 'paperId' or 'arxivId' or 'corpusid' (common)
      - JSONL with plain string per line (an id)
      - If file_path is None or missing, returns a small default list (for dev)

    Returns a list of IDs formatted for Semantic Scholar Graph API:
      Prefer 'CorpusId:<id>' if only numeric id found,
      Else 'ARXIV:<arxivid>' if arXiv id found,
      Else returns raw string id as-is.
    """
    # default fallback IDs (small sample) -- keep minimal so tests run offline
    default_ids = ['215416146', '259095910', '259370631', '7720039', '12424239']

    if not file_path:
        paper_ids = default_ids
    else:
        raw = load_jsonl(file_path)
        paper_ids = []
        for entry in raw:
            if isinstance(entry, dict):
                # Accept many possible keys (case-insensitive)
                # try 'corpusid' (existing repo used lower-case), 'paperId', 'paper_id', 'arxivId'
                if 'corpusid' in entry:
                    paper_ids.append(f"CorpusId:{entry['corpusid']}")
                elif 'paperId' in entry:
                    paper_ids.append(entry['paperId'])
                elif 'paper_id' in entry:
                    paper_ids.append(entry['paper_id'])
                elif 'arxivId' in entry:
                    arx = str(entry['arxivId'])
                    paper_ids.append(arx if arx.upper().startswith('ARXIV:') else f"ARXIV:{arx}")
                elif 'arxiv_id' in entry:
                    arx = str(entry['arxiv_id'])
                    paper_ids.append(arx if arx.upper().startswith('ARXIV:') else f"ARXIV:{arx}")
                else:
                    # If the dict doesn't contain a known id, try to use 'id' or 'paper' or 'corpusId'
                    for k in ('id', 'paper', 'corpusId', 'corpus_id'):
                        if k in entry:
                            paper_ids.append(str(entry[k]))
                            break
            elif isinstance(entry, str):
                # Plain string line: treat it as an id
                s = entry.strip()
                if s:
                    # if numeric, assume corpus id; otherwise return raw (could be ARXIV:xxxx)
                    if s.isdigit():
                        paper_ids.append(f"CorpusId:{s}")
                    else:
                        paper_ids.append(s)
        # If file was empty or parse failed, fallback to defaults
        if not paper_ids:
            paper_ids = default_ids

    # enforce num_papers limit and return
    return paper_ids[:num_papers]
