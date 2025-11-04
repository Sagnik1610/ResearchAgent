import os
import time
import logging
import threading
from itertools import chain

import requests
import torch
from sentence_transformers.util import cos_sim

# Module-level rate limiting state (1 request per second)
_last_request_time = 0.0
_rate_lock = threading.Lock()
_MIN_INTERVAL = 1.0  # seconds (Semantic Scholar: 1 request per second)


def _ensure_rate_limit():
    """Ensure at least _MIN_INTERVAL seconds between requests across the module."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _build_headers(user_headers: dict | None = None) -> dict:
    """Return headers merged with the Semantic Scholar API key from environment.

    The API key must be provided in the environment variable
    `SEMANTIC_SCHOLAR_API_KEY` and sent as the `x-api-key` header.
    """
    headers = (user_headers.copy() if user_headers else {})
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers.setdefault("x-api-key", api_key)
    else:
        logging.warning(
            "Environment variable SEMANTIC_SCHOLAR_API_KEY not set. Requests may fail."
        )
    return headers


def _request_with_retries(method, url, headers=None, params=None, json=None, timeout=60, return_type=None, max_attempts=3):
    """Internal helper to call requests.get/post with rate-limiting and retries.

    - Respects the 1 request/sec global rate limit.
    - Retries on network errors and 429 status codes with exponential backoff.
    - Returns `return_type` on failure (if provided) or `None`.
    """
    headers = _build_headers(headers)

    for attempt in range(1, max_attempts + 1):
        try:
            _ensure_rate_limit()

            if method == "GET":
                resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method == "POST":
                resp = requests.post(url, headers=headers, params=params, json=json, timeout=timeout)
            else:
                raise ValueError("Unsupported HTTP method")

            # If rate-limited by server, back off and retry
            if resp.status_code == 429:
                backoff = 2 ** (attempt - 1)
                logging.warning(f"Received 429 Too Many Requests. Backing off {backoff}s (attempt {attempt}).")
                time.sleep(backoff)
                continue

            resp.raise_for_status()

            # Successful response
            return resp.json()

        except requests.RequestException as e:
            # Network or HTTP error; retry with backoff
            backoff = 2 ** (attempt - 1)
            logging.warning(f"Request error on {url}: {e} (attempt {attempt}/{max_attempts}). Backing off {backoff}s.")
            time.sleep(backoff)
            continue

    # Exhausted attempts
    logging.error(f"Failed to get a successful response from {url} after {max_attempts} attempts.")
    return return_type


def get_request(url, headers=None, params=None, timeout=60, return_type=None):
    return _request_with_retries("GET", url, headers=headers, params=params, timeout=timeout, return_type=return_type)


def post_request(url, headers=None, params=None, json=None, timeout=60, return_type=None):
    return _request_with_retries("POST", url, headers=headers, params=params, json=json, timeout=timeout, return_type=return_type)


def batched(items: list, batch_size: int):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def flatten_list(items: list):
    return (
        list(chain.from_iterable(items))
        if items and items[0] is not None else []
    )


def get_papers(
    ids: list,
    fields: list = ['corpusId', 'title', 'abstract', 'year', 'publicationDate', 'referenceCount', 'citationCount', 'embedding.specter_v2', 'status', 'license'],
    batch_size: int = 100
):
    response = [
        post_request(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            headers=None,
            params={'fields': ','.join(fields)},
            json={'ids': ids_batched},
            return_type=[]
        )
        for ids_batched in batched(ids, batch_size)
    ]
    logging.info(f"S2 API papers response: {response}")
    return flatten_list(response)


def filter_papers(
    papers: list,
    categories: list = ['title', 'abstract', 'embedding']
):
    return [
        paper for paper in papers
        if isinstance(paper, dict)
        and all(paper.get(category) is not None for category in categories)
    ]


def get_relevant_references(
    paper: dict,
    fields: list = ['paperId', 'corpusId', 'isInfluential', 'title', 'abstract', 'year', 'publicationDate'],
    batch_size: int = 1000, top_k: int = 10
):
    # Fetch references in pages, respecting the rate limit via get_request
    references = [
        get_request(
            f'https://api.semanticscholar.org/graph/v1/paper/{paper["paperId"]}/references',
            headers=None,
            params={'fields': ','.join(fields), 'offset': index*batch_size, 'limit': batch_size},
            return_type={'data': []}
        )['data']
        for index in range(0, paper.get('referenceCount', 0) // batch_size + 1)
    ]

    references = [
        reference for reference in flatten_list(references)
        if all(
            reference.get('citedPaper') is not None and
            reference['citedPaper'].get(category) is not None
            for category in ['paperId', 'title', 'abstract']
        )
    ]

    reference_ids = [reference['citedPaper']['paperId'] for reference in references]

    reference2embedding = get_paper2embedding(reference_ids)

    if len(reference2embedding) == 0:
        return []

    # Compute similarity between the target paper embedding and reference embeddings
    try:
        paper_embedding = paper['embedding']['vector']
        ref_embeddings = list(reference2embedding.values())
        similar_reference_indices = torch.topk(
            cos_sim(paper_embedding, ref_embeddings),
            k=min(top_k, len(ref_embeddings))
        ).indices[0].tolist()
        similar_reference_ids = [list(reference2embedding.keys())[index] for index in similar_reference_indices]
    except Exception as e:
        logging.warning(f"Error computing embeddings similarity: {e}")
        return []

    return [
        reference['citedPaper'] for reference in references
        if reference['citedPaper']['paperId'] in similar_reference_ids
    ]


def get_paper2embedding(ids: list):
    return {
        paper['paperId']: paper['embedding']['vector']
        for paper in get_papers(ids, fields=['embedding.specter_v2'])
        if (
            isinstance(paper, dict) and
            'embedding' in paper.keys() and
            paper['embedding'] is not None
        )
    }
