# code/models/openai.py
import os
import json
import logging
from typing import Any, Optional

# Try import krutrim sdk
try:
    from krutrim_cloud import KrutrimCloud
except Exception as exc:
    KrutrimCloud = None
    logging.warning("krutrim_cloud not available: %s", exc)

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class OpenAIClient:
    """
    Wrapper that uses KrutrimCloud under the hood and returns a plain string
    (the assistant's content). Handles streaming 'data: {...}' chunks and
    iterable streaming objects from the SDK.
    """

    def __init__(self, model: str = "DeepSeek-R1-Llama-70B"):
        self.model = model
        self._client = None
        self._init_client()

    def _init_client(self):
        if KrutrimCloud is None:
            self._client = None
            return
        api_key = os.getenv("KRUTRIM_API_KEY")
        try:
            # Try to pass api_key if SDK accepts it; otherwise fallback
            try:
                self._client = KrutrimCloud(api_key=api_key) if api_key else KrutrimCloud()
            except TypeError:
                self._client = KrutrimCloud()
        except Exception as exc:
            logging.exception("Failed to initialize KrutrimCloud: %s", exc)
            self._client = None

    def _extract_from_choice_obj(self, choice: Any) -> Optional[str]:
        # choice may be dict or object
        try:
            if isinstance(choice, dict):
                # streaming delta style
                if "delta" in choice:
                    delta = choice["delta"]
                    if isinstance(delta, dict) and "content" in delta:
                        return str(delta["content"])
                # message style
                if "message" in choice:
                    msg = choice["message"]
                    if isinstance(msg, dict) and "content" in msg:
                        return str(msg["content"])
                # fallback to text
                if "text" in choice:
                    return str(choice["text"])
            else:
                # object-like access
                if hasattr(choice, "delta") and getattr(choice, "delta") is not None:
                    d = getattr(choice, "delta")
                    if isinstance(d, dict) and "content" in d:
                        return str(d["content"])
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return str(choice.message.content)
                if hasattr(choice, "text"):
                    return str(choice.text)
        except Exception:
            return None
        return None

    def _parse_data_line(self, line: str) -> Optional[str]:
        """
        Parse a single 'data: {...}' line. Returns extracted content string or None.
        """
        try:
            payload = line[len("data:"):].strip() if line.startswith("data:") else line.strip()
            if not payload or payload == "[DONE]":
                return None
            obj = json.loads(payload)
            # obj may contain choices list
            if isinstance(obj, dict) and "choices" in obj and obj["choices"]:
                return self._extract_from_choice_obj(obj["choices"][0])
            # fallback: maybe top-level has content field
            if isinstance(obj, dict) and "content" in obj:
                return str(obj["content"])
        except Exception:
            # not JSON or unexpected shape
            return None
        return None

    def _parse_stream_string(self, stream_text: str) -> str:
        """
        Parse a long string that contains multiple 'data: {...}' lines (as in your output).
        Concatenate all extracted text fragments and return the full assistant text.
        """
        pieces = []
        for raw_line in stream_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Many SDKs prefix streaming chunks with 'data: ' or 'data:'.
            if line.startswith("data:"):
                txt = self._parse_data_line(line)
                if txt:
                    pieces.append(txt)
            else:
                # If the line looks like JSON, try parsing
                if line.startswith("{") or line.startswith("["):
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "choices" in obj and obj["choices"]:
                            txt = self._extract_from_choice_obj(obj["choices"][0])
                            if txt:
                                pieces.append(txt)
                    except Exception:
                        continue
                else:
                    # plain text line — consider appending
                    pieces.append(line)
        return "".join(pieces).strip()

    def _collect_from_iterable(self, iterable) -> str:
        """
        Iterate through SDK streaming iterator. Each chunk may be dict, object, or string.
        Concatenate pieces and return final text.
        """
        pieces = []
        for chunk in iterable:
            # chunk is commonly either string like "data: {...}" or dict-like
            if isinstance(chunk, str):
                txt = self._parse_stream_string(chunk)
                if txt:
                    pieces.append(txt)
                continue
            if isinstance(chunk, dict):
                # chunk may be a single JSON chunk
                try:
                    if "choices" in chunk and chunk["choices"]:
                        txt = self._extract_from_choice_obj(chunk["choices"][0])
                        if txt:
                            pieces.append(txt)
                        continue
                except Exception:
                    pass
                # fallback: try parsing as string
                try:
                    s = json.dumps(chunk)
                    pieces.append(self._parse_stream_string(s))
                except Exception:
                    continue
            else:
                # object-like chunk: try to access attributes
                try:
                    if hasattr(chunk, "choices"):
                        choice = chunk.choices[0]
                        txt = self._extract_from_choice_obj(choice)
                        if txt:
                            pieces.append(txt)
                        continue
                except Exception:
                    pass
                # fallback: convert to str and parse
                try:
                    pieces.append(self._parse_stream_string(str(chunk)))
                except Exception:
                    continue
        return "".join(pieces).strip()

    def call(self, messages, max_retries: int = 3) -> Optional[str]:
        """
        Send messages to Krutrim model and return the assistant's combined text.
        Tries streaming first for faster initial response handling.
        """
        if self._client is None:
            self._init_client()
            if self._client is None:
                logging.error("Kutrim client unavailable.")
                return None

        attempt = 0
        while attempt < max_retries:
            try:
                # Request streaming for fastest response; SDK may return iterable or streaming text
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True
                )

                # If resp is a string containing many 'data:' lines (as in your debug), parse it
                if isinstance(resp, str):
                    return self._parse_stream_string(resp)

                # If resp is iterable/generator, iterate and collect text
                if hasattr(resp, "__iter__"):
                    collected = self._collect_from_iterable(resp)
                    if collected:
                        return collected
                    # If iterable produced no text, continue to next try

                # If resp is dict-like (non-stream), try to extract choices.message.content
                if isinstance(resp, dict):
                    try:
                        if "choices" in resp and len(resp["choices"]) > 0:
                            choice = resp["choices"][0]
                            txt = self._extract_from_choice_obj(choice)
                            if txt:
                                return txt
                    except Exception:
                        pass
                    # fallback: maybe resp contains a 'data' field as string
                    try:
                        as_text = json.dumps(resp)
                        parsed = self._parse_stream_string(as_text)
                        if parsed:
                            return parsed
                    except Exception:
                        pass

                # If resp is object-like (non-stream)
                try:
                    if hasattr(resp, "choices"):
                        choice = resp.choices[0]
                        txt = self._extract_from_choice_obj(choice)
                        if txt:
                            return txt
                except Exception:
                    pass

                # Nothing useful found — return None to indicate no textual content
                return None

            except Exception as exc:
                attempt += 1
                backoff = 2 ** attempt
                logging.warning("Kutrim API error (attempt %d/%d): %s — backing off %ds", attempt, max_retries, exc, backoff)
                import time
                time.sleep(backoff)
                continue

        logging.error("Kutrim API failed after %d attempts", max_retries)
        return None
