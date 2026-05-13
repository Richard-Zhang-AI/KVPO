import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_switch_frame_indices(raw_value: Any) -> List[int]:
    return [int(x) for x in str(raw_value).split(",") if str(x).strip()]


def resolve_switch_frame_indices(raw_value: Any, num_segments: int) -> List[int]:
    switch_indices = parse_switch_frame_indices(raw_value)
    expected = max(int(num_segments) - 1, 0)
    if len(switch_indices) < expected:
        raise ValueError(
            f"Expected at least {expected} switch_frame_indices for {num_segments} prompt segments, "
            f"but got {len(switch_indices)}"
        )
    return switch_indices[:expected]


def _normalize_prompt_list(
    prompts: List[Any],
    *,
    min_segments: Optional[int],
    max_segments: Optional[int],
) -> Optional[List[str]]:
    normalized = [str(p).strip() for p in prompts if str(p).strip()]
    if max_segments is not None:
        normalized = normalized[: int(max_segments)]
    if min_segments is not None and len(normalized) < int(min_segments):
        return None
    if not normalized:
        return None
    return normalized


def _extract_bracket_fields(text: str) -> List[str]:
    fields: List[str] = []
    depth = 0
    start = -1

    for idx, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif ch == "]":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                fields.append(text[start:idx].strip())
                start = -1

    return fields


def _load_jsonl_prompt_sequences(
    path: Path,
    *,
    field: str,
    max_n: int,
    min_segments: Optional[int],
    max_segments: Optional[int],
) -> List[Dict[str, List[str]]]:
    items: List[Dict[str, List[str]]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts = row.get(field)
            if not isinstance(prompts, list):
                continue
            normalized = _normalize_prompt_list(
                prompts,
                min_segments=min_segments,
                max_segments=max_segments,
            )
            if normalized is None:
                continue
            items.append({field: normalized})
            if max_n > 0 and len(items) >= max_n:
                break
    return items


def _load_formatted_prompt_sequences(
    path: Path,
    *,
    field: str,
    max_n: int,
    min_segments: Optional[int],
    max_segments: Optional[int],
) -> List[Dict[str, List[str]]]:
    items: List[Dict[str, List[str]]] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            head, sep, tail = line.partition(",")
            body = tail if sep else line

            if head.isdigit() and int(head) == 0 and not body:
                continue

            prompts = _extract_bracket_fields(body)
            normalized = _normalize_prompt_list(
                prompts,
                min_segments=min_segments,
                max_segments=max_segments,
            )
            if normalized is None:
                continue

            items.append({field: normalized})
            if max_n > 0 and len(items) >= max_n:
                break
    return items


def load_prompt_sequences(
    path: str,
    *,
    field: str = "prompts",
    max_n: int = -1,
    min_segments: Optional[int] = None,
    max_segments: Optional[int] = None,
) -> List[Dict[str, List[str]]]:
    prompt_path = Path(path)
    suffix = prompt_path.suffix.lower()

    if suffix == ".jsonl":
        return _load_jsonl_prompt_sequences(
            prompt_path,
            field=field,
            max_n=max_n,
            min_segments=min_segments,
            max_segments=max_segments,
        )

    return _load_formatted_prompt_sequences(
        prompt_path,
        field=field,
        max_n=max_n,
        min_segments=min_segments,
        max_segments=max_segments,
    )
