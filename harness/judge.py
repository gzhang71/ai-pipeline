"""The LLM judge -- tier 2 assertions.

Two design commitments:

1. **The judge is a prompt, versioned in the same system as the prompts under
   test.** It lives in `data/prompts/judge.*.prompt.md` and is hashed the same
   way. Every judged verdict records the judge hash. An unversioned judge
   silently invalidates every historical comparison you have: the same output
   scored by a quietly-edited judge produces a different verdict, and you would
   read that as a prompt regression.

2. **The verdict is structured, not prose.** The judge call sets
   `output_config.format` to a JSON schema, so the response is a parseable
   object rather than free text that needs a regex and a prayer.

Caching and sampling
--------------------
Judge calls are neither free nor deterministic, so they are cached on
`(judge_hash, prompt_hash, task_id, output_hash, criterion)`. All five matter:
change the judge, the prompt under test, the task, the sampled output, or the
criterion, and you get a fresh verdict. Re-running an unchanged prompt over an
unchanged task set therefore costs zero judge tokens.

The tradeoff is explicit: the cache freezes one sample of a stochastic process.
A judge that would answer "pass" 60% of the time is recorded as whatever it said
first, and that verdict is then reused forever. Two mitigations are available:

* `samples > 1` takes a best-of-N majority vote and stores the whole ballot, so
  the record shows the split (3-0 and 2-1 are not the same evidence).
* `Judge(cache=None)` disables caching entirely for a run where you want to
  measure judge variance rather than prompt variance.

The default is `samples=1` with caching on, because for a regression harness the
dominant cost of a flaky judge is *unattributable* diff noise, and a cached
verdict at least holds the judge constant across the two runs being compared.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .assertions import AssertionResult, extract_json, spec_id
from .hashing import hash_output, hash_text, short
from .model import ModelClient, ModelOutput, ModelRequest, sum_usage
from .prompts import Prompt
from .tasks import Task

DEFAULT_JUDGE_ID = "judge.rubric.v1"

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Verdict:
    verdict: str
    confidence: float = 0.0
    reasoning: str = ""
    error: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    cached: bool = False
    samples: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "error": self.error,
            "usage": dict(self.usage),
            "cached": self.cached,
            "samples": list(self.samples),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Verdict":
        return cls(
            verdict=data.get("verdict", "fail"),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            error=data.get("error"),
            usage=dict(data.get("usage") or {}),
            cached=bool(data.get("cached", False)),
            samples=tuple(data.get("samples") or ()),
        )


def cache_key(
    *, judge_hash: str, prompt_hash: str, task_id: str, output_hash: str, criterion: str
) -> str:
    return hash_text(
        "\x00".join([judge_hash, prompt_hash, task_id, output_hash, hash_text(criterion)])
    )


class JudgeCache:
    """A JSON-file cache of verdicts, safe for the runner's thread pool."""

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.is_file():
            try:
                self._entries = json.loads(self.path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._entries = {}

    def get(self, key: str) -> Verdict | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
        verdict = Verdict.from_dict(entry)
        # A cache hit costs no tokens; zero the usage so run totals stay honest.
        return Verdict(
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            error=verdict.error,
            usage={},
            cached=True,
            samples=verdict.samples,
        )

    def put(self, key: str, verdict: Verdict) -> None:
        with self._lock:
            self._entries[key] = verdict.to_dict()
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(json.dumps(self._entries, indent=2, sort_keys=True), "utf-8")
                tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._entries)


class Judge:
    """Runs judge assertions against a versioned judge prompt."""

    def __init__(
        self,
        *,
        client: ModelClient,
        prompt: Prompt,
        cache: JudgeCache | None = None,
        samples: int = 1,
        max_tokens: int = 1024,
        effort: str | None = None,
    ) -> None:
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self.client = client
        self.prompt = prompt
        self.cache = cache
        self.samples = samples
        self.max_tokens = max_tokens
        self.effort = effort or prompt.effort or "low"

    @property
    def hash(self) -> str:
        return self.prompt.hash

    def evaluate(
        self,
        spec: Mapping[str, Any],
        *,
        task: Task,
        prompt: Prompt,
        output: ModelOutput,
        index: int = 0,
    ) -> AssertionResult:
        criterion = str(spec["criterion"])
        ident = spec_id(spec, index)
        if output.error:
            return AssertionResult(
                ident, "judge", False, f"model error: {output.error}", meta={"judge": self.prompt.id}
            )

        out_hash = hash_output(output.text, output.tool_calls)
        key = cache_key(
            judge_hash=self.hash,
            prompt_hash=prompt.hash,
            task_id=task.id,
            output_hash=out_hash,
            criterion=criterion,
        )
        # `is not None`, not truthiness: JudgeCache defines __len__, so an empty
        # cache is falsy and `if self.cache` would silently disable caching.
        verdict = self.cache.get(key) if self.cache is not None else None
        if verdict is None:
            verdict = self._ask(criterion=criterion, task=task, output=output, samples=self.samples)
            if self.cache is not None and verdict.error is None:
                self.cache.put(key, verdict)

        meta = {
            "judge_prompt_id": self.prompt.id,
            "judge_hash": self.hash,
            "confidence": verdict.confidence,
            "reasoning": verdict.reasoning,
            "cached": verdict.cached,
            "cache_key": short(key, 16),
            "usage": dict(verdict.usage),
        }
        if len(verdict.samples) > 1:
            meta["samples"] = list(verdict.samples)
        if verdict.error:
            return AssertionResult(ident, "judge", False, verdict.error, meta=meta)
        detail = "" if verdict.passed else (verdict.reasoning or "judge returned fail")
        return AssertionResult(ident, "judge", verdict.passed, detail, meta=meta)

    def _ask(
        self, *, criterion: str, task: Task, output: ModelOutput, samples: int
    ) -> Verdict:
        ballots: list[Verdict] = []
        for _ in range(samples):
            ballots.append(self._ask_once(criterion=criterion, task=task, output=output))
        errored = [b for b in ballots if b.error]
        if errored and len(errored) == len(ballots):
            return errored[0]
        valid = [b for b in ballots if not b.error]
        tally = Counter(b.verdict for b in valid)
        winner = tally.most_common(1)[0][0]
        agreeing = [b for b in valid if b.verdict == winner]
        return Verdict(
            verdict=winner,
            confidence=sum(b.confidence for b in agreeing) / len(agreeing),
            reasoning=agreeing[0].reasoning,
            usage=sum_usage([b.usage for b in valid]),
            samples=tuple(b.verdict for b in valid),
        )

    def _ask_once(self, *, criterion: str, task: Task, output: ModelOutput) -> Verdict:
        request = ModelRequest(
            system=self.prompt.system,
            messages=(
                {"role": "user", "content": render_judge_input(criterion, task, output)},
            ),
            max_tokens=self.max_tokens,
            effort=self.effort,
            output_format={"type": "json_schema", "schema": VERDICT_SCHEMA},
            prompt_id=self.prompt.id,
            prompt_hash=self.prompt.hash,
            task_id=task.id,
            kind="judge",
        )
        result = self.client.complete(request)
        if result.error:
            return Verdict("fail", error=f"judge call failed: {result.error}")
        value, parse_error = extract_json(result.text)
        if parse_error or not isinstance(value, dict):
            return Verdict(
                "fail",
                error=f"judge returned unparseable verdict: {parse_error or 'not an object'}",
                usage=dict(result.usage),
            )
        verdict = str(value.get("verdict", "")).lower()
        if verdict not in {"pass", "fail"}:
            return Verdict(
                "fail",
                error=f"judge returned unknown verdict {verdict!r}",
                usage=dict(result.usage),
            )
        return Verdict(
            verdict=verdict,
            confidence=float(value.get("confidence", 0.0) or 0.0),
            reasoning=str(value.get("reasoning", "")),
            usage=dict(result.usage),
            samples=(verdict,),
        )


def render_judge_input(criterion: str, task: Task, output: ModelOutput) -> str:
    """The judge's user turn. Kept in code so the judge prompt file stays prose."""
    tool_calls = (
        json.dumps([c.to_dict() for c in output.tool_calls], indent=2)
        if output.tool_calls
        else "(none)"
    )
    return (
        "<criterion>\n"
        f"{criterion.strip()}\n"
        "</criterion>\n\n"
        "<task_input>\n"
        f"{task.input.strip()}\n"
        "</task_input>\n\n"
        "<candidate_output>\n"
        f"{output.text.strip()}\n"
        "</candidate_output>\n\n"
        "<tool_calls>\n"
        f"{tool_calls}\n"
        "</tool_calls>"
    )


def majority(verdicts: Sequence[str]) -> str:
    return Counter(verdicts).most_common(1)[0][0]
