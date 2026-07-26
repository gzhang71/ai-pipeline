"""Command line entry point: `python -m harness <command>`.

    python -m harness list prompts
    python -m harness list tasks
    python -m harness run --prompt triage.v2 --out runs/v2.jsonl
    python -m harness diff runs/v1.jsonl runs/v2.jsonl
    python -m harness demo

`diff` exits 1 when there is at least one regression, so it drops into CI
without a wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from common.client import MODEL, has_credentials

from . import fakes
from .diff import diff_files, format_json, format_report
from .judge import DEFAULT_JUDGE_ID, Judge, JudgeCache
from .model import AnthropicClient, ModelClient
from .prompts import Prompt, PromptError, default_prompt_dir, load_prompts
from .runner import RunError, default_run_path, run
from .tasks import Task, TaskError, default_task_dir, load_tasks

DEFAULT_JUDGE_CACHE = Path("runs/.judge-cache.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="Prompt regression harness: versioned prompts, a fixed task set, "
        "and a diff that tells you whether an edit helped or hurt.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--prompts", type=Path, default=default_prompt_dir(),
                        help="directory of *.prompt.md files")
    common.add_argument("--tasks", type=Path, default=default_task_dir(),
                        help="directory of *.toml task files")

    p_list = sub.add_parser("list", parents=[common], help="list prompts or tasks")
    p_list.add_argument("what", choices=["prompts", "tasks"])

    p_run = sub.add_parser("run", parents=[common], help="run one prompt over the task set")
    p_run.add_argument("--prompt", required=True, help="prompt id, e.g. triage.v2")
    p_run.add_argument("--out", type=Path, default=None, help="run record path (JSONL)")
    p_run.add_argument("--concurrency", type=int, default=4)
    p_run.add_argument("--tag", action="append", default=[],
                       help="only run tasks carrying this tag (repeatable)")
    p_run.add_argument("--task", action="append", default=[],
                       help="only run this task id (repeatable)")
    p_run.add_argument("--judge-prompt", default=DEFAULT_JUDGE_ID)
    p_run.add_argument("--judge-samples", type=int, default=1,
                       help="majority-vote over N judge samples (default 1)")
    p_run.add_argument("--judge-cache", type=Path, default=DEFAULT_JUDGE_CACHE)
    p_run.add_argument("--no-judge-cache", action="store_true",
                       help="ask the judge fresh every time (measures judge variance)")
    p_run.add_argument("--no-judge", action="store_true",
                       help="skip judge assertions; they record as unevaluated failures")
    p_run.add_argument("--no-resume", action="store_true",
                       help="refuse to append to an existing run file")
    p_run.add_argument("--fake-responses", type=Path, default=None,
                       help="run offline against a scripted response fixture")

    p_diff = sub.add_parser("diff", help="compare two run records")
    p_diff.add_argument("before", type=Path)
    p_diff.add_argument("after", type=Path)
    p_diff.add_argument("--json", action="store_true", dest="as_json")
    p_diff.add_argument("--verbose", action="store_true")
    p_diff.add_argument("--no-fail-on-regression", action="store_true",
                        help="always exit 0, even when tasks regressed")

    p_demo = sub.add_parser("demo", parents=[common],
                            help="run both bundled prompt versions offline and diff them")
    p_demo.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    p_demo.add_argument("--fixture", type=Path, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "diff":
            return _cmd_diff(args)
        if args.command == "demo":
            return _cmd_demo(args)
    except (PromptError, TaskError, RunError, fakes.FixtureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _cmd_list(args: argparse.Namespace) -> int:
    if args.what == "prompts":
        prompts = load_prompts(args.prompts)
        if not prompts:
            print(f"no prompts in {args.prompts}")
            return 0
        width = max(len(p.id) for p in prompts.values())
        for prompt in prompts.values():
            print(f"{prompt.id:<{width}}  {prompt.short_hash}  {prompt.description}")
        return 0

    tasks = load_tasks(args.tasks)
    width = max(len(t.id) for t in tasks.values())
    for task in tasks.values():
        kinds = ",".join(sorted({str(a.get("type")) for a in task.assertions}))
        tags = f" [{','.join(task.tags)}]" if task.tags else ""
        first_line = task.description.strip().splitlines()[0] if task.description else ""
        print(f"{task.id:<{width}}  {task.short_hash}  {len(task.assertions)}a  {kinds}{tags}")
        if first_line:
            print(f"{'':<{width}}  {first_line}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    prompts = load_prompts(args.prompts)
    if args.prompt not in prompts:
        print(
            f"error: unknown prompt {args.prompt!r}; known: {', '.join(sorted(prompts))}",
            file=sys.stderr,
        )
        return 2
    prompt = prompts[args.prompt]
    tasks = _select_tasks(load_tasks(args.tasks), args.task, args.tag)
    if not tasks:
        print("error: task filters matched nothing", file=sys.stderr)
        return 2

    fixture = fakes.load_fixture(args.fake_responses) if args.fake_responses else None
    client: ModelClient
    if fixture is not None:
        client = fakes.task_client(fixture)
    else:
        if not has_credentials():
            print(
                "error: no Anthropic credentials. Run offline with "
                "`--fake-responses harness/data/demo_responses.json`, or set "
                "ANTHROPIC_API_KEY.",
                file=sys.stderr,
            )
            return 2
        client = AnthropicClient()

    judge = None
    if not args.no_judge and any(t.has_judge for t in tasks):
        judge = _build_judge(args, prompts, fixture)

    out_path = args.out or default_run_path(prompt)
    summary = run(
        prompt=prompt,
        tasks=tasks,
        client=client,
        out_path=out_path,
        judge=judge,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        model=MODEL,
    )

    print(f"prompt   {prompt.id}@{prompt.short_hash}")
    print(f"run      {summary.run_id}  ->  {summary.path}")
    if summary.skipped:
        print(f"resumed  {len(summary.skipped)} task(s) already recorded, skipped")
    print(f"pass     {summary.passed}/{summary.total} ({summary.pass_rate:.0%})")
    for result in summary.results:
        if result.passed:
            continue
        reasons = "; ".join(
            f"{a.id}: {a.detail or 'failed'}" for a in result.failed_assertions
        )
        print(f"  FAIL {result.task_id}  {reasons[:200]}")
    usage = summary.usage()
    print(
        f"tokens   prompt={usage.get('total_prompt_tokens', 0):,} "
        f"output={usage.get('output_tokens', 0):,}"
    )
    if judge is not None and judge.cache is not None:
        print(f"judge    cache hits={judge.cache.hits} misses={judge.cache.misses}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    report = diff_files(args.before, args.after)
    print(format_json(report) if args.as_json else format_report(report, verbose=args.verbose))
    if report.regressions and not args.no_fail_on_regression:
        return 1
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run both bundled prompt versions offline and print the diff."""
    fixture = fakes.load_fixture(args.fixture)
    prompts = load_prompts(args.prompts)
    tasks = list(load_tasks(args.tasks).values())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_prompt = prompts[DEFAULT_JUDGE_ID]
    cache = JudgeCache(out_dir / "judge-cache.json")
    paths = []
    for prompt_id in ("triage.v1", "triage.v2"):
        prompt = prompts[prompt_id]
        path = out_dir / f"{prompt_id}.{prompt.short_hash}.jsonl"
        if path.exists():
            path.unlink()
        judge = Judge(client=fakes.judge_client(fixture), prompt=judge_prompt, cache=cache)
        summary = run(
            prompt=prompt,
            tasks=tasks,
            client=fakes.task_client(fixture),
            out_path=path,
            judge=judge,
            concurrency=4,
        )
        print(f"{prompt.id}@{prompt.short_hash}: {summary.passed}/{summary.total} passed")
        paths.append(path)

    print(f"judge cache: {cache.hits} hit(s), {cache.misses} miss(es), {len(cache)} entries")
    print()
    print(format_report(diff_files(paths[0], paths[1])))
    print()
    print(f"(run records written to {out_dir})")
    return 0


def _build_judge(
    args: argparse.Namespace, prompts: dict[str, Prompt], fixture: dict | None
) -> Judge:
    if args.judge_prompt not in prompts:
        raise PromptError(
            f"unknown judge prompt {args.judge_prompt!r}; known: {', '.join(sorted(prompts))}"
        )
    if fixture is not None:
        judge_model_client: ModelClient = fakes.judge_client(fixture)
    else:
        judge_model_client = AnthropicClient()
    cache = None if args.no_judge_cache else JudgeCache(args.judge_cache)
    return Judge(
        client=judge_model_client,
        prompt=prompts[args.judge_prompt],
        cache=cache,
        samples=args.judge_samples,
    )


def _select_tasks(
    tasks: dict[str, Task], ids: Sequence[str], tags: Sequence[str]
) -> list[Task]:
    selected = list(tasks.values())
    if ids:
        wanted = set(ids)
        selected = [t for t in selected if t.id in wanted]
    if tags:
        wanted_tags = set(tags)
        selected = [t for t in selected if wanted_tags & set(t.tags)]
    return selected
