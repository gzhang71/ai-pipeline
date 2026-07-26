"""Cumulative rendering of where the context window went.

Two renderers, both standard-library only:

``render_text``  -- plain-text stacked bars, one row per turn.
``render_html``  -- a self-contained page with inline SVG and CSS: stat tiles,
                    a stacked bar chart of prompt composition across turns, a
                    legend, a per-turn table view, and the reconciliation
                    residual for every turn.

Both show *growth over turns*, not a final snapshot: turn N's bar is the whole
prompt sent on turn N, broken down by segment kind, so the bars get taller as
the conversation accumulates and you can see which segment is doing the
growing.

Colour follows the dataviz reference palette's fixed categorical order (one
slot per segment kind, assigned by identity and never cycled), with neutral
grey reserved for the catch-all kinds. Series identity is never carried by
colour alone: there is always a legend, and the table view repeats every
number. Both light and dark modes are explicit steps from the same ramps.
"""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, Sequence

from .accuracy import AccuracyReport, report_text
from .schema import group_runs

#: Stack order == the API's render order (tools -> system -> messages), so the
#: bar reads bottom-up the way the prompt is actually assembled.
KIND_ORDER = (
    "framing",
    "tool_schemas",
    "system_prompt",
    "messages_total",
    "user_text",
    "assistant_text",
    "thinking",
    "tool_use",
    "tool_result",
    "other",
)

#: Categorical slots 1-8 in the palette's fixed order, plus neutrals for the
#: two catch-all kinds. (light, dark)
KIND_COLORS: dict[str, tuple[str, str]] = {
    "framing": ("#2a78d6", "#3987e5"),  # slot 1 blue
    "tool_schemas": ("#eb6834", "#d95926"),  # slot 2 orange
    "system_prompt": ("#1baf7a", "#199e70"),  # slot 3 aqua
    "user_text": ("#eda100", "#c98500"),  # slot 4 yellow
    "assistant_text": ("#e87ba4", "#d55181"),  # slot 5 magenta
    "thinking": ("#008300", "#008300"),  # slot 6 green
    "tool_use": ("#4a3aa7", "#9085e9"),  # slot 7 violet
    "tool_result": ("#e34948", "#e66767"),  # slot 8 red
    "messages_total": ("#898781", "#898781"),  # neutral
    "other": ("#c3c2b7", "#52514e"),  # neutral
}

KIND_LABELS = {
    "framing": "request framing",
    "tool_schemas": "tool schemas",
    "system_prompt": "system prompt",
    "user_text": "user text",
    "assistant_text": "assistant text",
    "thinking": "thinking",
    "tool_use": "tool_use blocks",
    "tool_result": "tool_result blocks",
    "messages_total": "messages (coarse)",
    "other": "other blocks",
}

#: Plain-text stand-ins, since a terminal has no colour channel we can rely on.
KIND_GLYPHS = {
    "framing": ".",
    "tool_schemas": "T",
    "system_prompt": "S",
    "user_text": "u",
    "assistant_text": "a",
    "thinking": "*",
    "tool_use": "x",
    "tool_result": "R",
    "messages_total": "m",
    "other": "?",
}


# --------------------------------------------------------------------------
# Shared shaping
# --------------------------------------------------------------------------


def _series(turns: Sequence[dict[str, Any]]) -> tuple[list[str], list[dict[str, int]]]:
    """Kinds present anywhere in the run (in stack order) and per-turn totals."""
    present = {kind for turn in turns for kind in turn["prompt_tokens"]["by_kind"]}
    kinds = [k for k in KIND_ORDER if k in present]
    kinds += sorted(k for k in present if k not in KIND_ORDER)
    rows = [
        {kind: int(turn["prompt_tokens"]["by_kind"].get(kind, 0)) for kind in kinds}
        for turn in turns
    ]
    return kinds, rows


def _nice_max(value: int) -> int:
    if value <= 0:
        return 1
    magnitude = 10 ** (len(str(value)) - 1)
    for step in (1, 2, 2.5, 5, 10):
        candidate = int(magnitude * step)
        if candidate >= value:
            return candidate
    return value


# --------------------------------------------------------------------------
# Plain text
# --------------------------------------------------------------------------


def render_text(records: Iterable[dict[str, Any]], *, width: int = 56) -> str:
    """Plain-text stacked bars. The acceptable minimum, and handy in CI logs."""
    runs = group_runs(records)
    if not runs:
        return "no runs"
    out: list[str] = []
    for run in runs:
        header, turns, footer = run["header"], run["turns"], run["footer"]
        out.append(f"run {header['run_id']}  model={header['model']}")
        out.append("=" * (width + 26))
        if not turns:
            out.append("  (no turns recorded)")
            out.append("")
            continue
        kinds, rows = _series(turns)
        peak = max(turn["prompt_tokens"]["counted_total"] for turn in turns)
        scale = _nice_max(peak)

        out.append(
            f"  prompt composition per turn (full bar = {scale:,} tokens)"
        )
        out.append("")
        for turn, row in zip(turns, rows):
            total = turn["prompt_tokens"]["counted_total"]
            bar = ""
            for kind in kinds:
                cells = int(round(row[kind] / scale * width)) if scale else 0
                bar += KIND_GLYPHS.get(kind, "?") * max(0, cells)
            bar = bar[:width].ljust(width, " ")
            recon = turn["reconciliation"]
            flag = "" if recon["within_tolerance"] else "  !recon"
            out.append(f"  t{turn['turn_index']:>2} |{bar}| {total:>9,}{flag}")
        out.append("")
        out.append("  legend: " + "  ".join(
            f"{KIND_GLYPHS.get(k, '?')}={KIND_LABELS.get(k, k)}" for k in kinds
        ))
        out.append("")
        totals = footer["totals"] if footer else {}
        by_kind = totals.get("by_kind_total", {})
        grand = sum(by_kind.values()) or 1
        out.append("  cumulative share of every prompt sent this run")
        for kind in kinds:
            value = by_kind.get(kind, 0)
            out.append(
                f"    {KIND_LABELS.get(kind, kind):<20} {value:>10,}  "
                f"{value / grand:>6.1%}"
            )
        out.append("")
        out.append(
            f"  turns={footer['turns']}  stop={footer['stop_reason']}  "
            f"peak_prompt={totals.get('peak_prompt_tokens', 0):,}  "
            f"output={totals.get('output_tokens_total', 0):,}"
        )
        worst = max(
            (abs(t["reconciliation"]["residual_fraction"]) for t in turns), default=0.0
        )
        out.append(
            f"  worst reconciliation residual vs. usage: {worst:.2%} "
            f"(tolerance {header['attribution']['reconcile_tolerance_fraction']:.2%})"
        )
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# HTML + inline SVG
# --------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
.viz-root {
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --good:#006300; --critical:#d03b3b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--plane); color: var(--text-primary);
  padding: 24px; max-width: 1100px; margin: 0 auto;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
    --good:#0ca30c; --critical:#d03b3b;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --good:#0ca30c; --critical:#d03b3b;
}
.viz-root h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
.viz-root h2 { font-size: 15px; margin: 28px 0 10px; color: var(--text-primary); }
.viz-root p.sub { margin: 0 0 20px; color: var(--text-secondary); font-size: 13px; }
.card {
  background: var(--surface-1); border: 1px solid var(--ring);
  border-radius: 10px; padding: 16px 18px; margin-bottom: 16px;
}
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--ring);
  border-radius: 10px; padding: 12px 16px; min-width: 148px; flex: 1 1 148px;
}
.tile .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--text-muted); }
.tile .value { font-size: 26px; margin-top: 4px; color: var(--text-primary); }
.tile .note { font-size: 11px; color: var(--text-secondary); margin-top: 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 12px;
  font-size: 12px; color: var(--text-secondary); }
.legend span.key { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12px; width: 100%;
  font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--grid); color: var(--text-secondary); }
th { color: var(--text-muted); font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: .05em; }
th:first-child, td:first-child { text-align: left; }
td.total { color: var(--text-primary); }
.ok { color: var(--good); } .bad { color: var(--critical); }
pre.report { font-size: 12px; line-height: 1.45; margin: 0; white-space: pre;
  color: var(--text-secondary); }
.caveat { font-size: 12px; color: var(--text-secondary); margin: 6px 0 0;
  padding-left: 16px; }
svg { display: block; max-width: 100%; height: auto; }
"""


def _color_vars() -> str:
    light = "; ".join(
        f"--k-{k.replace('_', '-')}:{v[0]}" for k, v in KIND_COLORS.items()
    )
    dark = "; ".join(
        f"--k-{k.replace('_', '-')}:{v[1]}" for k, v in KIND_COLORS.items()
    )
    return (
        f".viz-root {{ {light} }}\n"
        f'@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"]))'
        f" .viz-root {{ {dark} }} }}\n"
        f':root[data-theme="dark"] .viz-root {{ {dark} }}\n'
    )


def _stacked_svg(turns: Sequence[dict[str, Any]], kinds: Sequence[str],
                 rows: Sequence[dict[str, int]]) -> str:
    width, height = 760, 320
    left, right, top, bottom = 72, 14, 14, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = max((t["prompt_tokens"]["counted_total"] for t in turns), default=1)
    scale_max = _nice_max(peak)
    slot = plot_w / max(len(turns), 1)
    bar_w = min(54.0, slot * 0.62)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="prompt tokens by segment kind, per turn">'
    ]

    # Recessive gridlines + y ticks.
    for step in range(5):
        value = scale_max * step / 4
        y = top + plot_h - (value / scale_max) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--text-muted)">{int(value):,}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" '
        f'y2="{top + plot_h}" stroke="var(--axis)" stroke-width="1"/>'
    )

    for index, (turn, row) in enumerate(zip(turns, rows)):
        x = left + slot * index + (slot - bar_w) / 2
        cursor = float(top + plot_h)
        stack = [(k, row[k]) for k in kinds if row[k] > 0]
        for position, (kind, value) in enumerate(stack):
            seg_h = value / scale_max * plot_h
            # 2px surface gap between stacked segments.
            drawn = max(1.0, seg_h - (2 if position < len(stack) - 1 else 0))
            y = cursor - seg_h
            var = f"var(--k-{kind.replace('_', '-')})"
            radius = ' rx="4"' if position == len(stack) - 1 else ""
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{drawn:.1f}"{radius} fill="{var}">'
                f"<title>turn {turn['turn_index']} · "
                f"{escape(KIND_LABELS.get(kind, kind))}: {value:,} tokens</title>"
                f"</rect>"
            )
            cursor = y
        total = turn["prompt_tokens"]["counted_total"]
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{cursor - 6:.1f}" '
            f'text-anchor="middle" font-size="11" fill="var(--text-secondary)">'
            f"{total:,}</text>"
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18:.1f}" '
            f'text-anchor="middle" font-size="11" fill="var(--text-muted)">'
            f"t{turn['turn_index']}</text>"
        )
    parts.append(
        f'<text x="{left}" y="{height - 6}" font-size="11" '
        f'fill="var(--text-muted)">turn</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _legend(kinds: Sequence[str]) -> str:
    keys = "".join(
        f'<span class="key"><i style="background:var(--k-'
        f'{k.replace("_", "-")})"></i>{escape(KIND_LABELS.get(k, k))}</span>'
        for k in kinds
    )
    return f'<div class="legend">{keys}</div>'


def _table(turns: Sequence[dict[str, Any]], kinds: Sequence[str],
           rows: Sequence[dict[str, int]]) -> str:
    head = "".join(f"<th>{escape(KIND_LABELS.get(k, k))}</th>" for k in kinds)
    body: list[str] = []
    for turn, row in zip(turns, rows):
        cells = "".join(f"<td>{row[k]:,}</td>" for k in kinds)
        recon = turn["reconciliation"]
        css = "ok" if recon["within_tolerance"] else "bad"
        body.append(
            f"<tr><td>t{turn['turn_index']}</td>{cells}"
            f"<td class=\"total\">{turn['prompt_tokens']['counted_total']:,}</td>"
            f"<td>{turn['usage']['total_prompt_tokens']:,}</td>"
            f'<td class="{css}">{recon["residual_tokens"]:+,} '
            f'({recon["residual_fraction"]:+.2%})</td>'
            f"<td>{escape(str(turn['response']['stop_reason']))}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>turn</th>'
        f"{head}<th>counted total</th><th>usage total</th>"
        "<th>residual</th><th>stop</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="note">{escape(note)}</div>' if note else ""
    return (
        f'<div class="tile"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>{note_html}</div>'
    )


def _run_section(run: dict[str, Any]) -> str:
    header, turns, footer = run["header"], run["turns"], run["footer"]
    parts = [
        f"<h2>run {escape(header['run_id'])}</h2>",
        f'<p class="sub">model {escape(header["model"])} · '
        f"attribution {escape(header['attribution']['method'])} "
        f"@ {escape(header['attribution']['granularity'])} · "
        f"stop {escape((footer or {}).get('stop_reason', '?'))}</p>",
    ]
    if not turns:
        parts.append('<div class="card">no turns recorded</div>')
        return "".join(parts)

    kinds, rows = _series(turns)
    totals = (footer or {}).get("totals", {})
    worst = max(abs(t["reconciliation"]["residual_fraction"]) for t in turns)
    tolerance = header["attribution"]["reconcile_tolerance_fraction"]
    growth = (
        turns[-1]["prompt_tokens"]["counted_total"]
        - turns[0]["prompt_tokens"]["counted_total"]
    )

    parts.append(
        '<div class="tiles">'
        + _tile("peak prompt", f"{totals.get('peak_prompt_tokens', 0):,}", "tokens in one request")
        + _tile("prompt tokens billed-equivalent", f"{totals.get('prompt_tokens_total', 0):,}", "summed over turns")
        + _tile("growth", f"{growth:+,}", "first turn to last")
        + _tile("output", f"{totals.get('output_tokens_total', 0):,}", "tokens generated")
        + _tile(
            "worst residual",
            f"{worst:.2%}",
            f"vs. usage; tolerance {tolerance:.2%}",
        )
        + "</div>"
    )
    parts.append(
        '<div class="card">'
        + _stacked_svg(turns, kinds, rows)
        + _legend(kinds)
        + "</div>"
    )
    parts.append("<h2>per-turn table</h2>")
    parts.append('<div class="card">' + _table(turns, kinds, rows) + "</div>")
    return "".join(parts)


def render_html(
    records: Iterable[dict[str, Any]],
    *,
    title: str = "context budget",
    accuracy: AccuracyReport | None = None,
) -> str:
    """A complete, self-contained HTML page. No external requests of any kind."""
    runs = group_runs(records)
    body = [
        '<div class="viz-root">',
        f"<h1>{escape(title)}</h1>",
        '<p class="sub">Each bar is the whole prompt sent on that turn, broken '
        "down by segment. Counts come from the count_tokens endpoint via "
        "incremental prefix deltas; the residual column compares that "
        "decomposition against the response&rsquo;s authoritative usage "
        "(input + cache_creation + cache_read).</p>",
    ]
    if not runs:
        body.append('<div class="card">no runs in this record stream</div>')
    for run in runs:
        body.append(_run_section(run))

    if accuracy is not None:
        body.append("<h2>accuracy vs. context length</h2>")
        body.append(
            '<div class="card"><pre class="report">'
            + escape(report_text(accuracy))
            + "</pre></div>"
        )

    body.append("</div>")
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{_CSS}\n{_color_vars()}</style></head><body>"
        + "".join(body)
        + "</body></html>"
    )
