"""HTML fragments for the Q-EWS workbench (audit design system)."""
from __future__ import annotations

from html import escape
from typing import Iterable

from .experiment_catalog import CATALOG, CHAMPIONS, list_csv_paths


def _card(inner: str, *, warn: bool = False) -> str:
    klass = "card blueprint" + (" warn" if warn else "")
    return (
        f'<div class="{klass}">'
        '<i class="corner tl"></i><i class="corner tr"></i>'
        '<i class="corner bl"></i><i class="corner br"></i>'
        f"{inner}</div>"
    )


def dashboard_html() -> str:
    cards = []
    for c in CHAMPIONS:
        warn = c["r2"] < c["goal"]
        note = "ниже цели" if warn else "цель закрыта"
        cards.append(_card(
            f'<p class="card-kicker">{escape(c["scenario"])} · {escape(c["label"])}</p>'
            f'<p class="metric">{c["r2"]:.3f}</p>'
            f'<p class="card-body">R² · {escape(c["model"])} · '
            f'<span class="mono">{escape(c["run"])}</span> · {note} {c["goal"]:.2f}</p>',
            warn=warn,
        ))
    bars = []
    for c in CHAMPIONS:
        pct = max(0.0, min(100.0, c["r2"] * 100))
        fill = "bar-warn" if c["r2"] < c["goal"] else "bar-ok"
        bars.append(
            f'<div class="bar-row"><span class="mono">{escape(c["scenario"])}</span>'
            f'<div class="bar-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div>'
            f'<span class="mono">{c["r2"]:.3f}</span></div>'
        )
    claims = (
        '<div class="claim"><span class="tag tag-accent">C1</span> physics baseline R²≥0.999 ✓</div>'
        '<div class="claim"><span class="tag tag-accent">C2</span> residual &gt; ablations на B, C ✓</div>'
        '<div class="claim"><span class="tag tag-accent">C3</span> AUROC 0.947 @ f=0.15 ✓ (R² плато 0.73)</div>'
        '<div class="claim"><span class="tag tag-accent">C4</span> AUROC 0.942 @ σ=0.05 ✓</div>'
        '<div class="claim"><span class="tag tag-outline">C5</span> mixed слабее per-Hamiltonian</div>'
    )
    return f"""
<div class="wb-page">
  <h1>Results Dashboard</h1>
  <p class="lede">Лучшие модели по сценариям · числа из plan/CSV статей, не из текущей сессии UI.</p>
  <div class="grid-5">{''.join(cards)}</div>
  <div class="grid-2">
    {_card('<p class="card-kicker">R² по сценариям — champions</p>' + ''.join(bars) +
          '<p class="foot">AUROC: A 1.00 · B 0.965 · C 0.891 · D 0.937 · E 0.759</p>')}
    {_card('<p class="card-kicker">Claims — статус к статьям</p>' + claims +
          '<p class="foot">Paper 1 готов · Paper 2 блокер: TFIM R² 0.575 &lt; 0.70 (E8c)</p>')}
  </div>
</div>
"""


def models_html(slots: Iterable[dict]) -> str:
    cards = []
    for s in slots:
        if s.get("meta"):
            m = s["meta"]
            auroc = m.get("auroc")
            auroc_s = f"{auroc:.3f}" if isinstance(auroc, (int, float)) else "—"
            kind = escape(str(m.get("kind") or "IPredictor"))
            body = (
                f'<p class="card-title">{escape(s["label"])}</p>'
                f'<p class="mono muted">R² {m["r2"]:.4f} · MAE {m["mae"]:.4f} · '
                f'AUROC {auroc_s} · n={m["n_samples"]}</p>'
                f'<span class="tag tag-accent">{kind}</span>'
            )
        else:
            body = (
                f'<p class="card-title">{escape(s["label"])}</p>'
                f'<p class="muted">нет чемпиона в checkpoints/</p>'
                f'<span class="tag tag-outline">empty</span>'
            )
        cards.append(_card(
            f'<p class="card-kicker">{escape(s["kicker"])}</p>{body}',
            warn=s.get("warn", False),
        ))
    return f"""
<div class="wb-page">
  <h1>Model Registry</h1>
  <p class="lede">Один чемпион на (n_qubits, interaction, dissipator) · HamiltonianModelRegistry</p>
  <div class="grid-2">{''.join(cards)}</div>
</div>
"""


def catalog_html() -> str:
    rows = []
    for e in CATALOG:
        n = len(list_csv_paths(e))
        status = f"{n} CSV" if n else "нет CSV"
        rows.append(
            "<tr>"
            f'<td class="mono">{escape(e.id)}</td>'
            f"<td>{escape(e.title)}</td>"
            f"<td>{escape(e.paper)}</td>"
            f'<td class="mono">{escape(e.claim)}</td>'
            f"<td>{escape(e.scenarios)}</td>"
            f"<td>{escape(status)}</td>"
            f'<td class="mono muted">{escape(e.module)}</td>'
            "</tr>"
        )
    return f"""
<div class="wb-page">
  <h1>Experiment Runs</h1>
  <p class="lede">Replay = <span class="mono">python -m experiments.eN</span> · --fast или full · одна фоновая задача</p>
  <div class="table-wrap">{_card(
      '<p class="card-kicker">каталог E1–E14</p>'
      '<table class="wb-table"><thead><tr>'
      "<th>id</th><th>title</th><th>paper</th><th>claim</th><th>scenarios</th><th>csv</th><th>module</th>"
      "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
  )}</div>
</div>
"""


def compare_html(text: str) -> str:
    return _card(
        '<p class="card-kicker">compare</p>'
        f'<pre class="log">{escape(text)}</pre>'
    )


def job_html(md_or_text: str) -> str:
    return _card(
        '<p class="card-kicker">job queue</p>'
        f'<pre class="log">{escape(md_or_text)}</pre>'
    )


def page_title(title: str, lede: str) -> str:
    return f'<div class="wb-page"><h1>{escape(title)}</h1><p class="lede">{lede}</p></div>'
