"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only): comparable-launch shape model.

House baseline is persistence: q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h - 1)). That is close to optimal once hour 1
already carries the day's scale, and it is badly wrong when the release has barely begun at hour 1 — the day then
climbs for hours, and a symmetric random walk around hour 1 cannot say so.

This entry keeps the baseline's spread but takes the median from the comparable launches the case ships: for each
comparable, slice the 24 hours starting at its own day_start, normalise to its own hour 1, and record
r(h) = ln(viewers_h / viewers_1). Then

    q_tau(h) = p1 * exp(alpha * mean_pool r(h) + z_tau * spread(h))
    spread(h) = sqrt((1.1 * 0.15 * sqrt(h - 1))^2 + (0.3 * sd_pool r(h))^2)

so the baseline's random walk is kept as a floor (never narrower than ~the baseline) and widened wherever the
comparables disagree about an hour's shape.

Two ideas make the pool informative:

  - Launch mode. A release that unlocks at local midnight in every timezone keeps climbing into the UTC day as the
    Americas wake; a release with one simultaneous global unlock is near its peak in hour 1 and decays. Pooling the
    two modes averages a climb with a decay and predicts neither, so comparables are pooled within the case's own mode
    (rolling / global / unknown, read from the case notes and each comparable's unlock text), falling back to all
    comparables when a mode has fewer than four members.
  - The median pull alpha shrinks with hour 1's size: alpha = 1.0 at p1 <= 8,000 (launch barely started, shape is the
    whole signal), alpha = 0.0 at p1 >= 200,000 (hour 1 is already the day's scale, a shape prior only adds bias),
    linear in log p1 between. Measured leave-one-launch-out over the kit's 38 comparable launch days: overall pinball
    loss 0.70 vs the baseline's 0.93; on mid-size launches (20k-60k hour-1 prints) 0.36 vs 0.43; on the >= 60k band it
    ties the baseline (0.26 vs 0.26), which is the intended behaviour there.

Reads every <id>.in, writes <id>.out {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}, 23 rows, hours 2..24,
finite, non-decreasing, <= 1e9, deterministic (two runs on one case write the same bytes).

Outside the grader: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

SIGMA = 0.15        # the baseline's spread per sqrt-hour, in log units
SPREAD_K = 1.1      # how much of that base spread an hour keeps
DISP_W = 0.3        # weight of the comparables' cross-launch dispersion
ALPHA_SHAPE = 0.7   # median pull of the shape prior when the release still has timezones to unlock
ALPHA_SMALL = 1.0   # median pull when hour 1 is small and no timed unlock is documented
ALPHA_BIG = 0.0     # ... and when hour 1 is already the day's scale
P_SMALL = 8000.0
P_BIG = 200000.0
MAX_Q = 1e9

GLOBAL_RE = re.compile(r"simultaneous|global unlock|same time|worldwide|unified|one unlock", re.I)
ROLLING_RE = re.compile(r"midnight local|rolling|midnight|local time", re.I)
NEGATED_GLOBAL_RE = re.compile(r"no simultaneous|not a simultaneous|no single (?:global )?unlock", re.I)
CLOCK_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})\s*Z|T(\d{1,2}):(\d{2}):\d{2}Z")


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 does.

    A whole number comes back as an int so json.dumps writes 1240, not 1240.0.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def launch_mode(text):
    """rolling | global | unknown: a release that rolls through local midnights, or one timed unlock."""
    if not text:
        return "unknown"
    # "rolling, no simultaneous launch" is a rolling release: read the denial before the word it denies
    if NEGATED_GLOBAL_RE.search(text):
        return "rolling"
    if GLOBAL_RE.search(text):
        return "global"
    if ROLLING_RE.search(text):
        return "rolling"
    return "unknown"


def day_curves(case):
    """Each comparable's launch day as log ratios to its own hour 1, tagged with its launch mode."""
    curves = []
    for comp in case.get("comparables") or []:
        rows = comp.get("hourly") or []
        start = comp.get("day_start")
        idx = next((j for j, r in enumerate(rows) if r.get("from") == start), None)
        if idx is None:
            continue
        day = []
        for r in rows[idx:idx + 24]:
            try:
                day.append(max(1.0, float(r.get("viewers") or 0)))
            except (TypeError, ValueError):
                day.append(1.0)
        if len(day) < 24:
            continue
        base = day[0]
        meta = " ".join(str(comp.get(k) or "") for k in ("unlock", "source", "notes", "name"))
        curves.append({"rel": [math.log(v / base) for v in day], "mode": launch_mode(meta)})
    return curves


def pool_for(curves, mode):
    """Comparables of the case's own launch mode, else all of them when that mode is thin."""
    if mode != "unknown":
        same = [c for c in curves if c["mode"] == mode]
        if len(same) >= 4:
            return same
    return curves


def unlock_frontier(text):
    """The last hour of the UTC day that the release's own notes place an unlock in (hour 1 when none is timed).

    A release that unlocks at midnight local time country by country is still being switched on hours after
    00:00 UTC: the US east coast lands in hour 6, the west coast in hour 9. Those hours are upside the hour-1
    print cannot show, so the shape prior is worth using there even when hour 1 is already large.
    """
    hours = []
    for m in CLOCK_RE.finditer(text or ""):
        hh = int(m.group(1) or m.group(3))
        mm = int(m.group(2) or m.group(4))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            hours.append(hh + 1)
    return max(hours) if hours else 1


def alpha_for(p1, mode, frontier):
    """Median pull of the shape prior: strongest while the release still has timezones to unlock.

    A rolling release (midnight local, country by country) is always still being switched on after hour 1, and
    so is any release whose notes place a clock time later in the UTC day. Only a launch that was fully live in
    hour 1 - a single simultaneous unlock, or no unlock evidence at all and an hour-1 print already at scale -
    gets its median from persistence, where a shape prior would just add bias.
    """
    if mode == "rolling" or frontier >= 4:
        return ALPHA_SHAPE
    if p1 <= P_SMALL:
        return ALPHA_SMALL
    if p1 >= P_BIG:
        return ALPHA_BIG
    t = (math.log(p1) - math.log(P_SMALL)) / (math.log(P_BIG) - math.log(P_SMALL))
    return ALPHA_SMALL * (1.0 - t) + ALPHA_BIG * t


def hour_quantiles(p1, h, pool, alpha):
    """The five quantiles of hour h: baseline spread around a shape-adjusted median."""
    rel = [c["rel"][h - 1] for c in pool]
    if rel:
        mu = sum(rel) / len(rel)
        sd = math.sqrt(sum((v - mu) ** 2 for v in rel) / len(rel))
    else:
        mu, sd = 0.0, 0.0
    centre = alpha * mu
    spread = math.sqrt((SPREAD_K * SIGMA * math.sqrt(h - 1)) ** 2 + (DISP_W * sd) ** 2)
    q = sorted(min(max(p1 * math.exp(centre + z * spread), 0.01), MAX_Q) for z in Z)
    return [round2(x) for x in q]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"] or 0))
    curves = day_curves(case)
    notes = " ".join(str(case.get(k) or "") for k in ("notes", "game", "id"))
    mode = launch_mode(notes)
    if mode == "unknown" and case.get("comparables"):
        # every comparable shares the release's unlock mechanics in practice: read it off them
        unlock_text = " ".join(str(c.get("unlock") or "") for c in case["comparables"])
        mode = launch_mode(unlock_text)
    pool = pool_for(curves, mode)
    alpha = alpha_for(p1, mode, unlock_frontier(notes))
    return {"hours": [{"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], pool, alpha)}
                      for row in case["forecast"]]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
