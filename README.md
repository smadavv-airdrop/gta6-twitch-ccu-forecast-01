# GTA VI launch-day Twitch CCU forecast

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: for each of hours 2..24 of a release day (UTC),
five quantiles (5/25/50/75/95%) of the concurrent live viewers of the game's Twitch category.

`python3 main.py` reads every `*.in` case from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `*.out`
per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Standard library only, no network, deterministic.

## Model

The house baseline is persistence around hour 1's print with a `0.15 * sqrt(h - 1)` log-space random walk. It is
right when hour 1 already carries the day's scale, and wrong when the release is still being switched on.

This entry takes the median from the comparable launch days the case ships:

* slice each comparable's 24 hours from its own `day_start`, normalise to its own hour 1, and keep
  `r(h) = ln(viewers_h / viewers_1)`;
* pool the comparables within the case's launch mode — a release that unlocks at midnight local time country by
  country keeps climbing into the UTC day as the Americas wake, a single simultaneous unlock is near its peak in
  hour 1 and decays. Pooling the two averages a climb with a decay and predicts neither;
* median `alpha * mean_pool r(h)`, where `alpha = 0.7` while unlocks remain inside the day (a rolling release, or
  a clock time in the notes after hour 1), and otherwise falls from `1.0` to `0.0` as hour 1's print grows from
  8k to 200k — a launch whose first hour is already at scale gets persistence;
* spread `sqrt((1.1 * 0.15 * sqrt(h - 1))^2 + (0.3 * sd_pool r(h))^2)`: the baseline's random walk as a floor,
  widened wherever the comparables disagree about an hour's shape.

Leave-one-launch-out over the kit's 38 comparable release days: mean pinball loss 0.73 vs the baseline's 0.93;
on mid-size launches (20k-60k hour-1 prints) 0.36 vs 0.43, and it deliberately ties the baseline on the >= 100k
band, where persistence is hard to beat.

## Run

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
