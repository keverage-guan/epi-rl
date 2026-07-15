# Policy-conditioned SEIR PINN

A physics-informed neural network (DeepXDE / PyTorch) that fits the Libin et al.
age-structured SEIR meta-population model to the 2009 H1N1 ILI data, in a form that
can later serve as a fast rollout surrogate for an RL agent controlling school closures.

It reuses the reference repo's census, contact, and commute data **at native district
resolution** and calibrates against per-nation ILI series via a district → nation
crosswalk.

## Layout

```
pinn_seir/
  config.py      ModelConfig + TrainConfig (all hyperparameters live here)
  data.py        loads census/contacts/commute/crosswalk/flu into model-ready arrays
  network.py     SEIRPINN: augmented-coordinate net with softmax conservation head
  physics.py     SEIR residuals, mean-field adult coupling, incidence observation op
  schedules.py   closure-schedule sampling (true calendar + extremes + budgeted random)
  model.py       PINNTrainer: four loss terms + joint Adam/L-BFGS optimisation
  fit_pinn.py    CLI: fit, export checkpoint + params.json + per-nation fit plot
  plot_loss.py   CLI: parse a training .out log -> loss-component + parameter curves
  plot_fit.py    CLI: reload a checkpoint -> PINN vs. observed data, holidays shaded
```

Drop `pinn_seir/` next to the existing `epcontrol/` package.

## Install

```
pip install deepxde torch numpy pandas matplotlib tqdm
# DeepXDE selects the PyTorch backend automatically; force it with:
export DDE_BACKEND=pytorch
```

## Run

```
python -m pinn_seir.fit_pinn \
  --census    data/great_brittain/census.csv \
  --commute   data/great_brittain/commute.csv \
  --crosswalk data/great_brittain/crosswalk.tsv \
  --contacts  data/contacts \
  --flu       data/epidemic/uk_flu_per_100000.csv \
  --adam-iters 40000 \
  --out /tmp/seir_pinn
```

Free the identifiability-risky knobs only if the fit demands it:
`--train-kappa`, `--train-alpha`.

## Plotting

Loss and parameter curves from a training log:

```
python -m pinn_seir.plot_loss --log logs/seir_pinn_3309419.out --out outputs/seir_pinn
```

PINN prediction vs. observed data, with school holidays shaded (rebuilds the model
from the same data files, then loads the checkpoint):

```
python -m pinn_seir.plot_fit \
  --checkpoint outputs/seir_pinn/checkpoint.pt \
  --census    data/great_brittain/census.csv \
  --commute   data/great_brittain/commute.csv \
  --crosswalk data/great_brittain/crosswalk.tsv \
  --contacts  data/contacts \
  --flu       data/uk_flu_per_100000.csv \
  --out outputs/seir_pinn --dates
```

Pass `--dates` for a calendar x-axis (else model-week index) and `--no-holidays` to
disable the shaded holiday bands. Checkpoints record their network architecture, so
`plot_fit` rebuilds a matching network automatically; checkpoints saved before this
feature load correctly as long as the default architecture is used.

## Design decisions (and where they live)

- **Patches = districts, observations = nations.** The data loss aggregates district
  predictions up to England / Scotland / Wales using `crosswalk.tsv`
  (`physics.nation_weekly_incidence`, `model.loss_data`). This is deliberately
  underdetermined: many district decompositions sum to the same national curve, so
  district-level structure is supplied by the mechanistic priors, not the data. Treat
  national/age-at-national conclusions as well-founded and district-level claims as
  prior-driven hypotheses.
- **Single trainable R0 → per-district beta** via the next-generation matrix
  (`beta_p = R0·γ / ρ(reciprocal school CM)`), matching `Eames2012.compute_beta`
  (`physics.beta_per_patch`). ζ = 1/day, γ = 1/1.8/day fixed.
- **Exact population conservation** through a softmax head scaled by N; every residual
  is written in fractions of N so each compartment is O(1) (`network.SEIRPINN`).
- **Term/holiday switch by domain decomposition.** Each week is a smooth sub-domain
  (local time τ + a learned week embedding + the closure vector as inputs); adjacent
  weeks are coupled by an **overlapping-strip junction loss** (`model.loss_junction`,
  `overlap_delta ≈ 0.12`). The control itself is never smoothed.
- **Mean-field inter-patch coupling.** The stochastic Poisson ignition (their Eq. 4–5)
  is replaced by its deterministic mean-field analogue: a continuous adult-only inflow
  `Λ_{p,A} = κ·β_p·(S^A_p)^μ·M_AA·Σ_{p'≠p} T_{p'p}·I^A_{p'}/N^A_{p'}`
  (`physics.meanfield_inflow`). The flux enters at native district resolution.
- **Observation model = weekly incidence.** The ILI-per-100k series is treated as
  weekly symptomatic incidence: `α · ∫ ζE dt` over each week, summed over patch/age,
  aggregated to nations, converted to a per-100k rate. The `α = 1/4` symptomatic
  scaling is Libin et al.'s factor (`physics.nation_weekly_incidence`).
- **Seed IC.** 2 exposed adults in Falkirk (Scotland), S ≈ N elsewhere, at week 1
  (`model._build_ic_targets`, configurable via `--seed-district` / `--seed-exposed`).
  `epidemic_start` (default 2009-04-27, the seed date) anchors the model week grid.
- **Schedule generalisation.** Physics + junction losses range over sampled schedules
  (true calendar, all-open, all-closed, budgeted-random); the data loss stays tied to
  the true 2009 calendar (`schedules.py`, `model.total_loss`).

## Calendar and observation alignment

- The true school calendar is built from **real holiday date ranges** in
  `ModelConfig.holiday_ranges` (summer, autumn, christmas, spring), not a hard-coded
  week window (`schedules.holiday_calendar`). Model week `k` spans
  `[epidemic_start + 7k, epidemic_start + 7(k+1))`; a week is CLOSED when at least
  `overlap_threshold` (default 0.5) of its days fall in a holiday range. Lower the
  threshold toward 0 to close any week that touches a holiday; raise it to 1.0 to close
  only fully-enclosed weeks.
- Observations are aligned to model weeks by their **`week_end_date`** column (M/D/Y),
  not by row order: a row ending on date `d` maps to week `floor((d - epidemic_start)/7)`.
  Rows before week 0 or beyond the horizon are dropped, and the series is sorted by
  model week (`data._load_flu_series`). `obs_week_index` carries the true model week of
  each retained observation, and the data loss compares against exactly those weeks.

## Notes / knobs to watch

- Loss terms are normalised to be dimensionless and O(1): residuals in fractions of N,
  the data term divided by the mean-square of the observations. This makes the four
  `w_*` weights in `TrainConfig` directly comparable; retune them there if needed.
- The `flu` CSV is expected with columns `week, report_date, week_end_date, england,
  northern_ireland, scotland, wales, uk`. Northern Ireland and the UK aggregate are
  ignored; only the modelled nations are fit. Alignment is by `week_end_date` (see
  above), so the CSV need not start at epidemic week 1.
- If 43-week joint training is unstable, reduce `--adam-lr`, raise `w_ic`, or widen
  `overlap_delta` before raising `w_junction`. A sequential ("marching") week-by-week
  fallback is the documented next step for long-horizon ODE fitting.
- Well-posedness holds because the closure control is externally imposed, not a function
  of the SEIR state. Do **not** make the closure rule state-dependent inside the PINN.
```