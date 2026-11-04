# MovingHearts: Personalized Step-Goal Model

## Overview

A personalized step-goal system designed to support sustained improvement in daily activity while keeping symptoms and intervention burden manageable. Each morning, the system produces a menu of three step goals—**low**, **recommended**, and **high**—and uses a contextual bandit to adapt menu selection over time.

The objective is sustained improvement in average daily steps relative to each participant's baseline, without clinically meaningful symptom deterioration or unacceptable burden. Goal attainment, streaks, and app engagement are not optimization objectives by themselves.

## Inputs and Daily Output

| Input | Role in V1 |
| --- | --- |
| Activity and goal history | Informs candidate menus and recent activity level |
| Daily morning symptoms | Constrains feasible menus |
| Weather | Additional constraint, lower priority than symptoms |
| Data quality | Identifies missing or abnormal records |

Weekly PROMIS-29 and engagement/life-context features are not included in V1 decisions.

Each menu is parameterized by a **center** (`c`, the recommended goal) and **spacing** (`d`):

- Low: `c - d`
- Recommended: `c`
- High: `c + d`

The algorithm selects a feasible `(c, d)` pair after applying clinical and day-to-day change constraints. The offered menu, participant selection, and actual steps are logged.

## Learning and Decision Process

V1 uses **individual Bayesian linear regression** with fixed Gaussian observation noise and **Thompson sampling**. The reward-prediction model has **11 coefficients**, incorporating recent activity, symptoms, weather, relative menu center and spacing, and quadratic/interaction effects.

For each daily decision:

1. Apply clinical constraints to identify feasible menus.
2. Sample parameters from the participant's posterior distribution.
3. Predict rewards for feasible `(c, d)` choices and select a menu.
4. Present low, recommended, and high goals for participant choice.
5. Once actual steps and next-morning symptoms are available, calculate reward and update the participant's posterior.

Each participant learns from their own valid intervention observations. V1 uses a common prespecified prior and **does not use cross-participant pooling**. The initial covariance is diagonal, with variances of `0.25` or `0.0625`; quadratic and interaction terms receive lower prior uncertainty.

## Reward and Cold Start

Daily reward reflects **actual step improvement relative to baseline** together with a **penalty for worsening symptoms**. It is not determined solely by whether the selected goal was achieved. Reward parameters `D` (step scale), `K` (symptom-change scale), and `lambda_s` (symptom penalty weight) require team agreement.

The proposed baseline observation period is **7 days with at least 5 valid activity days**. Before sufficient data are available, decisions use predefined rule-based defaults and conservative bounds. Baseline observations do not train the menu-reward model.

Missing step or symptom outcomes **defer** model updates; they are not treated as zero activity or zero symptom change. Symptom alerts, missing context, model failure, or absence of an eligible menu trigger predefined fallback or escalation procedures. Every offered goal must satisfy clinical constraints.

## Separate Attainment Prediction

A per-participant **Bayesian logistic regression** model is proposed to estimate goal-attainment probability and uncertainty, using target height, recent activity, morning symptoms, external limitations, and menu position. **Its predictions are not used for V1 menu decisions.** Reliability and calibration are to be assessed separately.

## Evaluation Plan

**Offline:** Simulate participant context, menu choice, actual steps, and next-morning symptoms. Compare the adaptive policy against a predefined rule-based policy, using average reward, activity improvement, symptom deterioration, constraint violations, and fallback frequency. Include missing data, behavioral changes, and cases where adaptation provides no benefit.

**Online:** Begin with a supervised feasibility pilot before a controlled study. The planned no-app control evaluates the **complete intervention**, not the bandit alone. Outcomes include sustained change in daily steps, symptoms, PROMIS-29 trends, intervention burden, dropout, and secondary engagement and goal-attainment measures. Study size, duration, safety thresholds, and clinical response procedures remain to be specified with the team.

## Limitations

- One observation per participant per day may be insufficient to estimate all model coefficients reliably.
- The reward model and short outcome window may not capture long-term effects.
- Participant choice and limited menu variation complicate causal interpretation.
- Missing data and changing behavior can bias learning.
- Simulations cannot establish clinical effectiveness; the no-app comparison cannot isolate the bandit's contribution.

**Background references named in the proposal:** HeartSteps, DIAMANTE, Fit2ThriveMB.
