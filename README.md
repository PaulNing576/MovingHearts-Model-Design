# MovingHearts — Personalized Step-Goal Model

**A contextual-bandit model design for adaptive daily step goals in a digital health intervention.** The model proposes three options each morning—**low, recommended, and high**—and learns from each participant's activity and symptom outcomes while respecting clinical constraints.

**Core methods:** Thompson sampling · Bayesian linear regression · Personalized posterior updates · Constrained menu selection

> **Scope:** This repository documents the V1 model design and proposed evaluation. Simulation results do not establish clinical effectiveness.

## How it works

| Component | Design |
| --- | --- |
| **Daily decision** | Choose a three-goal menu based on recent activity, symptoms, weather, and clinical restrictions. |
| **Personalization** | Use Thompson sampling over an individual's Bayesian reward-model posterior. |
| **Feedback** | Learn from actual steps and next-morning symptom changes—not goal attainment alone. |
| **Safety and recovery** | Enforce goal boundaries; defer updates on missing outcomes and fall back when a valid menu cannot be produced. |

### Decision workflow

![MovingHearts model workflow](assets/model-workflow.png)

The workflow covers baseline collection, morning-context checks, feasible menu optimization, participant selection, reward calculation, posterior updating, and fallbacks.

## Model design

### 1. Three-goal menu

The recommended step target is the **center** $c$; $d$ determines the distance to the lower and higher choices:

$$
\mathrm{Menu}_t=(c_t-d_t,\;c_t,\;c_t+d_t).
$$

Candidate $(c,d)$ pairs must satisfy clinical and day-to-day change constraints. Morning inputs include recent step and goal history, symptom reports, weather, and data-quality checks. Weekly PROMIS-29 and engagement/life-context features are excluded from V1 decisions.

### 2. Reward prediction: 11-parameter model

The model estimates the reward of each feasible menu using current context and menu geometry:

$$
\begin{aligned}
\hat R_t(c,d;\theta) ={}& \theta_0+\theta_a a_t+\theta_f f_t+\theta_w w_t\\
&+\theta_c z_c+\theta_{cc}z_c^2\\
&+\theta_d z_d+\theta_{dd}z_d^2+\theta_{cd}z_cz_d\\
&+\theta_{cf}z_cf_t+\theta_{cw}z_cw_t.
\end{aligned}
$$

The parameter and feature vectors are:

$$
\theta=(\theta_0,\theta_a,\theta_f,\theta_w,\theta_c,\theta_{cc},\theta_d,\theta_{dd},\theta_{cd},\theta_{cf},\theta_{cw})^\top
$$

$$
x_t(c,d)=(1,a_t,f_t,w_t,z_c,z_c^2,z_d,z_d^2,z_cz_d,z_cf_t,z_cw_t)^\top.
$$

Here, $a_t$ is recent activity relative to baseline, $f_t$ is morning symptom burden, and $w_t$ represents weather. With baseline step count $B$ and step scale $D$:

$$
z_c=\frac{c-B}{D},\qquad z_d=\frac{d}{D},\qquad
 a_t=\frac{\text{7-day average steps}-B}{D}.
$$

The quadratic and interaction terms allow menu-center and spacing effects to vary with context.

### 3. Bayesian learning and Thompson sampling

V1 specifies **individual Bayesian linear regression** with fixed Gaussian observation noise:

$$
\theta\sim\mathcal N(\mu_0,\Sigma_0).
$$

All prior means are zero. The diagonal covariance corresponds to the following initial standard deviations:

| Coefficients | Initial standard deviation |
| --- | ---: |
| $\theta_0,\theta_a,\theta_f,\theta_w,\theta_c,\theta_d$ | $0.5$ |
| $\theta_{cc},\theta_{dd},\theta_{cd},\theta_{cf},\theta_{cw}$ | $0.25$ |

Each day, the policy samples parameters from the participant's posterior, scores feasible $(c,d)$ menus, and selects a menu. The posterior is updated when the required outcome data arrive. **V1 does not use cross-participant pooling.**

### 4. Actual reward and model update

The observed reward balances step improvement relative to baseline against symptom deterioration:

$$
R_t=\operatorname{clip}\left(\frac{S_t-B}{D},-1,1\right)-\lambda_sW_t
$$

$$
W_t=\operatorname{clip}\left(\frac{\max(0,F_{t+1}-F_t)}{K},0,1\right).
$$

$S_t$ is the actual step count; $F_t$ and $F_{t+1}$ are symptom scores from the current and following mornings. $K$ scales symptom deterioration, and $\lambda_s$ weights its penalty. These reward parameters require team agreement. **Missing outcome data postpone the posterior update rather than being treated as zero.**

### 5. Separate goal-attainment model

A per-participant Bayesian logistic regression model is also proposed for estimating goal-attainment probability and uncertainty:

$$
p=\frac{1}{1+e^{-z}},\qquad
z=\beta_0+\beta_1q+\beta_2a+\beta_3f+\beta_4c+\beta_5m.
$$

Its inputs cover target height, recent activity, symptoms, external limitations, and menu position. **Attainment predictions are not used to choose menus in V1.**

## Cold start, constraints, and fallbacks

- **Baseline:** Proposed observation period of 7 days, with at least 5 valid activity days; baseline observations do not train the reward model.
- **Constraints:** Every offered goal must respect predefined clinical limits and adjustment bounds.
- **Fallbacks:** Missing morning context, symptom alerts, model failure, or no eligible menu trigger predefined fallback, pause, or escalation procedures.
- **Missing outcomes:** Defer reward updates until required step and symptom records are available.

## Evaluation plan and limitations

**Offline:** Simulate participant contexts, goal selections, steps, and next-morning symptoms; compare the adaptive policy with a predefined rule-based policy on activity improvement, reward, symptom deterioration, constraint violations, and fallback frequency.

**Online (proposed):** Start with a supervised feasibility pilot, followed by a controlled study if safety and feasibility criteria are met. The planned no-app control would evaluate the **overall intervention**, not the bandit in isolation.

**Limitations:** Daily individual-level data may be too sparse to estimate all coefficients reliably; the short reward window may miss longer-term effects; participant choice and missing data complicate inference; simulation alone does not demonstrate clinical benefit.

*Background references noted in the proposal: HeartSteps, DIAMANTE, Fit2ThriveMB.*
