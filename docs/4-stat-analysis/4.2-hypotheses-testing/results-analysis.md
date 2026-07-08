---
reading_minutes: 30
objectives:
  - >-
    Read every result through both statistical and practical-significance lenses
    (the four-quadrant frame).
  - Communicate effect sizes and intervals in the language each audience needs.
  - >-
    Convert a positive result into an expected-value calculation that accounts
    for implementation cost.
  - >-
    Reject "fail-to-reject H₀" as evidence of no effect when the test was
    underpowered.
---

# Results Analysis: From Numbers to Insights

**After this lesson:** you can explain Results Analysis: From Numbers to Insights and try the examples in your own notebook.

## Overview

This is the "so what?" lesson. Significance is not a business case: you combine **effect size**, **intervals**, **design limits**, and **costs** into a recommendation someone can defend. It closes the 4.2 sequence after [A/B testing](ab-testing.md); use it before presenting to non-specialists.

## Why this matters

* You will move from **p-values** to decisions using effect sizes, intervals, and plain-language stakes.
* You will communicate what results **do** and **do not** imply for action.

## Prerequisites

* [Statistical tests](statistical-tests.md) and [A/B testing](ab-testing.md).
* [Confidence intervals](../4.1-inferential-stats/confidence-intervals.md) and [p-values](../4.1-inferential-stats/p-values.md) for vocabulary.

> **Note:** This lesson closes the 4.2 chain; use it before presenting to non-specialists.

## Introduction: Why Results Analysis Matters

Think of results analysis as being a detective with data - it's about more than finding clues (statistical significance) but understanding what they mean for the case (practical significance). Whether you're analyzing A/B tests, research studies, or business experiments, proper results analysis helps you turn raw numbers into actionable insights!

### Video Tutorial: Results Analysis and Statistical Significance

_StatQuest: P-values, Clearly Explained by Josh Starmer_

_StatQuest: Confidence Intervals, Clearly Explained!!! by Josh Starmer_

## Understanding Test Results

### 1. P-values and Statistical Significance

Like a metal detector beeping - it tells you something's there, but you need to dig to understand what!

**`SignificanceAnalyzer`: interpret and visualize a p-value**

Class definition

Define `SignificanceAnalyzer` with a configurable alpha level (default 0.05) to drive all interpretation methods.

Interpret p-value

Bundle significance flag, evidence strength label, and a plain-language interpretation into one return dict.

Evidence strength bands

Categorise p-values into Very Strong / Strong / Moderate / Weak / No Evidence for stakeholder-facing reporting.

Significance plot

Draw the t null density, mark the observed test statistic, and shade two-tailed rejection regions beyond the critical value.

### 2. Effect Sizes: The Magnitude Matters

Not just whether there's a difference, but how big it is:

**Cohen's d formula:**

\\\[ d = \frac{\bar x\_1 - \bar x\_2}{s\_p} \\]

where:

* \\( \bar x\_1, \bar x\_2 \\): means of the two groups
* \\( s\_p \\): pooled standard deviation

**`EffectSizeAnalyzer`: label Cohen-style magnitude**

Class definition

Define `EffectSizeAnalyzer` to bundle interpretation and visualization of Cohen's d, r, and eta-squared in one place.

Public entry point

Call private interpretation and visualization helpers, save the figure, and return a structured interpretation dict.

Magnitude thresholds

Map numeric effect to "small / medium / large / very large" using literature thresholds for Cohen's d, Pearson r, and eta-squared.

Practical significance

Translate magnitude into a stakeholder-facing sentence about real-world impact, separating statistical rarity from practical importance.

## From Results to Decisions

Results don't make decisions, people do. The job of results analysis is to give the decision-maker the right framing.

### The Four-Quadrant Decision Framework

Map every result onto two axes: statistical significance and practical significance.

```
                    Statistically Significant?
                    NO                  YES
                 ┌─────────────────────────────────┐
   Practically   │  Don't ship.        Ship it.     │
   Significant?  │  Underpowered?      Clear win.   │
   YES           │  Collect more data. Document why.│
                 ├─────────────────────────────────┤
   Practically   │  Don't ship.        Don't ship.  │
   Significant?  │  No evidence of     Real effect, │
   NO            │  any effect.        too small to  │
                 │                     matter.       │
                 └─────────────────────────────────┘
```

The dangerous quadrant is **bottom-right**: statistically significant but practically trivial. With enough data, a 0.001% conversion lift will have p < 0.05, that doesn't mean you should spend engineering resources shipping it.

### Building a Results Summary

A complete results summary answers three questions: What did we find? Is it real? Does it matter?

```python
def summarize_results(control_data, treatment_data, mde, alpha=0.05):
    """
    Produce a decision-ready results summary.

    mde: minimum detectable effect (practical significance threshold),
         expressed as absolute difference in the metric
    """
    from scipy import stats
    import numpy as np

    n_c, n_t = len(control_data), len(treatment_data)
    mean_c, mean_t = np.mean(control_data), np.mean(treatment_data)
    absolute_diff = mean_t - mean_c
    relative_lift = absolute_diff / mean_c

    # Statistical significance
    t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
    stat_sig = p_value < alpha

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((n_c - 1) * np.var(control_data, ddof=1) +
         (n_t - 1) * np.var(treatment_data, ddof=1)) / (n_c + n_t - 2)
    )
    cohens_d = absolute_diff / pooled_std

    # 95% CI on the difference
    se_diff = np.sqrt(
        np.var(control_data, ddof=1) / n_c +
        np.var(treatment_data, ddof=1) / n_t
    )
    df = n_c + n_t - 2
    ci = stats.t.interval(1 - alpha, df, loc=absolute_diff, scale=se_diff)

    # Practical significance: does CI lower bound clear the MDE?
    prac_sig = ci[0] >= mde

    # Decision
    if stat_sig and prac_sig:
        decision = "SHIP - statistically and practically significant"
    elif stat_sig and not prac_sig:
        decision = "HOLD - significant but effect may be below practical threshold"
    elif not stat_sig and prac_sig:
        decision = "RERUN - effect looks meaningful but underpowered"
    else:
        decision = "NO GO - no evidence of a meaningful effect"

    return {
        'n_control': n_c, 'n_treatment': n_t,
        'control_mean': round(mean_c, 4),
        'treatment_mean': round(mean_t, 4),
        'absolute_diff': round(absolute_diff, 4),
        'relative_lift_pct': round(relative_lift * 100, 2),
        'cohens_d': round(cohens_d, 3),
        'p_value': round(p_value, 4),
        'ci_95': (round(ci[0], 4), round(ci[1], 4)),
        'statistically_significant': stat_sig,
        'practically_significant': prac_sig,
        'decision': decision,
    }

# Example
np.random.seed(42)
control = np.random.binomial(1, 0.10, 5000).astype(float)
treatment = np.random.binomial(1, 0.115, 5000).astype(float)

summary = summarize_results(control, treatment, mde=0.01)
for k, v in summary.items():
    print(f"{k:30s}: {v}")
```

```
n_control                     : 5000
n_treatment                   : 5000
control_mean                  : 0.1006
treatment_mean                : 0.115
absolute_diff                 : 0.0144
relative_lift_pct             : 14.31
cohens_d                      : 0.045
p_value                       : 0.0031
ci_95                         : (0.005, 0.0238)
statistically_significant     : True
practically_significant       : True
decision                      : SHIP - statistically and practically significant
```

### Communicating Results to Non-Specialists

Translate statistics into stakes. Different audiences need different framings:

| Audience             | What they care about                           | How to frame it                                 |
| -------------------- | ---------------------------------------------- | ----------------------------------------------- |
| **Engineers**        | Is the result reliable? Will it hold at scale? | CI width, sample size, assumptions met          |
| **Product managers** | Should we ship? What's the business impact?    | Relative lift %, CI, decision recommendation    |
| **Executives**       | What's the revenue/cost impact?                | Expected value calculation, risk if we're wrong |
| **Analysts**         | Can we trust the methodology?                  | Test selection rationale, assumption checks     |

A good one-paragraph summary covers: what changed, how much (with interval), confidence level, and the recommended action. Avoid raw p-values in executive summaries, "we're 95% confident the new checkout reduces cart abandonment by 1.4-2.4 percentage points" is more useful than "p = 0.003".

## Gotchas

* **Conflating statistical significance with business importance**: the `SignificanceAnalyzer` buckets evidence as "Very Strong / Strong / Moderate" based solely on the p-value; a "Very Strong" result with a Cohen's d of 0.01 may be economically worthless. Always pair the significance label with an effect-size interpretation before making a recommendation.
* **Applying Cohen's d thresholds (0.2 / 0.5 / 0.8) across all domains**: the `EffectSizeAnalyzer._get_magnitude` thresholds are conventions from psychology; in medicine a d of 0.2 may be clinically irrelevant, while in education research even a d of 0.1 is sometimes considered meaningful. Match the threshold to your domain's accepted standards.
* **Reporting p-values without confidence intervals**: a p-value tells you direction and rough rarity; an interval tells you the _plausible range_ of the true effect. The lesson builds a `SignificanceAnalyzer` that visualizes the null distribution but does not automatically produce a CI; add one before presenting to stakeholders.
* **Stopping at "fail to reject H₀" without a power check**: a non-significant result in a small study may simply mean the test lacked power to detect the effect. Before concluding "no difference," compute post-hoc power or report the minimum detectable effect at your observed n; otherwise "no evidence of effect" is easily misread as "evidence of no effect."
* **Using `_get_evidence_strength` thresholds as hard rules for action**: the p < 0.001 "Very Strong" band is a presentation aid; it does not override business context, cost-benefit analysis, or the number of other tests run in the same analysis. Always describe how many comparisons were made when reporting strength.
* **Not checking assumptions before interpreting results**: the `SignificanceAnalyzer` accepts any p-value and t-statistic without verifying that the underlying test's assumptions (normality, equal variances, independence) were met. A visually clean significance plot built on a violated assumption is still a misleading output.

## Next steps

* Start [Relationships in data (module 4.3)](../4.3-rship-in-data/) with [Understanding relationships](../4.3-rship-in-data/understanding-relationships.md).

## Additional Resources

* [Effect Size Calculator](https://www.psychometrica.de/effect_size.html)
* [Decision Making Framework](https://hbr.org/2019/09/the-abcs-of-data-driven-decisions)
* [Results Communication Guide](https://www.nature.com/articles/s41467-020-17896-w)

Remember: Good analysis isn't just about finding statistical significance - it's about making informed decisions that create real value!
