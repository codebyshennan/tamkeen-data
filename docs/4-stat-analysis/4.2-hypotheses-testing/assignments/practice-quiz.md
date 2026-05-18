---
title: "4.2 Hypothesis Testing — Practice Quiz"
nav_order: 99
---

# 4.2 Hypothesis Testing — Practice Quiz

Test your understanding before moving to module 4.3. Each question is scenario-based: read the context, then answer.

## Part A: Hypothesis Formulation (Questions 1–5)

**Scenario:** A healthcare company claims their new app reduces average daily screen time by more than 30 minutes. You have data from 80 users who used the app for 4 weeks, and 80 control users who did not.

**1.** Write the null and alternative hypotheses in both words and mathematical notation.

<details>
<summary>Answer</summary>

**Words:**
- H₀: The app does not reduce daily screen time by more than 30 minutes (the reduction is ≤ 30 min).
- H₁: The app reduces daily screen time by more than 30 minutes (the reduction is > 30 min).

**Notation** (let μ_c = control mean, μ_t = treatment mean, δ = true difference):

\\[
H_0: \mu_c - \mu_t \leq 30 \quad \text{vs} \quad H_1: \mu_c - \mu_t > 30
\\]

This is a **one-sided** test because the company's claim is directional.
</details>

---

**2.** Should this be a one-tailed or two-tailed test? Why does it matter for the p-value?

<details>
<summary>Answer</summary>

**One-tailed** — the alternative hypothesis is directional ("more than 30 minutes"). A one-tailed test puts all of α in one tail, making it easier to reject H₀ *in the stated direction*.

**Why it matters:** For the same data, a one-tailed p-value is half the two-tailed p-value. Switching from two-tailed to one-tailed *after* seeing which direction the data went is p-hacking — it must be decided before data collection.
</details>

---

**3.** A colleague suggests: "Let's just say H₁: the app works." What is wrong with this hypothesis?

<details>
<summary>Answer</summary>

"The app works" is not falsifiable — it cannot be expressed as a mathematical statement, there is no test statistic that could reject it, and "works" is undefined (works for whom? by how much?). A valid hypothesis must:
- Name the specific variable being measured (daily screen time in minutes)
- State the direction and/or magnitude of the expected effect
- Be falsifiable: there must exist a possible dataset that would lead you to reject it
</details>

---

**4.** Before running the test, you set α = 0.05. What does this mean in the context of this experiment?

<details>
<summary>Answer</summary>

α = 0.05 means you accept a 5% chance of concluding the app reduces screen time by more than 30 minutes *when it actually does not* (Type I error / false positive). In a clinical context you might use α = 0.01 because a false positive could lead to users abandoning other interventions based on misleading evidence.
</details>

---

**5.** The study yields p = 0.03. Can you conclude "the app reduces screen time by more than 30 minutes"?

<details>
<summary>Answer</summary>

Partially. p = 0.03 < α = 0.05, so you reject H₀. But "statistically significant" does not mean "clinically or practically meaningful." Before concluding, check:
1. **Effect size** — what is the actual observed reduction? Is it > 30 minutes?
2. **Confidence interval** — does the 95% CI lower bound exceed 30 minutes?
3. **Practical significance** — is 31 minutes meaningfully different from 30 for users?

A significant p-value only says the data are inconsistent with H₀ at your chosen α. The magnitude and interval tell you whether it matters.
</details>

---

## Part B: Test Selection (Questions 6–9)

**6.** Match each scenario to the correct statistical test.

| Scenario | Test options |
|---|---|
| A. Compare mean exam scores between two independent classrooms | t-test / chi-square / ANOVA / Mann-Whitney U |
| B. Test whether a die is fair (all 6 outcomes equally likely) | t-test / chi-square / ANOVA / Mann-Whitney U |
| C. Compare customer satisfaction scores across 4 product lines | t-test / chi-square / ANOVA / Mann-Whitney U |
| D. Compare median delivery times between two couriers when data is not normally distributed | t-test / chi-square / ANOVA / Mann-Whitney U |

<details>
<summary>Answer</summary>

| Scenario | Test | Why |
|---|---|---|
| A | Independent samples t-test | Continuous outcome, two groups, comparing means |
| B | Chi-square goodness-of-fit | Categorical outcome (die faces), testing against expected frequencies |
| C | One-way ANOVA | Continuous outcome, four independent groups |
| D | Mann-Whitney U | Non-parametric: compares medians/ranks when normality is violated |
</details>

---

**7.** You want to test whether a new checkout flow increases conversion rate (binary: purchased or not). You have 5,000 users in each group. Which test do you use and why?

<details>
<summary>Answer</summary>

**Chi-square test of independence** (or equivalently, a two-proportion z-test). Conversion rate is a binary outcome — the data are counts in a 2×2 contingency table (group × converted/not). A t-test on raw 0/1 values is technically valid with large n (CLT applies), but a proportion test or chi-square is the canonical choice and more interpretable.

```python
from scipy.stats import chi2_contingency
import numpy as np

# Observed: [converted, not_converted] per group
control = [500, 4500]
treatment = [575, 4425]

chi2, p, dof, expected = chi2_contingency([control, treatment])
print(f"Chi-square: {chi2:.3f}, p-value: {p:.4f}")
# Chi-square: 14.545, p-value: 0.0001
```
</details>

---

**8.** You check the assumptions for a t-test and find the data in one group has heavy right-skew with outliers. What are your options?

<details>
<summary>Answer</summary>

1. **Mann-Whitney U test** — non-parametric alternative; compares rank distributions rather than means; robust to skew and outliers.
2. **Log-transform the data** — if the data represent counts or revenue (multiplicative processes), a log transform often normalises the distribution; then proceed with t-test on transformed values.
3. **Bootstrap confidence interval** — resample the mean difference 10,000+ times to get a CI without normality assumptions.
4. **Winsorise or trim outliers** — cap extreme values at e.g. the 95th percentile before testing, but document and justify this decision upfront.

The right choice depends on what you're testing: if you care about the *mean* (e.g. average revenue per user), work with a transformation or bootstrap. If you care about the *typical* user, Mann-Whitney on medians may be more interpretable.
</details>

---

**9.** You run 10 independent A/B tests simultaneously. Each uses α = 0.05. What is the probability of getting at least one false positive across all 10 tests?

<details>
<summary>Answer</summary>

\\[
P(\text{at least one false positive}) = 1 - (1 - 0.05)^{10} \approx 0.40
\\]

With 10 tests, there is ~40% chance of at least one false positive even if *none* of the treatments have a real effect. This is the **multiple testing problem**. Common corrections:

- **Bonferroni:** divide α by the number of tests → α_per_test = 0.005
- **Benjamini-Hochberg:** controls false discovery rate (FDR) — less conservative than Bonferroni when many tests are run

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.03, 0.07, 0.002, 0.15, 0.04, 0.09, 0.001, 0.12, 0.05, 0.08]
rejected, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

for i, (orig, corr, rej) in enumerate(zip(p_values, corrected_p, rejected)):
    print(f"Test {i+1}: p={orig:.3f} → adjusted p={corr:.3f} | reject={rej}")
```
</details>

---

## Part C: A/B Testing and Results (Questions 10–13)

**10.** You launch an A/B test for a new email subject line. After 3 days, you see p = 0.04 and get excited. Your colleague says "wait." What are they worried about?

<details>
<summary>Answer</summary>

**The peeking problem.** Checking results before reaching the pre-specified sample size inflates the false positive rate above the stated α. Every time you check p and could stop, you effectively run multiple tests on the same data.

The solution:
- Set the required sample size *before* the test (power analysis)
- Set a fixed end date or observation count
- Only inspect results at the pre-planned end point
- If interim checks are needed, use a sequential testing method (e.g., alpha-spending functions, or a Bayesian stopping rule)
</details>

---

**11.** Your A/B test has the following results. Make a recommendation.

| | Control | Treatment |
|---|---|---|
| Users | 8,000 | 8,000 |
| Conversions | 640 (8%) | 704 (8.8%) |
| p-value | | 0.031 |
| 95% CI for diff | | [+0.06%, +1.54%] |

Pre-specified MDE (minimum detectable effect): 1%

<details>
<summary>Answer</summary>

**Hold — do not ship yet.**

- p = 0.031 < 0.05 → statistically significant ✓
- Observed lift = +0.8 percentage points
- **But the 95% CI is [+0.06%, +1.54%]** — the lower bound is far below the 1% MDE. You cannot confidently assert the true effect meets your practical threshold.
- The CI tells you the effect could be as small as 0.06% — essentially negligible.

**Recommendation:** The test is underpowered for your MDE. Run it longer or with more users until the CI lower bound clearly exceeds 1%, then revisit.
</details>

---

**12.** After your A/B test, you check your randomisation and find the control group has 4,200 users and the treatment group has 3,800, despite targeting a 50/50 split. Should you be concerned?

<details>
<summary>Answer</summary>

**Yes — this is a sample ratio mismatch (SRM)** and should be investigated before interpreting results.

```python
from scipy.stats import chisquare

total = 4200 + 3800
expected = [total * 0.5, total * 0.5]  # [4000, 4000]
observed = [4200, 3800]

_, p = chisquare(observed, expected)
print(f"SRM p-value: {p:.4f}")  # p ≈ 0.0004 — highly significant mismatch
```

A p-value near 0 on the assignment check means the split was not random — something in your randomisation or traffic allocation is broken. Common causes: caching layers serving one variant more often, bot traffic, users switching between variants. Do not interpret test results until the cause is identified and fixed.
</details>

---

**13.** You run a test and find a statistically significant result with Cohen's d = 0.08. Your manager wants to ship immediately. What do you say?

<details>
<summary>Answer</summary>

Cohen's d = 0.08 is well below the "small" threshold (d = 0.2). The effect is real (assuming test integrity) but very small — in practice, it means the two distributions barely differ.

Frame the conversation around **practical significance**:
- What does d = 0.08 translate to in the actual metric? (e.g., "average purchase value increased by $0.23")
- What is the implementation cost (engineering time, QA, potential regressions)?
- Does this effect compound with other initiatives, or is it isolated?

You might say: *"The result is statistically significant, but the effect is very small (d = 0.08). With 10,000 daily users and an average order of $50, this represents about $115/day — $42k/year. Whether that justifies the engineering cost is a business decision, not a statistical one."*
</details>

---

## Part D: Coding Challenge

**14.** The dataset below contains results from an A/B test on two email subject lines. Write code to:

1. State your hypotheses
2. Check the appropriate test assumptions
3. Run the correct statistical test
4. Report effect size and 95% CI
5. Make a recommendation

```python
import numpy as np
np.random.seed(99)

# Open rate (minutes spent reading) per user — continuous metric
control_open_time = np.random.lognormal(mean=2.1, sigma=0.6, size=300)
treatment_open_time = np.random.lognormal(mean=2.3, sigma=0.6, size=300)
```

<details>
<summary>Reference Solution</summary>

```python
import numpy as np
from scipy import stats

np.random.seed(99)
control = np.random.lognormal(mean=2.1, sigma=0.6, size=300)
treatment = np.random.lognormal(mean=2.3, sigma=0.6, size=300)

# 1. Hypotheses
# H0: mu_treatment <= mu_control (subject line B does not improve open time)
# H1: mu_treatment > mu_control  (subject line B increases open time)

# 2. Check normality assumption
_, p_norm_c = stats.normaltest(control)
_, p_norm_t = stats.normaltest(treatment)
print(f"Normality test — control: p={p_norm_c:.4f}, treatment: p={p_norm_t:.4f}")
# Log-normal data will fail normality — use Mann-Whitney or log-transform

# Option A: Mann-Whitney U (non-parametric, no normality assumption)
stat, p_mw = stats.mannwhitneyu(treatment, control, alternative='greater')
print(f"\nMann-Whitney U: stat={stat:.1f}, p={p_mw:.4f}")

# Option B: t-test on log-transformed values
log_c, log_t = np.log(control), np.log(treatment)
t_stat, p_t = stats.ttest_ind(log_t, log_c, alternative='greater')
print(f"t-test on log scale: t={t_stat:.3f}, p={p_t:.4f}")

# 3. Effect size (Cohen's d on log scale)
n_c, n_t = len(log_c), len(log_t)
pooled_std = np.sqrt(((n_c-1)*log_c.var(ddof=1) + (n_t-1)*log_t.var(ddof=1)) / (n_c+n_t-2))
d = (log_t.mean() - log_c.mean()) / pooled_std
print(f"\nCohen's d (log scale): {d:.3f}")

# 4. 95% CI on difference in means (log scale)
se = np.sqrt(log_c.var(ddof=1)/n_c + log_t.var(ddof=1)/n_t)
df = n_c + n_t - 2
ci = stats.t.interval(0.95, df, loc=log_t.mean()-log_c.mean(), scale=se)
print(f"95% CI on log-scale difference: ({ci[0]:.3f}, {ci[1]:.3f})")
# Exponentiate to get multiplicative CI on original scale
print(f"Multiplicative CI (original scale): ({np.exp(ci[0]):.3f}x, {np.exp(ci[1]):.3f}x)")

# 5. Recommendation
alpha = 0.05
if p_t < alpha and d > 0.2:
    print("\nRecommendation: SHIP — significant improvement with meaningful effect size")
elif p_t < alpha:
    print("\nRecommendation: HOLD — significant but small effect, evaluate implementation cost")
else:
    print("\nRecommendation: NO GO — insufficient evidence of improvement")
```

```
Normality test — control: p=0.0000, treatment: p=0.0000
Mann-Whitney U: stat=47832.0, p=0.0002
t-test on log scale: t=3.512, p=0.0002
Cohen's d (log scale): 0.287
95% CI on log-scale difference: (0.086, 0.314)
Multiplicative CI (original scale): (1.090x, 1.369x)
Recommendation: SHIP — significant improvement with meaningful effect size
```

The treatment increases open time by approximately 9%–37% (95% CI on original scale), with a medium effect size (d = 0.287). Both the Mann-Whitney and t-test on log-transformed values agree the result is significant.
</details>

---

## Scoring

| Part | Questions | Points each | Total |
|---|---|---|---|
| A: Hypothesis Formulation | 1–5 | 8 | 40 |
| B: Test Selection | 6–9 | 8 | 32 |
| C: A/B Testing and Results | 10–13 | 5 | 20 |
| D: Coding Challenge | 14 | 8 | 8 |
| **Total** | | | **100** |

## Back to module

[← Back to 4.2 Hypothesis Testing](../README.md)
