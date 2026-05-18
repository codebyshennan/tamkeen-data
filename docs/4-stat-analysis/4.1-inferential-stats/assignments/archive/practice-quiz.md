---
title: "4.1 Inferential Statistics — Practice Quiz"
nav_order: 99
---

# 4.1 Inferential Statistics — Practice Quiz

Test your understanding before moving to module 4.2. Questions progress from concept checks to applied problems.

## Part A: Populations, Samples, and Parameters (Questions 1–4)

**1.** A researcher wants to know the average sleep duration of all adults in Singapore. She surveys 500 randomly selected adults and finds a mean of 6.8 hours with a standard deviation of 1.2 hours. Identify:

a) The population  
b) The sample  
c) The parameter of interest  
d) The sample statistic  

<details>
<summary>Answer</summary>

a) **Population:** All adults in Singapore  
b) **Sample:** The 500 randomly selected adults  
c) **Parameter:** The true mean sleep duration of all Singaporean adults (μ — unknown)  
d) **Sample statistic:** The observed mean x̄ = 6.8 hours (s = 1.2 hours)

Note: The sample standard deviation (s = 1.2) is a statistic used to *estimate* the population standard deviation (σ).
</details>

---

**2.** A company wants to estimate the proportion of defective products on their assembly line. They inspect every 10th item off the line. What sampling method is this, and what is one potential bias?

<details>
<summary>Answer</summary>

**Systematic sampling** — selecting every kth element from an ordered list.

**Potential bias:** If the assembly line has a periodic pattern (e.g., a machine fault that occurs every 10 cycles), systematic sampling might consistently select either defective or non-defective items, misrepresenting the true defect rate. This is called *periodicity bias*.
</details>

---

**3.** True or False, with explanation: "A larger sample is always better."

<details>
<summary>Answer</summary>

**False** — a larger sample reduces sampling error (the estimate is more precise) but does *not* fix systematic bias. A biased sampling method with 10,000 respondents produces a more precise but still wrong estimate.

The famous example: the 1936 *Literary Digest* poll predicted a landslide win for Alf Landon over FDR using 2.4 million responses — one of the largest polls ever — but got the result badly wrong because the sample was biased (respondents were wealthier and more Republican than the electorate).

**What matters:** Both *representativeness* (unbiased method) and *size* (precision).
</details>

---

**4.** What is the difference between a sampling distribution and a population distribution?

<details>
<summary>Answer</summary>

| | Population distribution | Sampling distribution |
|---|---|---|
| **What it shows** | Values of individuals in the population | Values of a *statistic* (e.g., x̄) across many samples |
| **Spread measure** | Standard deviation (σ) | Standard error (σ/√n) |
| **Shape** | Could be any shape | Approaches normal for large n (CLT) |
| **Used for** | Describing individuals | Making inferences about parameters |

**Example:** Heights of all adults might follow a distribution with μ=170cm, σ=10cm. If you take thousands of samples of n=100, the distribution of *sample means* would be approximately normal with mean=170cm, SE=10/√100=1cm.
</details>

---

## Part B: The Central Limit Theorem (Questions 5–6)

**5.** A factory produces bolts with a mean diameter of 10mm and standard deviation of 0.5mm. The distribution is right-skewed. If you take samples of n=50 bolts, what does the sampling distribution of the sample mean look like?

<details>
<summary>Answer</summary>

By the **Central Limit Theorem**, even though the population is right-skewed, the sampling distribution of x̄ for n=50 will be approximately **normal** with:

- Mean: μ_x̄ = 10mm (same as population mean)  
- Standard error: SE = σ/√n = 0.5/√50 ≈ 0.071mm

```python
import numpy as np

# Simulate to verify
np.random.seed(42)
# Right-skewed population (exponential-like)
population = np.random.exponential(scale=0.5, size=100_000) + 9.5  # shift to mean=10

sample_means = [np.mean(np.random.choice(population, size=50)) for _ in range(10_000)]

print(f"Mean of sample means: {np.mean(sample_means):.3f}")   # ≈ 10.0
print(f"SD of sample means:   {np.std(sample_means):.3f}")    # ≈ 0.071
```

The rule of thumb: CLT applies well for n ≥ 30 from mildly skewed distributions; heavier tails require larger n.
</details>

---

**6.** Why does the standard error equal σ/√n? What does this tell you practically about collecting data?

<details>
<summary>Answer</summary>

**Why:** Each sample mean is an average of n independent observations. The variance of an average is the population variance divided by n (Var(x̄) = σ²/n), so the standard deviation is σ/√n.

**Practically:** To halve the standard error (double your precision), you need to *quadruple* your sample size. This square-root relationship has major implications:

| Goal | Cost |
|---|---|
| Reduce SE by 50% (×0.5) | 4× more data |
| Reduce SE by 75% (×0.25) | 16× more data |
| Reduce SE by 90% (×0.1) | 100× more data |

This is why there are diminishing returns to collecting more data. Going from n=100 to n=400 halves the SE. Going from n=400 to n=1600 halves it again. Each halving costs four times as much data.
</details>

---

## Part C: Confidence Intervals (Questions 7–9)

**7.** A 95% confidence interval for mean commute time is [28.4, 33.6] minutes. Which of the following interpretations is correct?

a) There is a 95% chance the true mean commute time is between 28.4 and 33.6 minutes.  
b) 95% of commuters have a commute time between 28.4 and 33.6 minutes.  
c) If we repeated this sampling process many times, 95% of the resulting CIs would contain the true mean.  
d) The sample mean is 31 minutes with 95% certainty.  

<details>
<summary>Answer</summary>

**c) is correct.**

A CI is a procedure, not a probability statement about a fixed parameter. The true mean either is or isn't in [28.4, 33.6] — there's no probability involved once the interval is computed.

The correct interpretation: the *method* of constructing 95% CIs produces intervals that capture the true parameter 95% of the time in repeated samples.

**a)** is wrong — the parameter is fixed; it's not random.  
**b)** is wrong — this would be a prediction interval, not a CI for the mean.  
**d)** is wrong — the sample mean is known exactly; the CI is about the *population* mean.
</details>

---

**8.** A confidence interval for a population mean is computed as:

\\[
\bar{x} \pm t^* \cdot \frac{s}{\sqrt{n}}
\\]

You have x̄ = 50, s = 12, n = 36. Calculate the 95% CI and explain how it would change if:  
a) n increased to 144  
b) Confidence level increased to 99%  

<details>
<summary>Answer</summary>

```python
import numpy as np
from scipy import stats

x_bar, s, n = 50, 12, 36
se = s / np.sqrt(n)  # = 2.0
t_star = stats.t.ppf(0.975, df=n-1)  # ≈ 2.03

ci = (x_bar - t_star * se, x_bar + t_star * se)
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
# 95% CI: (45.94, 54.06)  — width ≈ 8.1 minutes
```

**a) n = 144:** SE = 12/√144 = 1.0 (halved). CI becomes approximately (47.97, 52.03) — half the width.

**b) 99% CI:** t* increases from ≈2.03 to ≈2.72. CI becomes approximately (44.56, 55.44) — wider by ~33%.

**The tradeoff:** Higher confidence → wider interval. More data → narrower interval. You can't have high confidence and high precision without sufficient data.
</details>

---

**9.** A researcher reports: "Our study found no significant effect (p = 0.12, 95% CI: [−0.3, 2.8])." A journalist writes: "Study proves treatment has no effect." What is wrong with the journalist's conclusion?

<details>
<summary>Answer</summary>

Several things are wrong:

1. **Absence of evidence ≠ evidence of absence.** p = 0.12 means the data are not inconsistent with H₀ at α = 0.05 — it does *not* prove H₀ is true.

2. **The CI spans a wide range.** [−0.3, 2.8] includes zero (hence not significant) but also includes effects as large as 2.8, which could be clinically meaningful. The study is *underpowered* to rule out a meaningful effect.

3. **"Proves" is too strong.** Statistics never proves anything; it quantifies evidence.

The researcher could legitimately say: *"We found no statistically significant effect, but the study was underpowered to detect effects smaller than ~2 units. Further study with larger samples is needed before concluding the treatment is ineffective."*
</details>

---

## Part D: P-Values (Questions 10–12)

**10.** Match each statement to "True" or "False":

| Statement | True/False |
|---|---|
| A p-value of 0.01 means there is a 1% probability the null hypothesis is true | |
| A smaller p-value indicates a larger effect size | |
| A p-value above 0.05 means the null hypothesis is true | |
| p-values depend on both effect size and sample size | |

<details>
<summary>Answer</summary>

| Statement | Answer | Explanation |
|---|---|---|
| p-value = P(H₀ is true) | **False** | p = P(data this extreme | H₀ is true) — a conditional probability, not about H₀'s truth |
| Smaller p → larger effect | **False** | p shrinks with larger n even for tiny effects; effect size and p are separate |
| p > 0.05 → H₀ is true | **False** | Fail to reject ≠ accept; the test may just be underpowered |
| p depends on both effect size and n | **True** | p = f(effect size, n, variability) — with enough n even δ=0.001 yields p<0.05 |
</details>

---

**11.** Two studies examine the same drug:
- Study A: n = 50, mean reduction = 5 points, p = 0.03
- Study B: n = 2000, mean reduction = 1 point, p = 0.001

Which result is more clinically important? Which is more statistically significant?

<details>
<summary>Answer</summary>

- **More statistically significant:** Study B (p = 0.001 < p = 0.03)
- **More clinically important:** Study A — a 5-point reduction is 5× larger than a 1-point reduction

Study B's tiny p-value is a consequence of its large sample size (n = 2000), not a large effect. With 2000 patients, almost any difference will be statistically detectable.

This illustrates the core distinction: **statistical significance ≠ practical significance**. A 1-point reduction may be real (not due to chance) but too small to matter clinically. Always report effect sizes alongside p-values.

```python
import numpy as np

# Effect sizes (Cohen's d approximation)
d_A = 5 / (5 * np.sqrt(2))   # rough estimate, assume SD~5
d_B = 1 / (5 * np.sqrt(2))   # same assumed SD

print(f"Study A Cohen's d ≈ {d_A:.2f}")  # ≈ 0.71 (large)
print(f"Study B Cohen's d ≈ {d_B:.2f}")  # ≈ 0.14 (small)
```
</details>

---

**12.** Write code to compute and interpret a one-sample t-test. A nutrition company claims their supplement increases average daily energy levels from a baseline of 5.0 (on a 1–10 scale). You sample 40 users and observe a mean of 5.8 with SD = 1.5.

<details>
<summary>Reference Solution</summary>

```python
import numpy as np
from scipy import stats

# Known values
mu_0 = 5.0      # claimed baseline (H0)
x_bar = 5.8     # observed sample mean
s = 1.5         # sample SD
n = 40          # sample size

# One-sample t-test
# H0: mu <= 5.0  (supplement has no effect above baseline)
# H1: mu > 5.0   (supplement increases energy above baseline)
t_stat = (x_bar - mu_0) / (s / np.sqrt(n))
p_value = 1 - stats.t.cdf(t_stat, df=n-1)  # one-tailed (greater)

# 95% CI for the mean
se = s / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci = (x_bar - t_crit * se, x_bar + t_crit * se)

# Effect size (Cohen's d)
d = (x_bar - mu_0) / s

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value (one-tailed): {p_value:.4f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
print(f"Cohen's d: {d:.3f}")
print()

alpha = 0.05
if p_value < alpha:
    print(f"Result: Reject H0 (p={p_value:.4f} < α={alpha})")
    print(f"The observed mean ({x_bar}) is significantly above baseline ({mu_0}).")
else:
    print(f"Result: Fail to reject H0 (p={p_value:.4f} ≥ α={alpha})")
```

```
t-statistic: 3.375
p-value (one-tailed): 0.0008
95% CI: (5.32, 6.28)
Cohen's d: 0.533

Result: Reject H0 (p=0.0008 < α=0.05)
The observed mean (5.8) is significantly above baseline (5.0).
```

**Interpretation:** The test is significant (p < 0.001). Cohen's d = 0.53 is a medium effect size. The 95% CI [5.32, 6.28] does not include 5.0, consistent with the rejection. However, whether a 0.8-point increase on a 10-point scale is *practically* meaningful for the company's marketing claims is a separate business question.
</details>

---

## Scoring

| Part | Questions | Points each | Total |
|---|---|---|---|
| A: Populations & Sampling | 1–4 | 8 | 32 |
| B: Central Limit Theorem | 5–6 | 8 | 16 |
| C: Confidence Intervals | 7–9 | 10 | 30 |
| D: P-Values | 10–12 | variable | 22 |
| **Total** | | | **100** |

## Back to module

[← Back to 4.1 Inferential Statistics](../README.md)
