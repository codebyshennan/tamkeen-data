---
reading_minutes: 35
objectives:
  - State the precise definition of a p-value (and three things it is NOT).
  - Read a reported p-value alongside an effect size and a confidence interval.
  - Distinguish Type I error, Type II error, and statistical power.
  - Apply Bonferroni / FDR corrections when running multiple tests.
---

# Understanding P-values: Your Statistical Detective Tool

**After this lesson:** you can explain Understanding P-values: Your Statistical Detective Tool and try the examples in your own notebook.

## Overview

A **p-value** answers a narrow question: "If the null hypothesis were true, how often would we see a test statistic this extreme or more?" It is not a probability that a hypothesis is true, and it is not the same as effect size. This lesson connects the definition to plots and common misreadings; [parameters and statistics](parameters-statistics.md) next revisits notation and estimators in one place.

## Why this matters

Software and papers will keep showing p-values whether you like them or not. This lesson matters because:

* You will read **p-values** in papers, A/B tools, and software output; you will align words with what the number does (and does not) mean.
* You will avoid the common mistake of confusing **statistical significance** with **practical importance**.

## Prerequisites

* [Population vs sample](population-sample.md), [confidence intervals](confidence-intervals.md), and [sampling distributions](sampling-distributions.md).
* Optional: [Module 1.3 statistics](../../1-data-fundamentals/1.3-intro-statistics/) if you want a notation refresher before diving in.

> **Note:** P-values do not measure the probability that either hypothesis is true.

## Introduction: The Story of P-values

Imagine you're a detective trying to solve a mystery. You have a default theory (null hypothesis), but you've found some evidence that might suggest otherwise. How strong does this evidence need to be to convince you to reject your default theory? That's where p-values come in!

### Video Tutorial: P-values Explained

_StatQuest: P-values, Clearly Explained by Josh Starmer_

![P-value Concept](../../../.gitbook/assets/p_value_concept_diagram.png) _Figure 1: Visual representation of p-value concept. The shaded area represents the probability of observing results as extreme or more extreme than what we got, assuming the null hypothesis is true._

## What is a P-value?

A p-value is the **probability of observing results at least as extreme as what we got, assuming our null hypothesis is true**. Think of it as a measure of surprise - how unexpected are our results if nothing interesting is actually happening?

### The Mathematical Definition

$$p = P(|T| \geq |t| | H_0)$$

where:

* T is the test statistic
* t is the observed value
* H\_0 is the null hypothesis

![P-value Calculation](../../../.gitbook/assets/p_value_calculation_diagram.png) _Figure 2: Visual explanation of p-value calculation. The red line shows our observed test statistic, and the shaded area represents the p-value._

## The Key Players in Hypothesis Testing

### 1. Null Hypothesis (H₀)

* The "nothing special happening" theory
* The default position we assume is true
* Examples:
  * "The new drug has no effect"
  * "The dice is fair"
  * "The new website design doesn't affect sales"

### 2. Alternative Hypothesis (H₁ or Hₐ)

* The "something's happening" theory
* What we're actually interested in proving
* Examples:
  * "The new drug affects recovery time"
  * "The dice is loaded"
  * "The new design increases sales"

### 3. Significance Level (α)

* Our threshold for "surprising enough"
* Usually 0.05 (5%) or 0.01 (1%)
* Must be set before analyzing data!

![Hypothesis Testing Framework](../../../.gitbook/assets/hypothesis_testing_diagram.png) _Figure 3: Visual representation of the hypothesis testing framework. The diagram shows the relationship between null and alternative hypotheses, and how the significance level divides the decision space._

## How to Interpret P-values: A Decision Guide

### The Basic Rules

```
if p < α:
    "Reject H₀ (Result is statistically significant)"
else:
    "Fail to reject H₀ (Result is not statistically significant)"
```

> Note the wording: when \\(p \geq \alpha\\) we say "**fail to reject** H₀," not "**accept** H₀." A non-significant result means the data are consistent with the null, not that the null has been proven true.

### What p-values do NOT tell you

The definition above is narrow on purpose. Three of the most common misreadings flip the conditional, swap the question, or confuse statistical significance with practical importance:

* **NOT the probability that H₀ is true.** The p-value is computed _assuming_ H₀ is true; it cannot turn around and tell you the probability that H₀ itself is true. A Bayesian posterior probability answers that different question.
* **NOT the probability the result is due to chance.** A p-value of 0.03 does not mean "there is a 3% chance the effect is random"-it means "if the null were true, results this extreme would happen about 3% of the time."
* **NOT the probability of being wrong if you reject H₀.** That is a Type I error rate (\\(\alpha\\)) you set in advance for the procedure, not an attribute of one specific p-value.

A separate trap, covered next, is treating a small p-value as evidence of a _large_ effect; sample size alone can drive p-values down.

**Two-sample t-test + overlapping histograms**

```
Clinical Trial Analysis
Control group mean: 9.62 days
Treatment group mean: 8.76 days
P-value: 0.0722
Result: Not significant
```

Two simulated arms

30 control patients with mean recovery 10 days; 30 treatment patients with mean 9 days. Real difference: 1 day.

The test itself

`ttest_ind` compares the two means and returns the two-sided p-value used in the decision.

Decision rule

Compare p to α = 0.05. With this small sample the real 1-day difference may not cross the threshold, power matters.

![Recovery Times Distribution](../../../.gitbook/assets/recovery_times_distribution.png) _Figure 4: Distribution of recovery times for control and treatment groups. The dashed lines indicate the mean recovery time for each group._

## Common Misconceptions: What P-values Are NOT

### 1. NOT the Probability H₀ is True

P-values don't tell us the probability of our hypothesis being correct.

### 2. NOT the Probability of Getting Results by Chance

This common misinterpretation can lead to poor decisions.

### 3. NOT the Effect Size

A tiny p-value doesn't mean a huge effect!

**Small δ + huge n vs large δ + tiny n**

```

Effect Size vs P-value Comparison

Scenario 1: Small Effect, Large Sample
P-value: 0.0264
Effect size: 0.10

Scenario 2: Large Effect, Small Sample
P-value: 0.0477
Effect size: 0.71
```

Tiny shift, big n

Means differ by only 1 unit but n=1000. The huge sample drives the p-value down despite a trivial real-world effect.

Big shift, tiny n

Means differ by 10 units but n=20. The large effect is real but the small sample produces a borderline p-value.

Effect size = mean difference ÷ SD

The ratio (Cohen's d-style) measures practical magnitude independently of sample size. Compare the two p-values against the two effect sizes.

![Effect Size Comparison](../../../.gitbook/assets/effect_size_comparison.png) _Figure 5: Comparison of small effect with large sample (left) vs large effect with small sample (right). This demonstrates how p-values can be misleading without considering effect size._

## Factors Affecting P-values

### 1. Sample Size

Larger samples can make tiny effects statistically significant.

**Fixed lift, varying n (grid of histograms)**

```

Sample Size Effect Demo
n=  20: p=0.1828 Not significant
n= 100: p=0.0227 Significant
n= 500: p=0.0022 Significant
n=1000: p=0.0000 Significant
```

Effect size held fixed

The real difference between groups stays at 0.2 SD across all four trials, only n changes.

Same shift, different n

Draw control and treatment with the fixed shift, then test. Watch the printed p-value: it drops as n grows from 20 → 1000 even though the underlying effect is unchanged.

![Sample Size Effect](../../../.gitbook/assets/sample_size_effect.png) _Figure 6: Effect of sample size on p-values. As sample size increases, the same effect size becomes more detectable (smaller p-value)._

#### Interactive: same effect, different n

The two groups in the left panel always differ by the same effect size (0.2 SD, small but real). Slide \\(n\\) from 10 to 2,000. Right panel: the resulting p-value plotted on a log scale, with the gold dot marking the current sample size. The red dashed line is the conventional \\(\alpha = 0.05\\) threshold.

**Takeaway:** at \\(n = 10\\) the p-value is huge, the effect is real but invisible. At \\(n = 2{,}000\\) the same effect produces \\(p < 0.001\\). The effect didn't get bigger; we just bought more statistical power.

### 2. Effect Size

Bigger differences are easier to detect.

### 3. Variability in Data

More consistent data makes effects easier to spot.

## Real-world Application: A/B Testing

### Website Conversion Rate Example

**Bernoulli arms → chi-square on a 2×2 table**

```

A/B Test Results
Control conversion: 10.4%
Treatment conversion: 11.7%
P-value: 0.3921
Decision: Keep current version
```

Bernoulli arms

Simulate binary conversion outcomes (0/1) for 1,000 visitors each in control (10%) and treatment (12%) groups.

Chi-square test

Build a 2×2 contingency table of success/failure counts and run `chi2_contingency` to test whether conversion rates differ.

Bar chart with error bars

Plot mean conversion rates with ±1 SE error bars and embed the p-value in the chart title.

Launch decision

Print rates, p-value, and a simple rule: launch if p < 0.05, otherwise keep the current version.

![A/B Test Results](../../../.gitbook/assets/ab_test_results.png) _Figure 7: A/B test results showing conversion rates for control and treatment groups with error bars._

## Type I error, Type II error, and statistical power

Every test can be wrong in two different ways. Naming them is the easiest way to think clearly about _why_ p-values alone aren't enough.

### The 2×2 of decisions

|                           | Reality: H₀ true (no effect)                             | Reality: H₀ false (real effect)                            |
| ------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| **You reject H₀**         | **Type I error** (false alarm), probability \\(\alpha\\) | Correct rejection, probability \\(1 - \beta\\) = **power** |
| **You fail to reject H₀** | Correct non-rejection, probability \\(1 - \alpha\\)      | **Type II error** (missed effect), probability \\(\beta\\) |

Three things to remember:

* **Type I error rate \\(\alpha\\)** is what you choose up front (usually 0.05). It's the long-run rate at which the test cries wolf when nothing is happening.
* **Type II error rate \\(\beta\\)** is harder to control because it depends on the true effect size, the sample size, and the noise. Smaller real effects are easier to miss.
* **Power = \\(1 - \beta\\)** is the probability of _catching_ a real effect of a given size. The conventional target is **80% power**.

### The trade-off

You can always lower \\(\alpha\\) (be stricter), but that **raises \\(\beta\\)**, you'll miss more real effects. The only way to lower _both_ simultaneously is to increase the sample size, which is why power calculations and sample-size planning matter.

### What drives power?

Power goes up when:

| Lever                                                         | Effect on power                                                 |
| ------------------------------------------------------------- | --------------------------------------------------------------- |
| **Effect size grows** (the real difference is bigger)         | ↑, easier to detect                                             |
| **Sample size grows**                                         | ↑, less noise around the estimate                               |
| **\\(\alpha\\) is more lenient** (e.g., 0.10 instead of 0.01) | ↑, easier to "cross the line," but you accept more false alarms |
| **Variability shrinks** (cleaner measurement)                 | ↑, same effect against less noise                               |

### Why this matters for reading results

* A **non-significant** p-value (p > α) does **not** mean "no effect." It might mean the test was underpowered. Always ask: _"What size of effect would this study have been able to detect?"_
* A **significant** result (p < α) doesn't mean the effect is large or important. It just means it's not zero given the noise.
* When designing a study, you usually fix three of {effect size, n, α, power} and solve for the fourth. The most common case: "I want 80% power to detect a 5% lift at α = 0.05, how many users do I need?"

### Interactive: Power vs sample size and effect size

Move the sliders to set the true effect size (in standardized units) and the per-group sample size. The widget shows the null and alternative distributions, shades the rejection region (Type I, in red) and the missed-detection region (Type II, in orange), and reports the resulting power.

**Try this:**

* Set effect size = 0.2 (a small effect). Slide \\(n\\) up. How big does \\(n\\) need to be before power crosses 80%?
* Set effect size = 0.5 (medium). Notice power crosses 80% much sooner, bigger effects are cheaper to detect.
* Set α = 0.01 (stricter). Power drops at every \\(n\\), the price of being more careful about false alarms.

## Best Practices for Using P-values

### 1. Set α Before Looking at Data

Avoid p-hacking by deciding your threshold in advance.

### 2. Consider Practical Significance

Statistical significance ≠ Practical importance.

### 3. Report Exact P-values

Don't just say "p < 0.05".

### 4. Use Multiple Testing Corrections

When performing multiple tests:

**Bonferroni on a batch of null t-tests**

\`\`\`

Multiple Testing Correction Original significant results: 2 Corrected significant results: 0

```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Twenty null tests</span>
    </div>
    <div class="code-callout__body">
      <p>Run 20 independent t-tests on pure noise; some will fall below 0.05 by chance alone (false positives).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-9" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Bonferroni correction</span>
    </div>
    <div class="code-callout__body">
      <p>Apply Bonferroni adjustment via <code>multipletests</code>, which multiplies each p-value by the number of tests to control family-wise error rate.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-21" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scatter comparison</span>
    </div>
    <div class="code-callout__body">
      <p>Plot raw and corrected p-values together against the α = 0.05 threshold to show how correction shrinks apparent significance.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-25" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Count comparison</span>
    </div>
    <div class="code-callout__body">
      <p>Print before-and-after counts of "significant" results to illustrate how Bonferroni reduces false discoveries.</p>
    </div>
  </div>
</aside>
</div>

![Multiple Testing Correction](assets/multiple_testing_correction.png)
*Figure 8: Effect of multiple testing correction. The Bonferroni method adjusts p-values to control for the increased chance of false positives when performing multiple tests.*

## Practice Questions

Try each question on your own first, then expand the answer to check.

**1.** A study finds p = 0.03. What does this mean in plain English?

<details>
<summary>Show answer</summary>

p = 0.03 means: **if the null hypothesis were true (i.e., no real effect), there's a 3% chance of seeing a result as extreme as ours, or more extreme, just by random sampling.**

That's small enough that most analysts would say "the data are surprising under the null, so we lean toward rejecting it."

What it does **not** mean:

- ❌ "There's a 3% chance the null hypothesis is true." (You can't flip the conditional like that, see the misconceptions section.)
- ❌ "There's a 97% chance our finding is real." (Same flip, same error.)
- ❌ "The effect is large." (p says nothing about size, it could be a tiny effect detected with a huge sample.)

Practical reading: it's evidence against the null, but you should still report the **effect size and confidence interval** to know whether the effect is meaningful.

</details>

**2.** Why might a study with n = 10,000 find "significant" results for tiny effects?

<details>
<summary>Show answer</summary>

The standard error shrinks like \\(1/\sqrt{n}\\), so test statistics like \\(t = \dfrac{\bar x - \mu_0}{s/\sqrt{n}}\\) get larger as \\(n\\) grows even if the actual difference \\(\bar x - \mu_0\\) is tiny. With \\(n = 10{,}000\\), almost any non-zero true effect is statistically detectable.

Concrete example: a true effect of 0.01 standard deviations is detectable at p < 0.05 once \\(n\\) is around 40,000. That tells you the effect is real, not zero, but a 0.01 SD difference is almost always meaningless in business or clinical terms.

Lesson: **always report effect size alongside p-values.** With huge samples, a p < 0.001 may be a triviality; with tiny samples, p = 0.20 may hide a real effect.

</details>

**3.** Your A/B test shows p = 0.04 but only a 0.1% increase in conversions. What should you do?

<details>
<summary>Show answer</summary>

**Don't ship blindly.** A statistically-significant 0.1% lift is probably not worth the cost of switching, and it might not even be real once you account for engineering, support, and ongoing maintenance trade-offs. Steps:

1. **Check the confidence interval, not just p.** A 95% CI of, say, [0.01%, 0.19%] tells you the lift could be near zero. If the CI includes a practically-meaningless region, the result is weak even if statistically significant.
2. **Compare against your minimum detectable effect (MDE).** Was 0.1% above what you considered worth pursuing when you designed the test? If not, this was an underpowered or over-powered test in the wrong direction.
3. **Look for novelty and seasonality effects.** A one-week test catching a Monday holiday or a launch announcement can show small lifts that disappear later.
4. **Consider business cost.** Engineering, QA, possible regression risk, and any user disruption all weigh against a 0.1% lift.
5. **Re-run with proper sizing** if you genuinely care about a 0.1% lift, you'll likely need many more users to confirm it stably.

The healthy default is "p < α is necessary but not sufficient, also need a meaningful effect size."

</details>

**4.** How would you explain p-values to a non-technical stakeholder?

<details>
<summary>Show answer</summary>

> "Imagine the boring outcome, the new design has *no* real effect on conversions. Even if that's true, sometimes by pure luck the test will show a difference. The p-value is just: 'how often would we see a difference this big purely by luck if nothing was actually going on?'
>
> A small p-value (say 0.03) means 'this would be quite surprising under "no effect", only happens 3% of the time by luck, so we lean toward "something real is happening."'
>
> A big p-value means 'this could easily be noise, we don't have enough evidence to claim something changed.'
>
> Two warnings:
>
> 1. A small p-value tells us 'real, not zero' but not '*how big*' the effect is. We need a separate number for that.
> 2. A big p-value doesn't *prove* nothing's happening, sometimes our test is just too small to spot a real effect."

</details>

**5.** When would you use a stricter significance level (e.g., 0.01 instead of 0.05)?

<details>
<summary>Show answer</summary>

Use a stricter \\(\alpha\\) when the cost of a false positive is high. Common cases:

| Setting | Why stricter |
|---|---|
| **Medical / drug trials** | A wrongly-approved drug can harm patients. Regulatory bodies often demand 0.01 or stricter. |
| **Genome-wide association studies** | You're testing millions of genetic variants. Without a tiny α (e.g., 5×10⁻⁸), you'd get thousands of false positives. |
| **Multiple A/B tests in parallel** | If you run 20 tests at α = 0.05, ~1 will be a false positive by chance. Use Bonferroni (α/20 = 0.0025) or FDR control. |
| **High-stakes business decisions** | E.g., shutting down a product line. Cost of acting on a false positive is huge. |
| **Physics, particle discovery** | The "5-sigma" rule (~3×10⁻⁷) is the standard for claiming new fundamental discoveries. |

You **shouldn't** make α stricter just because you want to be "extra sure", wider α tightens Type I but loosens Type II (you'll miss real effects). Match the threshold to the actual cost of each error type.

</details>

## Key Takeaways

1. P-values measure evidence against H₀
2. Small p-values don't mean large effects
3. Sample size strongly influences p-values
4. Statistical significance ≠ Practical significance
5. Correct for multiple testing
6. Always consider both statistical and practical importance
7. Visualize your data to better understand the results

## Gotchas

- **Choosing α after seeing the data**: setting your significance threshold to 0.05 (or any value) *after* computing the p-value invalidates the Type I error guarantee. Always fix α before you run the test; changing it post-hoc to make a result "significant" is p-hacking.
- **p = 0.049 is not meaningfully different from p = 0.051**: the binary significant/not-significant framing treats a bright line as though the underlying evidence is radically different on either side. Report exact p-values and consider the full context rather than the threshold alone.
- **A large p-value does not prove the null hypothesis**: "fail to reject H₀" means the data are consistent with no effect, not that no effect exists. Underpowered studies routinely produce large p-values even when a real effect is present.
- **Confusing p-value with the probability that H₀ is true**: the p-value is computed *assuming* H₀ is true; it cannot tell you how likely H₀ is. Bayesian posterior probabilities answer that different question.
- **Running multiple tests without correction inflates false-positive rate**: if you run 20 independent tests at α=0.05, you expect about one false positive by chance alone. Use Bonferroni or FDR corrections (as shown in the lesson's `multipletests` example) whenever you test several hypotheses.
- **Treating `ttest_ind` as the only option for comparing two groups**: `ttest_ind` assumes approximately normal data with independent observations. For paired measurements use `ttest_rel`; for non-normal data with small samples prefer `mannwhitneyu`; the wrong test can produce a misleading p-value silently.

## Next steps

- Continue to [Parameters and statistics](./parameters-statistics.md) to consolidate notation and estimation, then finish the submodule and move on to [Hypothesis testing (module 4.2)](../4.2-hypotheses-testing/README.md).

## Additional Resources

- [Interactive P-value Simulator](https://seeing-theory.brown.edu/frequentist-inference/index.html)
- [ASA Statement on P-values](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf)
- [Common P-value Mistakes](https://statisticsbyjim.com/hypothesis-testing/interpreting-p-values/)

Remember: P-values are just one tool in your statistical toolbox. Use them wisely, but don't rely on them exclusively!
```
