# Quiz: Inferential Statistics

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

Try each question closed-book first. Click **Show hint** if you get stuck, hints point you at the relevant lesson section and how to think about the question, without naming the answer.

## Questions

1. A researcher wants to understand the average income of all adults in a country. She surveys 2,000 people chosen at random. What is the "population" in this study?

- [ ] The 2,000 people surveyed
- [ ] All adults in the country
- [ ] The average income figure the researcher computes
- [ ] All countries in the world

<details>
<summary>Show hint</summary>

- **Where:** [Population vs Sample](../population-sample.md), "What is a Population?"
- **Think:** Ask which group the researcher wants to draw conclusions about, the people she measured, or the much larger group she cannot fully measure?

</details>

2. Which of the following is a statistic (not a parameter)?

- [ ] The true average height of all humans on Earth
- [ ] The proportion of voters who support a candidate across an entire country
- [ ] The mean salary computed from a sample of 500 employees
- [ ] The exact standard deviation of all transactions ever processed by a bank

<details>
<summary>Show hint</summary>

- **Where:** [Population vs Sample](../population-sample.md), "Key terms".
- **Think:** A parameter describes the whole population (you usually cannot measure it directly); a statistic is computed from the data you actually collected. Which option comes from a subset?

</details>

3. A quality inspector picks every 20th item off an assembly line to test. What sampling method is this?

- [ ] Simple random sampling
- [ ] Stratified sampling
- [ ] Systematic sampling
- [ ] Cluster sampling

<details>
<summary>Show hint</summary>

- **Where:** [Population vs Sample](../population-sample.md), "Sampling Methods: Choosing Your Strategy".
- **Think:** The key feature here is a fixed interval or skip pattern applied to a list. Which method is defined by that regular spacing?

</details>

4. A hospital researcher divides patients into age groups (under 40, 40-60, over 60) and randomly selects patients from each group. This is an example of:

- [ ] Cluster sampling
- [ ] Systematic sampling
- [ ] Stratified sampling
- [ ] Convenience sampling

<details>
<summary>Show hint</summary>

- **Where:** [Population vs Sample](../population-sample.md), "Stratified Sampling".
- **Think:** This method guarantees representation from each defined sub-group. The population is divided into groups first, then a random draw happens within each group.

</details>

5. As sample size increases, what happens to the standard error of the mean?

- [ ] It increases proportionally to n
- [ ] It stays the same regardless of n
- [ ] It decreases proportionally to 1/√n
- [ ] It decreases proportionally to 1/n

<details>
<summary>Show hint</summary>

- **Where:** [Sampling Distributions](../sampling-distributions.md), "Standard Error: Measuring the Spread".
- **Think:** The formula for SE of the mean has n in the denominator, inside a square root. What happens to 1/√n as n grows large?

</details>

6. The Central Limit Theorem (CLT) states that the sampling distribution of the mean:

- [ ] Matches the shape of the population distribution regardless of sample size
- [ ] Is approximately normal for sufficiently large samples, regardless of population shape
- [ ] Is only normal when the population is normally distributed
- [ ] Becomes uniform as sample size increases

<details>
<summary>Show hint</summary>

- **Where:** [Sampling Distributions](../sampling-distributions.md), "The Central Limit Theorem (CLT): Statistical Magic".
- **Think:** The power of the CLT is that it makes no requirement about the shape of the original data. What does it say about the distribution of sample means as n grows?

</details>

7. A 95% confidence interval for mean daily sales is ($4,200, $4,800). Which interpretation is correct?

- [ ] 95% of all daily sales fall between $4,200 and $4,800
- [ ] There is a 95% probability that today's sales will be in this range
- [ ] 95% of intervals built by this procedure will contain the true mean
- [ ] The true mean is $4,500 with 95% certainty

<details>
<summary>Show hint</summary>

- **Where:** [Confidence Intervals](../confidence-intervals.md), "Common Misconceptions: What CIs Are NOT".
- **Think:** A frequentist confidence interval is a statement about the procedure, not about any single interval. The interval either contains the truth or it doesn't; the 95% is a long-run property of the method.

</details>

8. If you increase the confidence level from 95% to 99% while keeping sample size fixed, what happens to the confidence interval?

- [ ] It becomes narrower
- [ ] It becomes wider
- [ ] It stays exactly the same width
- [ ] It shifts its center but keeps the same width

<details>
<summary>Show hint</summary>

- **Where:** [Confidence Intervals](../confidence-intervals.md), "Confidence Level Effect".
- **Think:** A higher confidence level requires a larger critical value (the t or z multiplier). Larger multiplier × the same standard error = what kind of margin of error?

</details>

9. A study has a sample of size n = 16. To cut the confidence interval width in half, the researcher would need to increase the sample size to approximately:

- [ ] 32
- [ ] 48
- [ ] 64
- [ ] 128

<details>
<summary>Show hint</summary>

- **Where:** [Confidence Intervals](../confidence-intervals.md), "Sample Size Effect" and [Sampling Distributions](../sampling-distributions.md), "Standard Error: Measuring the Spread".
- **Think:** CI width is proportional to SE = σ/√n. To halve the width, you need to halve the SE. If SE halves, what must happen to n? Apply the square-root relationship.

</details>

10. A p-value of 0.03 at significance level α = 0.05 means:

- [ ] There is a 3% chance the null hypothesis is true
- [ ] There is a 97% chance the alternative hypothesis is true
- [ ] Results this extreme would occur 3% of the time if the null hypothesis were true
- [ ] The effect size is 3% of the expected value

<details>
<summary>Show hint</summary>

- **Where:** [P-values](../p-values.md), "What is a P-value?" and "What p-values do NOT tell you".
- **Think:** The p-value is computed assuming H₀ is true, so it cannot tell you the probability that H₀ is true. It answers: given H₀ is true, how surprising is our result?

</details>

11. A researcher runs 20 independent hypothesis tests at α = 0.05, all on null effects (no real differences exist). On average, how many false positives should they expect?

- [ ] 0
- [ ] 1
- [ ] 5
- [ ] 20

<details>
<summary>Show hint</summary>

- **Where:** [P-values](../p-values.md), "Use Multiple Testing Corrections".
- **Think:** Each test has a false positive rate of α = 0.05 when the null is true. Multiply the number of tests by that rate.

</details>

12. In hypothesis testing, a Type II error occurs when:

- [ ] You reject H₀ when H₀ is actually true
- [ ] You fail to reject H₀ when H₀ is actually false
- [ ] You set α too low before collecting data
- [ ] You compute the wrong test statistic

<details>
<summary>Show hint</summary>

- **Where:** [P-values](../p-values.md), "Type I error, Type II error, and statistical power".
- **Think:** The 2×2 decision table has four cells. Type I and Type II label two of the "wrong" cells. Type I is a false alarm; Type II is the other kind of mistake, missing something that is actually there.

</details>
