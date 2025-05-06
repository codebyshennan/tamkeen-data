# Experimental Design: Building a Strong Foundation

## Introduction

Think of experimental design as a recipe for scientific discovery. Just as a chef carefully plans each ingredient and step, we must thoughtfully plan our experiments to get reliable results. This guide will help you understand the key principles of good experimental design, with practical code, real-world examples, and best practices.

![Scientific Method Flowchart](assets/scientific_method.png)

---

## The Three Pillars of Experimental Design

### 1. Control

**Control** is about managing variables to isolate the effect we want to study. Without proper control, it's hard to know if your results are due to your treatment or something else.

- **Control groups** provide a baseline for comparison.
- **Controlled variables** reduce noise in our measurements.
- **Randomization** helps eliminate bias.

> **Real-world example:**
> In a clinical trial, the control group receives a placebo, while the treatment group receives the new drug. This helps you see if the drug actually works.

### 2. Randomization

Randomization is crucial for reducing bias. It ensures that each experimental unit has an equal chance of receiving any treatment, so that groups are comparable.

**Why randomize?**

- Prevents selection bias
- Balances unknown confounders
- Makes statistical tests valid

**Python Example: Randomly Assigning Groups**

Suppose you have 10 participants and want to assign them to two groups at random:

```python
import numpy as np

participants = [f'P{i+1}' for i in range(10)]
groups = ['Control', 'Treatment']

np.random.seed(42)  # For reproducibility
assignments = np.random.choice(groups, size=len(participants))
for p, g in zip(participants, assignments):
    print(f"{p} → {g}")
```

**What to look for:**

- Each participant is randomly assigned to a group.
- The split may not be exactly even, but over many experiments, it will average out.

> **Best Practice:**
> Always set a random seed when sharing code for reproducibility.

**Try it yourself:**

- How would you ensure an equal number of participants in each group?

### 3. Replication

Replication helps verify our findings. The more times you repeat an experiment, the more confident you can be in your results.

- **Multiple trials** increase confidence.
- **Sample size calculations** ensure adequate power.

![Power Analysis](assets/power_analysis.png)

> **Reflect:**
> Why is it risky to draw conclusions from a single experiment?

---

## Types of Experimental Designs

Experimental design is about how you assign treatments to units. Here are two common designs:

### 1. Completely Randomized Design (CRD)

A **Completely Randomized Design (CRD)** is the simplest setup. Every experimental unit (e.g., a plant, a person, a website visitor) has an equal chance of receiving any treatment. Use this when your units are similar and you want to avoid bias.

**Real-world example:**
Suppose you're testing two fertilizers (A and B) on 20 identical plants. You want to assign each plant to a fertilizer at random.

```python
import numpy as np

def completely_randomized_design(treatments, units):
    """
    Randomly assign treatments to experimental units.
    treatments: list of treatment labels (e.g., ['A', 'B'])
    units: list of unit identifiers (e.g., plant IDs)
    Returns a dict mapping unit to assigned treatment.
    """
    np.random.seed(42)
    assignments = np.random.choice(treatments, size=len(units))
    return dict(zip(units, assignments))

# Example usage:
units = [f'Plant_{i+1}' for i in range(20)]
treatments = ['A', 'B']
assignment = completely_randomized_design(treatments, units)
print(assignment)
```

**What to look for:**

- Each plant is randomly assigned to either fertilizer A or B.
- This helps ensure that any observed differences in plant growth are due to the fertilizer, not to pre-existing differences between plants.

> **Best Practice:**
> Use a random seed for reproducibility.

**Try it yourself:**

- How would you modify the code if you had three fertilizers instead of two?

### 2. Randomized Block Design (RBD)

A **Randomized Block Design (RBD)** is used when your experimental units can be grouped into blocks that are similar. Within each block, treatments are randomly assigned. This helps control for known sources of variation.

**Real-world example:**
Suppose you're testing two fertilizers on plants, but your garden has sunny and shady areas. You want to make sure both fertilizers are tested in both conditions.

```python
import numpy as np

def randomized_block_design(treatments, blocks):
    """
    Assign treatments within each block.
    treatments: list of treatments (e.g., ['A', 'B'])
    blocks: dict mapping block name to list of unit IDs
    Returns a dict mapping unit to assigned treatment.
    """
    np.random.seed(42)
    assignments = {}
    for block, units in blocks.items():
        block_assignments = np.random.permutation(treatments * (len(units)//len(treatments)))
        assignments.update(dict(zip(units, block_assignments)))
    return assignments

# Example usage:
blocks = {
    'Sunny': [f'Sunny_{i+1}' for i in range(6)],
    'Shady': [f'Shady_{i+1}' for i in range(6)]
}
treatments = ['A', 'B', 'A', 'B', 'A', 'B']  # 3 of each per block
assignment = randomized_block_design(['A', 'B'], blocks)
print(assignment)
```

**What to look for:**

- Each block (e.g., sunny or shady) gets a random mix of treatments.
- This controls for the effect of sunlight on plant growth.

> **Common Pitfall:**
> Forgetting to block for important sources of variation can lead to misleading results.

**Try it yourself:**

- How would you handle blocks of different sizes?

![Experimental Design Flowchart](assets/experimental_design_flowchart.png)

---

## Statistical Considerations

Choosing the right statistical test for your design is crucial. The test you use depends on your data type, number of groups, and design.

![Statistical Test Decision Tree](assets/statistical_test_tree.png)

> **Tip:**
> Always check the assumptions of your chosen test (e.g., normality, equal variances).

---

## Effect Size and Power

Understanding the relationship between sample size and effect size is key to planning a successful experiment. If your sample is too small, you might miss real effects. If it's too large, you might waste resources.

![Effect Sizes](assets/effect_sizes.png)

### Sample Size Determination

The required sample size for comparing two means can be calculated as:

\[
n = 2 \left( \frac{z_{\alpha/2} + z_{\beta}}{d} \right)^2
\]

where:

- \( z_{\alpha/2} \): critical value for significance level
- \( z_{\beta} \): critical value for desired power
- \( d \): effect size (Cohen's d)

![Sample Size Determination](assets/sample_size_determination.png)

> **Best Practice:**
> Use a power analysis tool (like G*Power or Python's `statsmodels`) to calculate sample size before you start collecting data.

**Reflect:**

- What happens if you run an experiment with too few samples?

---

## Common Mistakes to Avoid

1. **Insufficient sample size** – leads to low power and inconclusive results.
2. **Poor randomization** – introduces bias.
3. **Inadequate controls** – makes it hard to attribute effects to your treatment.
4. **Confounding variables** – can mask or mimic treatment effects.
5. **Measurement bias** – inaccurate or inconsistent measurements can ruin your study.

---

## Best Practices

### Planning Phase

1. Define clear objectives
2. Calculate required sample size
3. Identify potential confounders
4. Plan data collection methods

### Implementation Phase

1. Follow randomization protocols
2. Maintain consistent conditions
3. Document everything
4. Monitor for issues

### Analysis Phase

1. Check assumptions
2. Use appropriate tests
3. Consider effect sizes
4. Account for multiple testing

### Reporting Phase

1. Be transparent about methods
2. Report all relevant statistics
3. Acknowledge limitations
4. Share raw data when possible

---

## Additional Resources

1. Books:
   - "Design of Experiments" by R.A. Fisher
   - "Experimental Design for Biologists" by David J. Glass

2. Online Tools:
   - G*Power for sample size calculations
   - R's experimental design packages
   - Python's statsmodels library

3. Software:
   - SAS JMP for design of experiments
   - Minitab for industrial experiments
   - Python's scipy for statistical analysis
