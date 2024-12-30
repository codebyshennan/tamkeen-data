# Experimental Design: Building the Foundation for Scientific Discovery 🔬

## Introduction: Why Experimental Design Matters 🎯
Imagine you're a chef developing a new recipe. You wouldn't randomly throw ingredients together and hope for the best! Similarly, good experimental design helps us systematically test ideas and draw valid conclusions. Whether you're optimizing a website, developing a new drug, or studying customer behavior, proper experimental design is your roadmap to reliable results.

## The Three Pillars of Experimental Design 🏛️

### 1. Control: Managing Variables 🎮
Control is about isolating the effect you want to study. Think of it as a scientific detective work - you want to catch the true culprit (effect) without being misled by other suspects (confounding variables)!

\`\`\`python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def demonstrate_control():
    """
    Demonstrate the importance of controlling variables
    using a website A/B test example
    """
    np.random.seed(42)
    
    # Morning traffic (higher baseline)
    morning_control = np.random.normal(100, 15, 100)
    morning_treatment = np.random.normal(110, 15, 100)
    
    # Evening traffic (lower baseline)
    evening_control = np.random.normal(80, 15, 100)
    evening_treatment = np.random.normal(90, 15, 100)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot morning data
    plt.subplot(121)
    sns.boxplot(data=[morning_control, morning_treatment])
    plt.title("Morning Traffic")
    plt.xticks([0, 1], ['Control', 'Treatment'])
    plt.ylabel('Visitors')
    
    # Plot evening data
    plt.subplot(122)
    sns.boxplot(data=[evening_control, evening_treatment])
    plt.title("Evening Traffic")
    plt.xticks([0, 1], ['Control', 'Treatment'])
    
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/control_example.png')
    plt.close()
    
    # Statistical analysis
    results = {
        'Morning': stats.ttest_ind(morning_control, morning_treatment),
        'Evening': stats.ttest_ind(evening_control, evening_treatment),
        'Combined': stats.ttest_ind(
            np.concatenate([morning_control, evening_control]),
            np.concatenate([morning_treatment, evening_treatment])
        )
    }
    
    return results

# Example usage
results = demonstrate_control()
\`\`\`

### 2. Randomization: Eliminating Bias 🎲
Randomization is your shield against systematic bias. It's like shuffling a deck of cards - it ensures every unit has an equal chance of receiving any treatment.

\`\`\`python
def demonstrate_randomization():
    """
    Show the importance of random assignment
    """
    np.random.seed(42)
    
    def simulate_experiment(n_subjects=1000, random_assign=True):
        # Simulate subject characteristics
        age = np.random.normal(35, 10, n_subjects)
        tech_savvy = age < 30  # Younger users are more tech-savvy
        
        if random_assign:
            # Random assignment
            treatment = np.random.choice([0, 1], size=n_subjects)
        else:
            # Non-random: assign based on age
            treatment = (age < np.median(age)).astype(int)
        
        # Simulate outcome (affected by both treatment and tech-savviness)
        base_conversion = 0.1
        treatment_effect = 0.05
        tech_effect = 0.1
        
        conversion = np.random.binomial(1, 
            base_conversion + 
            treatment * treatment_effect +
            tech_savvy * tech_effect
        )
        
        return pd.DataFrame({
            'age': age,
            'tech_savvy': tech_savvy,
            'treatment': treatment,
            'conversion': conversion
        })
    
    # Compare random vs non-random assignment
    random_results = simulate_experiment(random_assign=True)
    nonrandom_results = simulate_experiment(random_assign=False)
    
    return {
        'random': random_results.groupby('treatment')['conversion'].mean(),
        'nonrandom': nonrandom_results.groupby('treatment')['conversion'].mean()
    }
\`\`\`

### 3. Replication: Ensuring Reliability 🔄
One successful experiment might be luck - replication helps confirm your findings are real and generalizable.

\`\`\`python
def demonstrate_replication(n_replications=5):
    """
    Show how replication helps establish reliability
    """
    np.random.seed(42)
    results = []
    
    for i in range(n_replications):
        # Simulate one experiment
        control = np.random.normal(100, 15, 100)
        treatment = np.random.normal(110, 15, 100)
        
        # Calculate statistics
        t_stat, p_value = stats.ttest_ind(control, treatment)
        effect_size = (np.mean(treatment) - np.mean(control)) / np.std(control)
        
        results.append({
            'replication': i + 1,
            'control_mean': np.mean(control),
            'treatment_mean': np.mean(treatment),
            'p_value': p_value,
            'effect_size': effect_size
        })
    
    # Create visualization
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(121)
    plt.scatter(results_df['replication'], results_df['effect_size'])
    plt.axhline(y=results_df['effect_size'].mean(), color='r', linestyle='--')
    plt.title('Effect Sizes Across Replications')
    plt.xlabel('Replication Number')
    plt.ylabel('Effect Size')
    
    plt.subplot(122)
    plt.scatter(results_df['replication'], -np.log10(results_df['p_value']))
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
    plt.title('Statistical Significance')
    plt.xlabel('Replication Number')
    plt.ylabel('-log10(p-value)')
    
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/replication_example.png')
    plt.close()
    
    return results_df
\`\`\`

## Types of Experimental Designs 📋

### 1. Completely Randomized Design (CRD) 🎲
The simplest design - like flipping a coin to assign treatments.

\`\`\`python
def completely_randomized_design(n_subjects=100, n_treatments=2):
    """
    Implement a completely randomized design
    """
    treatments = [f'Treatment_{i}' for i in range(n_treatments)]
    assignments = np.random.choice(treatments, size=n_subjects)
    
    # Visualize distribution
    plt.figure(figsize=(8, 5))
    pd.Series(assignments).value_counts().plot(kind='bar')
    plt.title('Treatment Assignment Distribution')
    plt.xlabel('Treatment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/crd_example.png')
    plt.close()
    
    return pd.DataFrame({
        'subject_id': range(n_subjects),
        'treatment': assignments
    })
\`\`\`

### 2. Randomized Block Design (RBD) 🏗️
Like organizing a tournament where teams are first grouped by skill level.

\`\`\`python
def randomized_block_design(subjects_data, treatments, block_variable):
    """
    Implement a randomized block design
    
    Parameters:
    -----------
    subjects_data : DataFrame
        Contains subject information including blocking variable
    treatments : list
        List of treatment names
    block_variable : str
        Name of the blocking variable in subjects_data
    """
    results = []
    
    for block in subjects_data[block_variable].unique():
        # Get subjects in this block
        block_subjects = subjects_data[subjects_data[block_variable] == block]
        
        # Randomly assign treatments within block
        block_treatments = np.random.choice(
            treatments,
            size=len(block_subjects),
            replace=True
        )
        
        block_results = pd.DataFrame({
            'subject_id': block_subjects.index,
            'block': block,
            'treatment': block_treatments
        })
        
        results.append(block_results)
    
    return pd.concat(results)
\`\`\`

### 3. Factorial Design 🔲
Testing multiple factors at once - like a chess game where you consider multiple moves.

\`\`\`python
def factorial_design(factors):
    """
    Create a full factorial design
    
    Parameters:
    -----------
    factors : dict
        Dictionary of factors and their levels
        Example: {'temperature': ['low', 'high'],
                 'pressure': ['low', 'medium', 'high']}
    """
    # Create all combinations
    design = pd.DataFrame(columns=factors.keys())
    
    # Add each combination
    for values in itertools.product(*factors.values()):
        design.loc[len(design)] = values
    
    # Visualize design
    plt.figure(figsize=(10, 6))
    for i, factor in enumerate(factors.keys()):
        plt.subplot(1, len(factors), i+1)
        design[factor].value_counts().plot(kind='bar')
        plt.title(f'{factor} Levels')
    
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/factorial_design.png')
    plt.close()
    
    return design
\`\`\`

## Power Analysis: Getting the Right Sample Size 📊

\`\`\`python
def power_analysis_demo():
    """
    Demonstrate power analysis for different effect sizes
    """
    from scipy import stats
    
    effect_sizes = np.linspace(0.1, 1.0, 10)
    sample_sizes = []
    
    for effect in effect_sizes:
        analysis = stats.TTestIndPower()
        n = analysis.solve_power(
            effect_size=effect,
            power=0.8,
            alpha=0.05,
            ratio=1.0
        )
        sample_sizes.append(n)
    
    # Visualize relationship
    plt.figure(figsize=(10, 6))
    plt.plot(effect_sizes, sample_sizes, 'b-')
    plt.fill_between(effect_sizes, sample_sizes, alpha=0.2)
    plt.title('Required Sample Size vs Effect Size')
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.ylabel('Required Sample Size per Group')
    plt.grid(True)
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/power_analysis.png')
    plt.close()
    
    return pd.DataFrame({
        'effect_size': effect_sizes,
        'sample_size': sample_sizes
    })
\`\`\`

## Common Pitfalls and Solutions ⚠️

### 1. Selection Bias 🎯
- ❌ Problem: Non-random sample selection
- ✅ Solution: Proper randomization techniques
\`\`\`python
# Example: Random sampling with stratification
def stratified_sample(data, strata, size=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    sampled = data.groupby(strata).apply(
        lambda x: x.sample(n=int(np.rint(size * len(x)/len(data))))
    )
    return sampled.reset_index(drop=True)
\`\`\`

### 2. Confounding Variables 🔄
- ❌ Problem: Hidden variables affecting results
- ✅ Solution: Control, blocking, or measurement
\`\`\`python
# Example: Checking for confounders
def check_confounders(data, treatment_col, outcome_col, potential_confounders):
    results = {}
    for confounder in potential_confounders:
        # Check association with treatment
        treat_assoc = stats.chi2_contingency(
            pd.crosstab(data[treatment_col], data[confounder])
        )[1]
        
        # Check association with outcome
        out_assoc = stats.chi2_contingency(
            pd.crosstab(data[outcome_col], data[confounder])
        )[1]
        
        results[confounder] = {
            'treatment_association_p': treat_assoc,
            'outcome_association_p': out_assoc
        }
    
    return pd.DataFrame(results).T
\`\`\`

### 3. Hawthorne Effect 👀
- ❌ Problem: Behavior changes due to observation
- ✅ Solution: Blinding and control groups

## Best Practices for Success 🌟

### 1. Plan Ahead 📝
- Define clear objectives
- Identify variables
- Calculate sample size
- Document everything

### 2. Monitor Quality 📊
\`\`\`python
class ExperimentMonitor:
    def __init__(self, name):
        self.name = name
        self.start_time = pd.Timestamp.now()
        self.log = []
        self.metrics = {}
    
    def log_event(self, event_type, details):
        self.log.append({
            'timestamp': pd.Timestamp.now(),
            'event_type': event_type,
            'details': details
        })
    
    def add_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_summary(self):
        return {
            'duration': pd.Timestamp.now() - self.start_time,
            'n_events': len(self.log),
            'metrics': {k: np.mean(v) for k, v in self.metrics.items()}
        }
\`\`\`

### 3. Analyze Thoroughly 📈
- Check assumptions
- Use appropriate tests
- Consider practical significance

## Practice Questions 🤔
1. A company wants to test three different website designs. What experimental design would you recommend and why?
2. How would you control for time-of-day effects in an online experiment?
3. Calculate the required sample size for detecting a medium effect size (d=0.5) with 80% power.
4. Design a blocked experiment for testing a new teaching method across different grade levels.
5. How would you handle missing data in a randomized experiment?

## Key Takeaways 🎯
1. 📋 Good design is crucial for valid results
2. 🎲 Randomization eliminates systematic bias
3. 📊 Power analysis ensures adequate sample size
4. ⚖️ Control variables for clean comparisons
5. 🔄 Replication validates findings

## Additional Resources 📚
- [Design of Experiments Guide](https://online.stat.psu.edu/stat503/)
- [Power Analysis Calculator](https://www.statmethods.net/stats/power.html)
- [Experimental Design Patterns](https://www.sciencedirect.com/topics/mathematics/experimental-design)

Remember: A well-designed experiment is like a well-built house - it needs a solid foundation to stand the test of time! 🏗️
