# Probability Fundamentals with Python 🎲

## Understanding Probability Through Code

{% stepper %}
{% step %}
### Implementing Basic Probability
Let's explore probability concepts using Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

class ProbabilityExperiment:
    """Simulate and analyze probability experiments"""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize experiment"""
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def flip_coin(self, n_flips: int) -> Dict[str, float]:
        """
        Simulate coin flips
        
        Args:
            n_flips: Number of flips
            
        Returns:
            Dictionary of probabilities
        """
        flips = np.random.choice(['H', 'T'], size=n_flips)
        counts = pd.Series(flips).value_counts()
        probs = counts / n_flips
        
        return {
            'heads_prob': probs.get('H', 0),
            'tails_prob': probs.get('T', 0)
        }
    
    def roll_dice(self, n_rolls: int) -> pd.Series:
        """
        Simulate dice rolls
        
        Args:
            n_rolls: Number of rolls
            
        Returns:
            Series with probabilities for each outcome
        """
        rolls = np.random.randint(1, 7, size=n_rolls)
        return pd.Series(rolls).value_counts() / n_rolls
    
    def plot_results(
        self,
        data: Union[Dict[str, float], pd.Series],
        title: str
    ) -> None:
        """Plot probability results"""
        plt.figure(figsize=(10, 6))
        
        if isinstance(data, dict):
            plt.bar(data.keys(), data.values())
        else:
            data.plot(kind='bar')
        
        plt.title(title)
        plt.ylabel('Probability')
        plt.axhline(y=1/len(data), color='r', linestyle='--',
                   label='Theoretical Probability')
        plt.legend()
        plt.show()

# Example usage
experiment = ProbabilityExperiment(random_seed=42)

# Simulate coin flips
coin_results = experiment.flip_coin(1000)
print("\nCoin Flip Probabilities:")
for outcome, prob in coin_results.items():
    print(f"{outcome}: {prob:.3f}")

experiment.plot_results(
    coin_results,
    "Coin Flip Probabilities (1000 flips)"
)

# Simulate dice rolls
dice_results = experiment.roll_dice(1000)
print("\nDice Roll Probabilities:")
print(dice_results)

experiment.plot_results(
    dice_results,
    "Dice Roll Probabilities (1000 rolls)"
)
```
{% endstep %}

{% step %}
### Monte Carlo Simulation
Let's use simulation to understand probability:

```python
class MonteCarloSimulation:
    """Perform Monte Carlo simulations for probability problems"""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
    
    def birthday_problem(
        self,
        n_people: int
    ) -> float:
        """
        Simulate birthday problem
        
        Args:
            n_people: Number of people in room
            
        Returns:
            Probability of birthday match
        """
        matches = 0
        
        for _ in range(self.n_simulations):
            # Generate random birthdays
            birthdays = np.random.randint(0, 365, n_people)
            # Check for matches
            if len(birthdays) != len(set(birthdays)):
                matches += 1
        
        return matches / self.n_simulations
    
    def monty_hall(
        self,
        switch: bool = True
    ) -> float:
        """
        Simulate Monty Hall problem
        
        Args:
            switch: Whether to switch doors
            
        Returns:
            Probability of winning
        """
        wins = 0
        
        for _ in range(self.n_simulations):
            # Set up doors
            doors = [1, 2, 3]
            prize_door = np.random.choice(doors)
            chosen_door = np.random.choice(doors)
            
            # Host opens a door
            remaining_doors = [
                d for d in doors
                if d != chosen_door and d != prize_door
            ]
            opened_door = np.random.choice(remaining_doors)
            
            if switch:
                # Switch to other unopened door
                final_door = [
                    d for d in doors
                    if d != chosen_door and d != opened_door
                ][0]
            else:
                final_door = chosen_door
            
            if final_door == prize_door:
                wins += 1
        
        return wins / self.n_simulations
    
    def plot_birthday_probabilities(
        self,
        max_people: int = 50
    ) -> None:
        """Plot birthday problem probabilities"""
        people = range(2, max_people + 1)
        probs = [
            self.birthday_problem(n)
            for n in people
        ]
        
        plt.figure(figsize=(12, 6))
        plt.plot(people, probs, marker='o')
        plt.axhline(y=0.5, color='r', linestyle='--',
                   label='50% Probability')
        plt.xlabel('Number of People')
        plt.ylabel('Probability of Match')
        plt.title('Birthday Problem Probability')
        plt.grid(True)
        plt.legend()
        plt.show()

# Example usage
mc = MonteCarloSimulation(n_simulations=10000)

# Birthday problem
print("\nBirthday Problem:")
prob_23 = mc.birthday_problem(23)
print(f"Probability with 23 people: {prob_23:.3f}")

mc.plot_birthday_probabilities()

# Monty Hall problem
print("\nMonty Hall Problem:")
prob_switch = mc.monty_hall(switch=True)
prob_stay = mc.monty_hall(switch=False)
print(f"Probability when switching: {prob_switch:.3f}")
print(f"Probability when staying: {prob_stay:.3f}")
```
{% endstep %}
{% endstepper %}

## Probability Rules and Calculations

{% stepper %}
{% step %}
### Implementing Probability Rules
Let's create tools for probability calculations:

```python
class ProbabilityCalculator:
    """Calculate probabilities using various rules"""
    
    @staticmethod
    def complement(p: float) -> float:
        """Calculate complement probability"""
        return 1 - p
    
    @staticmethod
    def union_independent(
        p1: float,
        p2: float
    ) -> float:
        """Calculate union of independent events"""
        return p1 + p2 - (p1 * p2)
    
    @staticmethod
    def intersection_independent(
        p1: float,
        p2: float
    ) -> float:
        """Calculate intersection of independent events"""
        return p1 * p2
    
    @staticmethod
    def conditional_probability(
        p_intersection: float,
        p_given: float
    ) -> float:
        """Calculate conditional probability"""
        return p_intersection / p_given
    
    @staticmethod
    def bayes_theorem(
        p_a: float,
        p_b_given_a: float,
        p_b_given_not_a: float
    ) -> float:
        """
        Calculate probability using Bayes' theorem
        
        P(A|B) = P(B|A) * P(A) / P(B)
        """
        p_not_a = 1 - p_a
        p_b = (p_b_given_a * p_a) + (p_b_given_not_a * p_not_a)
        return (p_b_given_a * p_a) / p_b

# Example usage
calc = ProbabilityCalculator()

# Medical test example
p_disease = 0.01  # 1% have disease
p_positive_given_disease = 0.95  # 95% accuracy for sick
p_positive_given_healthy = 0.10  # 10% false positive

p_disease_given_positive = calc.bayes_theorem(
    p_disease,
    p_positive_given_disease,
    p_positive_given_healthy
)

print("\nMedical Test Example:")
print(f"Probability of disease given positive test: "
      f"{p_disease_given_positive:.3f}")
```
{% endstep %}

{% step %}
### Visualizing Probability Concepts
Create visual representations of probability:

```python
class ProbabilityVisualizer:
    """Visualize probability concepts"""
    
    @staticmethod
    def plot_venn2(
        set_a: set,
        set_b: set,
        labels: Tuple[str, str]
    ) -> None:
        """Plot two-set Venn diagram"""
        from matplotlib_venn import venn2
        
        plt.figure(figsize=(10, 6))
        venn2([set_a, set_b], labels)
        plt.title("Venn Diagram")
        plt.show()
        
        # Calculate probabilities
        union = len(set_a.union(set_b))
        intersection = len(set_a.intersection(set_b))
        
        print("\nProbabilities:")
        print(f"P(A): {len(set_a)/union:.3f}")
        print(f"P(B): {len(set_b)/union:.3f}")
        print(f"P(A∩B): {intersection/union:.3f}")
        print(f"P(A∪B): {1.0:.3f}")
    
    @staticmethod
    def plot_probability_tree(
        probabilities: Dict[str, float],
        outcomes: Dict[str, List[str]]
    ) -> None:
        """Plot probability tree diagram"""
        import networkx as nx
        
        G = nx.DiGraph()
        pos = {}
        
        # Add nodes and edges
        G.add_node("Start", pos=(0, 0))
        
        for i, (event, prob) in enumerate(probabilities.items()):
            # First level
            G.add_node(event, pos=(1, -i))
            G.add_edge("Start", event, probability=f"P={prob:.2f}")
            
            # Second level
            for j, outcome in enumerate(outcomes[event]):
                node_name = f"{event}_{outcome}"
                G.add_node(node_name, pos=(2, -i-j*0.5))
                G.add_edge(event, node_name,
                          probability=f"P={1/len(outcomes[event]):.2f}")
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, arrowsize=20)
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'probability')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Probability Tree Diagram")
        plt.axis('off')
        plt.show()

# Example usage
viz = ProbabilityVisualizer()

# Venn diagram example
students_math = {1, 2, 3, 4, 5, 7}
students_science = {2, 4, 6, 7, 8}
viz.plot_venn2(
    students_math,
    students_science,
    ('Math', 'Science')
)

# Probability tree example
weather_probs = {
    'Sunny': 0.7,
    'Rainy': 0.3
}
weather_outcomes = {
    'Sunny': ['Hot', 'Mild'],
    'Rainy': ['Mild', 'Cold']
}
viz.plot_probability_tree(weather_probs, weather_outcomes)
```
{% endstep %}
{% endstepper %}

## Advanced Probability Concepts

{% stepper %}
{% step %}
### Implementing Advanced Probability
Let's create tools for advanced probability analysis:

```python
class AdvancedProbability:
    """Advanced probability calculations and simulations"""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_random_walk(
        self,
        n_steps: int,
        n_simulations: int
    ) -> np.ndarray:
        """
        Simulate random walks
        
        Args:
            n_steps: Number of steps
            n_simulations: Number of simulations
            
        Returns:
            Array of random walk paths
        """
        steps = np.random.choice(
            [-1, 1],
            size=(n_simulations, n_steps)
        )
        paths = np.cumsum(steps, axis=1)
        return paths
    
    def calculate_hitting_probability(
        self,
        paths: np.ndarray,
        threshold: int
    ) -> float:
        """Calculate probability of hitting threshold"""
        hit_any = np.any(np.abs(paths) >= threshold, axis=1)
        return np.mean(hit_any)
    
    def plot_random_walks(
        self,
        paths: np.ndarray,
        threshold: Optional[int] = None
    ) -> None:
        """Plot random walk paths"""
        plt.figure(figsize=(12, 6))
        
        # Plot paths
        for path in paths:
            plt.plot(path, alpha=0.1, color='blue')
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        plt.plot(mean_path, color='red', linewidth=2,
                label='Mean Path')
        
        # Add threshold lines if specified
        if threshold is not None:
            plt.axhline(y=threshold, color='g', linestyle='--',
                       label=f'+{threshold}')
            plt.axhline(y=-threshold, color='g', linestyle='--',
                       label=f'-{threshold}')
        
        plt.title('Random Walk Simulations')
        plt.xlabel('Step')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
ap = AdvancedProbability(random_seed=42)

# Simulate random walks
n_steps = 100
n_sims = 1000
threshold = 10

paths = ap.simulate_random_walk(n_steps, n_sims)
hit_prob = ap.calculate_hitting_probability(paths, threshold)

print(f"\nProbability of hitting ±{threshold}: {hit_prob:.3f}")

# Plot results
ap.plot_random_walks(paths[:100], threshold)  # Plot first 100 paths
```
{% endstep %}

{% step %}
### Probability in Machine Learning
Example of using probability in ML contexts:

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

class ProbabilisticClassifier:
    """Demonstrate probability in classification"""
    
    def __init__(self):
        self.model = GaussianNB()
    
    def create_sample_data(
        self,
        n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sample dataset"""
        # Generate features
        X = np.random.randn(n_samples, 2)
        
        # Generate labels based on probability
        probs = 1 / (1 + np.exp(-X.sum(axis=1)))
        y = np.random.binomial(n=1, p=probs)
        
        return X, y
    
    def fit_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Fit model and evaluate results"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'predictions': y_pred,
            'probabilities': y_prob,
            'confusion_matrix': cm,
            'true_labels': y_test
        }
    
    def plot_results(
        self,
        results: Dict[str, Any]
    ) -> None:
        """Plot classification results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot confusion matrix
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            ax=ax1
        )
        ax1.set_title('Confusion Matrix')
        
        # Plot probability distribution
        probs_class1 = results['probabilities'][:, 1]
        sns.histplot(
            probs_class1,
            bins=30,
            ax=ax2
        )
        ax2.set_title('Probability Distribution (Class 1)')
        ax2.set_xlabel('Predicted Probability')
        
        plt.tight_layout()
        plt.show()

# Example usage
pc = ProbabilisticClassifier()

# Create and analyze data
X, y = pc.create_sample_data()
results = pc.fit_and_evaluate(X, y)

# Plot results
pc.plot_results(results)
```
{% endstep %}
{% endstepper %}

## Practice Exercises 🎯

Try these probability programming exercises:

1. **Card Game Simulator**
   ```python
   # Create a simulator that:
   # - Deals cards and calculates probabilities
   # - Simulates different poker hands
   # - Visualizes results
   ```

2. **Disease Testing Model**
   ```python
   # Implement a system that:
   # - Simulates medical test accuracy
   # - Calculates false positive/negative rates
   # - Uses Bayes' theorem for diagnosis
   ```

3. **Stock Market Probability**
   ```python
   # Build analysis tools for:
   # - Calculating probability of price movements
   # - Simulating trading strategies
   # - Risk assessment
   ```

Remember:
- Use NumPy for efficient calculations
- Implement proper error handling
- Validate probability assumptions
- Create clear visualizations
- Document your code

Happy coding! 🚀
