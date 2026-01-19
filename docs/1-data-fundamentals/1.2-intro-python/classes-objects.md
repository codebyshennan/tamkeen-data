# Python Classes and Objects in Data Science

> **🎨 Visualize OOP:** Python Tutor can show object creation and method calls!

> **🤖 AI Helper:** "Explain classes using real-world objects as examples"

> **📓 Interactive:** Practice OOP concepts in your own Colab notebooks

## Introduction to Object-Oriented Programming

### Core OOP Concepts

1. **Encapsulation**: Bundling data and methods that operate on that data
2. **Inheritance**: Creating new classes based on existing ones
3. **Polymorphism**: Using a single interface for different data types
4. **Abstraction**: Hiding complex implementation details

### Why OOP in Data Science?

- Modular and reusable code
- Maintainable data pipelines
- Scalable machine learning systems
- Consistent interfaces

## Design Patterns in Data Science

### 1. Factory Pattern

```python
from abc import ABC, abstractmethod
from typing import Dict, Type

class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

class RandomForestModel(Model):
    def train(self, X, y):
        print("Training Random Forest")
    
    def predict(self, X):
        print("Predicting with Random Forest")

class XGBoostModel(Model):
    def train(self, X, y):
        print("Training XGBoost")
    
    def predict(self, X):
        print("Predicting with XGBoost")

class ModelFactory:
    _models: Dict[str, Type[Model]] = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel
    }
    
    @classmethod
    def create_model(cls, model_type: str) -> Model:
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type]()
```

### 2. Strategy Pattern

```python
from typing import Protocol, Dict, Any

class FeatureEngineeringStrategy(Protocol):
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

class DateFeatures:
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in data.select_dtypes('datetime64'):
            data[f'{col}_year'] = data[col].dt.year
            data[f'{col}_month'] = data[col].dt.month
        return data

class TextFeatures:
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in data.select_dtypes('object'):
            data[f'{col}_length'] = data[col].str.len()
            data[f'{col}_word_count'] = data[col].str.split().str.len()
        return data

class FeatureEngineer:
    def __init__(self, strategies: List[FeatureEngineeringStrategy]):
        self.strategies = strategies
    
    def apply_all(self, data: pd.DataFrame) -> pd.DataFrame:
        for strategy in self.strategies:
            data = strategy.engineer_features(data)
        return data
```

### 3. Observer Pattern

```python
from typing import List, Protocol
from dataclasses import dataclass
from datetime import datetime

class ModelObserver(Protocol):
    def update(self, metrics: Dict[str, float]):
        ...

@dataclass
class ModelMetrics:
    timestamp: datetime
    metrics: Dict[str, float]

class MetricsLogger(ModelObserver):
    def __init__(self):
        self.history: List[ModelMetrics] = []
    
    def update(self, metrics: Dict[str, float]):
        self.history.append(
            ModelMetrics(datetime.now(), metrics)
        )

class AlertSystem(ModelObserver):
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def update(self, metrics: Dict[str, float]):
        if metrics.get('error', 0) > self.threshold:
            print(f"Alert: Model error {metrics['error']} "
                  f"exceeded threshold {self.threshold}")

class ObservableModel:
    def __init__(self):
        self._observers: List[ModelObserver] = []
    
    def attach(self, observer: ModelObserver):
        self._observers.append(observer)
    
    def notify(self, metrics: Dict[str, float]):
        for observer in self._observers:
            observer.update(metrics)
```

## Testing and Debugging

### Unit Testing

```python
import unittest
from typing import List, Dict

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': ['A', None, 'B', 'A']
        })
        
        self.pipeline = MLPipeline([
            MissingValueImputer(),
            FeatureScaler()
        ])
    
    def test_missing_value_imputation(self):
        result = self.pipeline.transform(self.sample_data)
        self.assertFalse(result.isnull().any().any())
    
    def test_feature_scaling(self):
        result = self.pipeline.transform(self.sample_data)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertAlmostEqual(result[col].mean(), 0, places=2)
            self.assertAlmostEqual(result[col].std(), 1, places=2)

if __name__ == '__main__':
    unittest.main()
```

### Debugging Tips

1. **Use Logging Effectively**

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DebuggableTransformer(BaseTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting transformation on {X.shape} data")
        try:
            result = self._transform_implementation(X)
            logger.info("Transformation successful")
            return result
        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            raise
```

2. **Data Validation**

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DataValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class DataValidator:
    def validate(self, data: pd.DataFrame) -> DataValidationResult:
        errors = []
        warnings = []
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            warnings.append(
                f"Missing values found in columns: "
                f"{missing[missing > 0].index.tolist()}"
            )
        
        # Check data types
        if not all(data.select_dtypes(include=[np.number]).columns):
            errors.append("Non-numeric data found in feature columns")
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

## Error Handling Best Practices

### 1. Custom Exceptions

```python
class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass

class DataValidationError(PipelineError):
    """Raised when data validation fails"""
    pass

class ModelError(PipelineError):
    """Raised when model operations fail"""
    pass

class TransformerError(PipelineError):
    """Raised when transformer operations fail"""
    def __init__(self, transformer_name: str, message: str):
        self.transformer_name = transformer_name
        super().__init__(f"{transformer_name}: {message}")
```

### 2. Graceful Error Handling

```python
class RobustPipeline:
    def __init__(self, steps: List[BaseTransformer]):
        self.steps = steps
        self.errors: List[Dict] = []
    
    def process(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            # Validate input data
            validation_result = DataValidator().validate(data)
            if not validation_result.is_valid:
                raise DataValidationError(
                    f"Validation failed: {validation_result.errors}"
                )
            
            # Process each step
            current_data = data
            for step in self.steps:
                try:
                    current_data = step.transform(current_data)
                except Exception as e:
                    self.errors.append({
                        'step': step.__class__.__name__,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    raise TransformerError(
                        step.__class__.__name__, str(e)
                    )
            
            return current_data
            
        except Exception as e:
            self.errors.append({
                'step': 'pipeline',
                'error': str(e),
                'timestamp': datetime.now()
            })
            raise PipelineError(f"Pipeline failed: {str(e)}")
```

## Performance Optimization

### 1. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable

class ParallelTransformer(BaseTransformer):
    def __init__(self, func: Callable, n_jobs: int = -1):
        self.func = func
        self.n_jobs = n_jobs
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Split data into chunks
        chunks = np.array_split(X, self.n_jobs)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(self.func, chunks))
        
        # Combine results
        return pd.concat(results)
```

### 2. Memory Optimization

```python
class MemoryEfficientPipeline:
    def __init__(self, steps: List[BaseTransformer]):
        self.steps = steps
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Process data in chunks
        chunk_size = 1000
        chunks = []
        
        for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
            # Process each chunk through pipeline
            for step in self.steps:
                chunk = step.transform(chunk)
            chunks.append(chunk)
        
        return pd.concat(chunks)
```

## Advanced Data Science Classes

{% stepper %}
{% step %}

### Machine Learning Pipeline

Example of a modular ML pipeline:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BaseTransformer(ABC):
    """Abstract base class for transformers"""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'BaseTransformer':
        """Fit transformer to data"""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        pass
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data"""
        return self.fit(X).transform(X)

class MissingValueImputer(BaseTransformer):
    """Handle missing values in dataset"""
    
    def __init__(
        self,
        numeric_strategy: str = 'mean',
        categorical_strategy: str = 'mode'
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_values = {}
    
    def fit(self, X: pd.DataFrame) -> 'MissingValueImputer':
        """Calculate fill values from data"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        # Calculate fill values for numeric columns
        for col in numeric_cols:
            if self.numeric_strategy == 'mean':
                self.fill_values[col] = X[col].mean()
            elif self.numeric_strategy == 'median':
                self.fill_values[col] = X[col].median()
        
        # Calculate fill values for categorical columns
        for col in categorical_cols:
            if self.categorical_strategy == 'mode':
                self.fill_values[col] = X[col].mode()[0]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values"""
        X = X.copy()
        for col, fill_value in self.fill_values.items():
            X[col] = X[col].fillna(fill_value)
        return X

class OutlierHandler(BaseTransformer):
    """Handle outliers in numeric columns"""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.bounds = {}
    
    def fit(self, X: pd.DataFrame) -> 'OutlierHandler':
        """Calculate outlier bounds"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = {
                'lower': Q1 - self.threshold * IQR,
                'upper': Q3 + self.threshold * IQR
            }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers"""
        X = X.copy()
        for col, bounds in self.bounds.items():
            mask = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
            X.loc[mask, col] = np.nan
        return X

class FeatureScaler(BaseTransformer):
    """Scale numeric features"""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scalers = {}
    
    def fit(self, X: pd.DataFrame) -> 'FeatureScaler':
        """Calculate scaling parameters"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.method == 'standard':
                self.scalers[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std()
                }
            elif self.method == 'minmax':
                self.scalers[col] = {
                    'min': X[col].min(),
                    'max': X[col].max()
                }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features"""
        X = X.copy()
        for col, scaler in self.scalers.items():
            if self.method == 'standard':
                X[col] = (X[col] - scaler['mean']) / scaler['std']
            elif self.method == 'minmax':
                X[col] = (X[col] - scaler['min']) / (
                    scaler['max'] - scaler['min']
                )
        return X

class MLPipeline:
    """Machine learning pipeline"""
    
    def __init__(
        self,
        transformers: List[BaseTransformer],
        model: Optional[BaseEstimator] = None
    ):
        self.transformers = transformers
        self.model = model
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'MLPipeline':
        """Fit pipeline to data"""
        data = X.copy()
        
        # Fit transformers
        for transformer in self.transformers:
            data = transformer.fit_transform(data)
        
        # Fit model if provided
        if self.model is not None and y is not None:
            self.model.fit(data, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through pipeline"""
        data = X.copy()
        
        # Apply transformations
        for transformer in self.transformers:
            data = transformer.transform(data)
        
        return data
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("No model provided")
        
        # Transform data and predict
        data = self.transform(X)
        return self.model.predict(data)

# Using the ML pipeline
from sklearn.ensemble import RandomForestClassifier

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, np.nan, 100, 35],
    'income': [50000, 60000, 75000, 1000000, 65000],
    'category': ['A', 'B', 'A', 'C', 'B'],
    'target': [0, 1, 1, 1, 0]
})

# Create pipeline
pipeline = MLPipeline(
    transformers=[
        MissingValueImputer(),
        OutlierHandler(threshold=2.0),
        FeatureScaler(method='standard')
    ],
    model=RandomForestClassifier(random_state=42)
)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Fit pipeline
pipeline.fit(X, y)

# Make predictions
predictions = pipeline.predict(X)
print("\nPredictions:", predictions)
```

{% endstep %}

{% step %}

### Data Pipeline Architecture

Example of a data processing pipeline:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataPipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data"""
        pass
    
    @abstractmethod
    def get_step_name(self) -> str:
        """Get step name"""
        pass

class DataLoader(DataPipelineStep):
    """Load data from various sources"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def process(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Load data from file"""
        if self.filepath.endswith('.csv'):
            return pd.read_csv(self.filepath)
        elif self.filepath.endswith('.parquet'):
            return pd.read_parquet(self.filepath)
        else:
            raise ValueError(f"Unsupported file type: {self.filepath}")
    
    def get_step_name(self) -> str:
        return "DataLoader"

class DataCleaner(DataPipelineStep):
    """Clean and preprocess data"""
    
    def __init__(
        self,
        drop_duplicates: bool = True,
        handle_missing: bool = True
    ):
        self.drop_duplicates = drop_duplicates
        self.handle_missing = handle_missing
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        df = data.copy()
        
        if self.drop_duplicates:
            df = df.drop_duplicates()
        
        if self.handle_missing:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(
                df[numeric_cols].mean()
            )
            
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            df[categorical_cols] = df[categorical_cols].fillna(
                df[categorical_cols].mode().iloc[0]
            )
        
        return df
    
    def get_step_name(self) -> str:
        return "DataCleaner"

class FeatureEngineer(DataPipelineStep):
    """Create new features"""
    
    def __init__(self, date_columns: List[str]):
        self.date_columns = date_columns
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features"""
        df = data.copy()
        
        for col in self.date_columns:
            df[col] = pd.to_datetime(df[col])
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        
        return df
    
    def get_step_name(self) -> str:
        return "FeatureEngineer"

class DataPipeline:
    """Data processing pipeline"""
    
    def __init__(self, steps: List[DataPipelineStep]):
        self.steps = steps
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run pipeline"""
        current_data = data
        start_time = datetime.now()
        
        self.logger.info("Starting pipeline execution")
        
        for step in self.steps:
            step_start = datetime.now()
            self.logger.info(f"Running step: {step.get_step_name()}")
            
            try:
                current_data = step.process(current_data)
                
                step_duration = datetime.now() - step_start
                self.logger.info(
                    f"Completed step: {step.get_step_name()} "
                    f"in {step_duration.total_seconds():.2f} seconds"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error in step {step.get_step_name()}: {str(e)}"
                )
                raise
        
        total_duration = datetime.now() - start_time
        self.logger.info(
            f"Pipeline completed in {total_duration.total_seconds():.2f} "
            "seconds"
        )
        
        return current_data

# Using the data pipeline
# Create pipeline steps
steps = [
    DataLoader('data.csv'),
    DataCleaner(drop_duplicates=True, handle_missing=True),
    FeatureEngineer(date_columns=['date'])
]

# Create and run pipeline
pipeline = DataPipeline(steps)
try:
    result = pipeline.run()
    print("\nPipeline execution successful")
    print("\nProcessed data shape:", result.shape)
except Exception as e:
    print(f"\nPipeline execution failed: {str(e)}")
```

{% endstep %}
{% endstepper %}

## Practice Exercises for Data Science

Try these advanced exercises:

1. **Create a Feature Selection System**

   ```python
   # Build classes for:
   # - Feature importance calculation
   # - Correlation analysis
   # - Feature selection based on metrics
   # - Feature ranking and visualization
   ```

2. **Implement a Model Evaluation Pipeline**

   ```python
   # Create classes for:
   # - Cross-validation
   # - Metric calculation
   # - Model comparison
   # - Results visualization
   ```

3. **Build an Automated Report Generator**

   ```python
   # Develop classes for:
   # - Data profiling
   # - Statistical analysis
   # - Visualization generation
   # - Report formatting
   ```

Remember:

- Use type hints for better code documentation
- Implement proper error handling
- Consider performance implications
- Write unit tests for your classes
- Follow SOLID principles

Happy coding!

## Additional Resources

1. **Books**
   - "Clean Code" by Robert C. Martin
   - "Design Patterns" by Gang of Four
   - "Python Patterns" by Brandon Rhodes

2. **Online Resources**
   - [Real Python OOP Tutorials](https://realpython.com/python3-object-oriented-programming/)
   - [Python Design Patterns](https://python-patterns.guide/)
   - [Scikit-learn Development Guide](https://scikit-learn.org/stable/developers/index.html)

3. **Tools**
   - [PyTest](https://docs.pytest.org/) for testing
   - [Black](https://github.com/psf/black) for code formatting
   - [Mypy](http://mypy-lang.org/) for type checking

Remember: "Clean code is not written by following a set of rules. You don't become a software craftsman by learning a list of heuristics. Professionalism and craftsmanship come from values that drive disciplines." - Robert C. Martin

---

## 🎯 Modern Learning Tips

### Use AI for OOP Learning
```
"Explain the difference between classes and objects using real-world examples"
"Show me when to use inheritance vs composition"
"Review my class design: [paste code]"
"Create practice exercises for OOP concepts"
```

### Visualize with Python Tutor
Perfect for visualizing:
- Object creation and initialization
- Method calls and `self`
- Inheritance relationships
- Instance vs class variables

### Debug with Modern Tools
- Use VS Code / Cursor debugger
- Set breakpoints in methods
- Inspect object attributes
- Step through method calls

> **📺 Video Help:** See [Video Resources](./video-resources.md) - OOP section for detailed tutorials

Happy coding!
