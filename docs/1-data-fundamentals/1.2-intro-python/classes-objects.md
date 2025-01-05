[Previous content remains the same...]

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

## Practice Exercises for Data Science 🎯

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

Happy coding! 🚀
