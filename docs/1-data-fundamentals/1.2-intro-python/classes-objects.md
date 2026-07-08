# Python Classes and Objects in Data Science

**After this lesson:** you can explain Python Classes and Objects in Data Science and try the examples in your own notebook.

> **Visualize OOP:** Python Tutor can show object creation and method calls!

> **AI Helper:** "Explain classes using real-world objects as examples"

> **Interactive:** Practice OOP concepts in your own Colab notebooks

### Video

_Corey Schafer, Python OOP: classes and instances_

## Introduction to object-oriented programming

**Object-oriented programming (OOP)** groups **state** (attributes) with **behavior** (methods) in **classes**. You build **instances** (objects) from those blueprints. Libraries you already use (`DataFrame`, estimators in scikit-learn) are OOP under the hood; learning classes helps you read their APIs and package your own helpers cleanly.

### Core OOP concepts

* **Encapsulation**: Keep related fields and functions together (a `Dataset` class that knows how to validate itself) instead of scattering dicts across files.
* **Inheritance**: Reuse and specialize behavior (`BaseModel` → `RegressionModel`) without duplicating every method.
* **Polymorphism**: Call the same method name on different types (`fit`, `transform`) and let each class implement the details.
* **Abstraction**: Expose a simple interface (`model.predict(X)`) while hiding optimization details inside the class.

_`BaseModel` defines the **interface** (`fit`, `predict`, `evaluate`); subclasses override the details. This is exactly how scikit-learn estimators are structured._

### Why OOP in data science?

Teams use classes for **reusable pipelines**, **shared evaluation code**, and **wrappers** around models or APIs. You can write good analysis without heavy OOP, but you cannot avoid **reading** objects once you use pandas and sklearn.

## Design Patterns in Data Science

### 1. Factory Pattern

Imports

Import `ABC` and `abstractmethod` for defining abstract base classes, plus type hints for the registry dict.

Abstract Base

`Model` is an abstract class declaring `train` and `predict` as required, any subclass must implement both or Python raises a `TypeError`.

Concrete Models

Each concrete class fulfills the contract by implementing `train` and `predict`. Swapping implementations is easy without touching the factory.

Factory Class

`ModelFactory` maps string names to classes. `create_model` looks up the right class and instantiates it, callers never import concrete model classes directly.

### 2. Strategy Pattern

Protocol Interface

`FeatureEngineeringStrategy` is a structural interface: any class with an `engineer_features` method qualifies, no explicit inheritance needed.

Date Strategy

`DateFeatures` extracts year and month from every datetime column, adding them as new numeric features the model can use.

Text Strategy

`TextFeatures` computes character length and word count for string columns, simple proxies for text complexity.

Context Class

`FeatureEngineer` accepts a list of strategies and chains them in `apply_all`. Adding a new strategy requires no changes here, just pass it in at construction.

### 3. Observer Pattern

Observer Interface & Data

`ModelObserver` is the subscriber interface; `ModelMetrics` is a dataclass that bundles a timestamp with the metrics dict for structured storage.

Metrics Logger

Appends every update to a history list so you can replay or analyze training metrics over time.

Alert System

A second independent observer that fires an alert when error exceeds a configurable threshold, added without modifying any existing code.

Observable Model

The subject maintains a list of observers and calls `update` on each in `notify`. The model itself knows nothing about logging or alerting.

## Testing and Debugging

### Unit Testing

Test Setup

`setUp` runs before every test method, creating a fresh DataFrame with known nulls and a pipeline to transform it.

Null Check Test

Asserts that after imputation, no `NaN` values remain anywhere in the output DataFrame.

Scaling Test

Verifies standard scaling: each numeric column should have mean ≈ 0 and std ≈ 1 after `FeatureScaler` runs.

Run Tests

Standard entry point so the test suite runs when the file is executed directly with `python test_pipeline.py`.

### Debugging Tips

1. **Use Logging Effectively**

Logger Setup

Configure a module-level logger with timestamps and severity levels. Using `__name__` means each module gets its own logger namespace.

Instrumented Transform

Logs data shape on entry, success on exit, and the exception message on failure before re-raising, so the caller still sees the error while the log captures full context.

2. **Data Validation**

Result Dataclass

`DataValidationResult` is a typed container for validation output, separating hard errors from soft warnings lets callers decide how strict to be.

Missing Value Check

Counts nulls per column; columns with any missing values are reported as a warning (non-fatal) with their names listed for easy diagnosis.

Type Check & Return

Flags non-numeric data as a hard error, then returns a result whose `is_valid` field is `True` only when the errors list is empty.

## Error Handling Best Practices

### 1. Custom Exceptions

Base Exception

`PipelineError` is the root of the hierarchy, callers can catch this single type to handle any pipeline failure.

Specialised Subclasses

`DataValidationError` and `ModelError` are plain subclasses, their type alone communicates the failure category without any extra data.

Transformer Error

Stores the transformer name as an attribute and formats it into the message string so tracebacks immediately identify which step failed.

### 2. Graceful Error Handling

Pipeline Init

Stores the list of transformer steps and an `errors` log that accumulates failure records throughout the pipeline's lifetime.

Input Validation

Runs `DataValidator` before any transformation, raises a typed `DataValidationError` immediately if the data doesn't meet requirements.

Step Execution

Each step runs in its own try/except. Failures are logged with step name and timestamp, then re-raised as `TransformerError` for specific error identification.

Outer Handler

Catches any uncaught exception at the pipeline level, logs it as a pipeline-level error, and wraps it in a `PipelineError` for the caller.

## Performance Optimization

### 1. Parallel Processing

Constructor

Accepts any callable transformation function and a worker count. Setting `n_jobs=-1` is conventional for "use all available CPUs".

Data Splitting

`np.array_split` divides the DataFrame into `n_jobs` roughly equal chunks to distribute across workers.

Parallel Execution

A `ProcessPoolExecutor` maps the function over all chunks in parallel processes, then `pd.concat` reassembles the results in the original order.

### 2. Memory Optimization

Init

Stores the list of transformer steps to apply sequentially to each data chunk.

Chunked Reading

`pd.read_csv` with `chunksize` streams the file in 1,000-row batches rather than loading everything into RAM at once.

Process & Concat

Each chunk is passed through every pipeline step before being collected; `pd.concat` joins all processed chunks into one final DataFrame.

## Advanced Data Science Classes

***

### Machine Learning Pipeline

Example of a modular ML pipeline:

Base Transformer

`BaseTransformer` defines the `fit`/`transform` contract all steps must implement, plus a free `fit_transform` convenience method.

Missing Value Imputer

`fit` computes fill values (mean/median for numeric, mode for categorical) from training data; `transform` applies those stored values to new data.

Outlier Handler

Computes IQR-based bounds per column during `fit`, then replaces outliers with `NaN` in `transform`, a common pre-processing step before imputation.

Feature Scaler

Supports standard (z-score) and min-max scaling; parameters are learned in `fit` and applied identically to train and test sets.

ML Pipeline Class

Orchestrates any list of transformers plus an optional estimator. `fit` chains `fit_transform` steps then trains the model; `predict` runs the transform chain before calling the model.

Usage Example

Builds a sample DataFrame with nulls and an outlier, wires up the three-step pipeline with a `RandomForestClassifier`, then fits and predicts in two lines.

***

### Data Pipeline Architecture

Example of a data processing pipeline:

\`\`\`

Pipeline execution failed: \[Errno 2] No such file or directory: 'data.csv'

```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-20" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Abstract Step</span>
    </div>
    <div class="code-callout__body">
      <p><code>DataPipelineStep</code> mandates a <code>process</code> method and a <code>get_step_name</code> string on every concrete step, used for logging and error messages.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-38" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data Loader</span>
    </div>
    <div class="code-callout__body">
      <p>Reads CSV or Parquet based on file extension, raising a clear error for unsupported formats so the pipeline fails fast with a useful message.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="40-68" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data Cleaner</span>
    </div>
    <div class="code-callout__body">
      <p>Optionally drops duplicate rows, then fills numeric nulls with column means and categorical nulls with the mode, both controlled by constructor flags.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="70-86" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Feature Engineer</span>
    </div>
    <div class="code-callout__body">
      <p>Parses specified date columns and expands each into year, month, day, and day-of-week numeric features for downstream models.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="88-130" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline Orchestrator</span>
    </div>
    <div class="code-callout__body">
      <p><code>DataPipeline.run</code> iterates steps, logs start/end times and durations for each, and re-raises any step exception after logging the error with its step name.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="132-143" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Usage Example</span>
    </div>
    <div class="code-callout__body">
      <p>Wires three steps into a pipeline and runs it inside a try/except so any step failure prints a human-readable message rather than a raw traceback.</p>
    </div>
  </div>
</aside>
</div>

```

Pipeline execution failed: \[Errno 2] No such file or directory: 'data.csv'

````

## Practice Exercises for Data Science
Try these advanced exercises:

1. **Create a Feature Selection System**

   ```python
  # Build classes for:
  # - Feature importance calculation
  # - Correlation analysis
  # - Feature selection based on metrics
  # - Feature ranking and visualization
````

2.  **Implement a Model Evaluation Pipeline**

    ```python
    ```

## Create classes for:

## - Cross-validation

## - Metric calculation

## - Model comparison

## - Results visualization

````

3. **Build an Automated Report Generator**

```python
# Develop classes for:
# - Data profiling
# - Statistical analysis
# - Visualization generation
# - Report formatting
````

Remember:

* Use type hints for better code documentation
* Implement proper error handling
* Consider performance implications
* Write unit tests for your classes
* Follow SOLID principles

Happy coding!

## Additional Resources

1. **Books**

* "Clean Code" by Robert C. Martin
* "Design Patterns" by Gang of Four
* "Python Patterns" by Brandon Rhodes

2. **Online Resources**

* [Real Python OOP Tutorials](https://realpython.com/python3-object-oriented-programming/)
* [Python Design Patterns](https://python-patterns.guide/)
* [Scikit-learn Development Guide](https://scikit-learn.org/stable/developers/index.html)

3. **Tools**

* [PyTest](https://docs.pytest.org/) for testing
* [Black](https://github.com/psf/black) for code formatting
* [Mypy](http://mypy-lang.org/) for type checking

Remember: "Clean code is not written by following a set of rules. You don't become a software craftsman by learning a list of heuristics. Professionalism and craftsmanship come from values that drive disciplines." - Robert C. Martin

***

## Modern Learning Tips

### Use AI for OOP Learning

```
"Explain the difference between classes and objects using real-world examples"
"Show me when to use inheritance vs composition"
"Review my class design: [paste code]"
"Create practice exercises for OOP concepts"
```

### Visualize with Python Tutor

Perfect for visualizing:

* Object creation and initialization
* Method calls and **self**
* Inheritance relationships
* Instance vs class variables

### Debug with Modern Tools

* Use VS Code / Cursor debugger
* Set breakpoints in methods
* Inspect object attributes
* Step through method calls

> **Video Help:** See [Video Resources](video-resources.md) - OOP section for detailed tutorials

## Common pitfalls

* **Forgetting self**: Instance methods need **self** as the first parameter so Python can pass the object.
* **Confusing class and instance attributes**: Mutable class-level defaults (like lists) are shared across instances unless you set them in **init**.
* **Overusing inheritance**: Prefer composition when you only need to reuse behavior without an "is-a" relationship.

## Next steps

Continue to [Modules](modules.md) to organize code across files and reuse libraries.

Happy coding!
