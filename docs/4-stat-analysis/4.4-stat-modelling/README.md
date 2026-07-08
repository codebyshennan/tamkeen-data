# Statistical Modeling with Python

**After this submodule:** you can use the lessons linked below and complete the exercises that match **Statistical Modeling with Python** in your course schedule.

## Overview

Submodule 4.4 extends [linear regression ideas from 4.3](../4.3-rship-in-data/) into **classification** (logistic regression), **nonlinear structure** (polynomials), **choosing complexity** (model selection), **controlling complexity** (regularization), and **explaining predictions** (interpretation). The flow is deliberate: build a flexible model class, learn to pick and constrain it, then communicate what it means to others.

See the [Module 4 overview](../) for prerequisites and the full learning path.

## Helpful video

StatQuest: connecting regression-style thinking to common tests.

## Learning Objectives

By the end of this module, you will be able to:

* Fit and interpret logistic regression for binary classification (sigmoid, odds ratios, confusion matrix, ROC).
* Extend linear methods to curved relationships with polynomial features, balancing flexibility and overfitting.
* Compare candidate models using cross-validation and information criteria, not training error alone.
* Apply Ridge and Lasso regularization to shrink coefficients and select features, tuning penalty strength with care.
* Communicate model behaviour with coefficients, partial dependence, and modern attribution tools.

## Topics Covered

1. [Logistic Regression](logistic-regression.md), binary classification, the sigmoid link, odds ratios, ROC.
2. [Polynomial Regression](polynomial-regression.md), non-linear relationships via feature transformation.
3. [Model Selection](model-selection.md), cross-validation, feature selection, AIC/BIC-style thinking.
4. [Regularization](regularization.md), Ridge, Lasso, Elastic Net, and hyperparameter tuning.
5. [Model Interpretation](model-interpretation.md), coefficients, partial dependence, SHAP-style explanations.

## Prerequisites

* [Regression basics from module 4.3](../4.3-rship-in-data/), especially [simple](../4.3-rship-in-data/simple-linear-regression.md) and [multiple linear regression](../4.3-rship-in-data/multiple-linear-regression.md).
* Comfort with NumPy and pandas for tabular work; basic statistics (mean, variance, correlation).

## Why this matters

Classification, curvature, and complexity control show up in nearly every applied modelling problem. The interpretation toolkit at the end of this submodule connects what models do to what stakeholders can act on.
