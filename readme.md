# MLClassifier - Machine Learning Classification System

## Homework 2 - Machine Learning Classification Algorithms Implementation

This project implements and compares four classification algorithms with different data preprocessing methods and parameter optimization techniques for comprehensive machine learning model evaluation.

## Overview

MLClassifier is a Go-based machine learning library that implements multiple classification algorithms including K-Nearest Neighbors (KNN), Decision Trees, Random Forest, and Naive Bayes. The system provides an interactive command-line interface for model training, evaluation, and experimentation, along with automated parameter optimization and comprehensive result export capabilities.

## Key Features

- **Four Classification Algorithms**:
  - K-Nearest Neighbors (KNN) with configurable k and distance metrics
  - Decision Tree with depth control and minimum split parameters
  - Random Forest ensemble method with parallel training
  - Naive Bayes probabilistic classifier with smoothing

- **Three Data Preprocessing Methods**:
  - Raw data (no preprocessing)
  - MinMax Normalization (scales features to [0,1])
  - Standardization (zero mean, unit variance)

- **Comprehensive Evaluation**:
  - Cross-validation with configurable folds (including Leave-One-Out)
  - Multiple metrics: Accuracy, Precision, Recall, F1-Score
  - Confusion matrices with per-class analysis
  - Statistical validation with mean and standard deviation

- **Automated Experimentation**:
  - Grid search for parameter optimization
  - Systematic algorithm comparison
  - Result export to CSV format for spreadsheet analysis

## Quick Start Guide

### Basic Workflow

```bash
# Start the CLI
./mlc

# Load a dataset
mlc> load data/iris.csv

# Train a model
mlc> train knn 5 --prep=normalized

# Evaluate performance
mlc> evaluate

# Run cross-validation
mlc> cv
```

### Running a Complete Experiment

```bash
mlc> load data/iris.csv
mlc> experiment
# Select: 1 (Full comparison)
# Enter: 10 (CV folds)
```

This will test all algorithms with all preprocessing methods and export comprehensive results to the `results/` directory.

## Command Reference

### Data Management

```bash
load <filepath>                 # Load CSV dataset
load-streaming <filepath>       # Stream large files
info                           # Display dataset information
```

### Model Training

```bash
# K-Nearest Neighbors
train knn <k> [distance] [--prep=method]
# Example: train knn 5 euclidean --prep=normalized

# Decision Tree
train tree <depth> [min_split] [--prep=method]
# Example: train tree 10 2 --prep=normalized

# Random Forest
train forest <n_trees> [depth] [--prep=method]
# Example: train forest 50 10 --prep=normalized

# Naive Bayes
train bayes [smoothing] [--prep=method]
# Example: train bayes 1e-9 --prep=standardized
```

### Preprocessing Options
- `--prep=raw` - No preprocessing (default)
- `--prep=normalized` - MinMax scaling to [0,1]
- `--prep=standardized` - Zero mean, unit variance

### Model Evaluation

```bash
evaluate                    # Show confusion matrix and metrics
cv                         # Interactive cross-validation
save-results [folds]       # Save results with CV
compare                    # Compare all saved models
best                       # Find best performing model
```

### Predictions

```bash
predict <values>           # Single prediction
batch <file>              # Batch predictions from CSV
test                      # Interactive testing mode
```

### Experiments

```bash
experiment                 # Interactive experiment setup
experiment-bg              # Run experiment in background
experiments                # List past experiments
view <exp_name>           # View experiment results
```

### Model Management

```bash
list                      # List saved models
loadmodel <file>          # Load specific model
current                   # Show current model info
model-versions            # List model versions
model-compare <v1> <v2>   # Compare versions
```

### Job Management

```bash
job-status               # List all background jobs
job-status <id>          # Check specific job
job-cancel <id>          # Cancel running job
job-logs <id>            # View job logs
```

## Experiment Types

### 1. Full Comparison
Tests all algorithms with all preprocessing methods and parameter combinations:
- 4 algorithms × 3 preprocessing × multiple parameters
- Generates comprehensive comparison tables

### 2. Single Algorithm Optimization
Focuses on one algorithm with parameter grid search:
- KNN: k values [3, 5, 7] × distances [euclidean, manhattan]
- Decision Tree: depths [3, 5, 10, 15] × min_splits [2, 5, 10]
- Random Forest: trees [10, 50, 100] × depths [5, 10, 15]

### 3. Custom Selection
Choose specific algorithms and preprocessing methods for targeted comparison

### 4. Current Model Only
Evaluate the currently trained model with comprehensive metrics

## Results and Output

All experiment results are saved in the `results/` directory with timestamps:

```
results/exp_<dataset>_<type>_<timestamp>/
├── experiment_results.csv       # Main metrics table
├── evaluation_data.csv         # All predictions
├── cv_performance.csv          # Per-fold metrics
├── cv_folds.csv               # Cross-validation details
├── data_splits.csv            # Train/test assignments
├── confusion_matrix_*.csv     # Per-model confusion matrices
├── algorithm_comparison.csv   # Algorithm ranking
├── preprocessing_comparison.csv # Preprocessing impact
├── parameter_optimization.csv  # Best parameters found
└── experiment_summary.txt      # Human-readable report
```

### CSV Output Format

The main results file (`experiment_results.csv`) contains:
- Algorithm name and parameters
- Preprocessing method used
- Train/test split ratio
- Accuracy, Precision, Recall, F1-Score
- Cross-validation mean and standard deviation

## Parameter Optimization Examples

### KNN Parameter Study
```bash
# Test different k values
mlc> train knn 1   # Accuracy: varies (high variance)
mlc> train knn 3   # Accuracy: good balance
mlc> train knn 5   # Accuracy: often optimal
mlc> train knn 7   # Accuracy: may underfit

# Test distance metrics
mlc> train knn 5 euclidean  # Best for continuous features
mlc> train knn 5 manhattan  # Better for high dimensions
```

### Decision Tree Optimization
```bash
# Depth impact
mlc> train tree 3   # Shallow: may underfit
mlc> train tree 5   # Balanced: good generalization
mlc> train tree 10  # Deep: risk of overfitting
mlc> train tree 15  # Very deep: likely overfitting
```

### Random Forest Tuning
```bash
# Number of trees
mlc> train forest 10   # Quick but high variance
mlc> train forest 50   # Good balance
mlc> train forest 100  # Marginal improvement, slower
```

## Performance Metrics

### Accuracy Metrics
- **Simple Accuracy**: (TP+TN)/Total
- **Balanced Accuracy**: Average of per-class recall
- **Weighted Accuracy**: Weighted by class frequency

### Classification Metrics
- **Precision**: TP/(TP+FP) - When predict positive, how often correct
- **Recall**: TP/(TP+FN) - Of all positive cases, how many found
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: TN/(TN+FP) - Of all negative cases, how many identified

### Averaging Methods
- **Macro**: Simple average (treats classes equally)
- **Micro**: Global calculation (aggregates all)
- **Weighted**: Weighted by class support

## Cross-Validation Options

### Standard K-Fold
```bash
mlc> cv
Number of folds: 5
```

### Leave-One-Out (LOO)
```bash
mlc> cv
Number of folds: loo  # Uses n_samples as folds
```

## Advanced Features

### Background Processing
Run long experiments without blocking:
```bash
mlc> experiment-bg
mlc> job-status
```

### Model Versioning
Track model evolution:
```bash
mlc> model-versions
mlc> model-promote <version>
mlc> model-rollback
```

### Streaming for Large Datasets
Process files larger than RAM:
```bash
mlc> load-streaming data/large.csv --batch-size=1000
```

### Parallel Processing
- Cross-validation runs folds in parallel (4-8 workers)
- Random Forest trains trees in parallel
- Multiple background jobs run concurrently

## Complete Workflow Example

```bash
# 1. Start CLI and load data
./mlc
mlc> load data/iris.csv
mlc> info

# 2. Quick model comparison
mlc> train knn 5 --prep=normalized
mlc> train tree 10 --prep=normalized
mlc> train forest 50 --prep=normalized
mlc> compare

# 3. Run comprehensive experiment
mlc> experiment
Select type: 1 (Full comparison)
CV folds: 10

# 4. View results
mlc> experiments
mlc> view exp_iris_comparison_[timestamp]

# 5. Make predictions with best model
mlc> loadmodel models/best_model.model
mlc> predict 5.1 3.5 1.4 0.2
```