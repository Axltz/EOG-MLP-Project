# EOG Signal Classification with a Scratch-Built MLP

Implementation of a Multi-Layer Perceptron (MLP) from scratch using **NumPy** to classify 10 eye-movement commands from Electrooculography (EOG) signals based on the *Blink-to-Speak* language.

## Key Technical Highlights
* **Pure NumPy Architecture:** Forward propagation, backpropagation, ReLU/Tanh activations, numerically stable Softmax, and Categorical Cross-Entropy implemented without high-level ML frameworks.
* **Data Leakage Prevention:** Custom 5-Fold Cross-Validation computing Z-score parameters strictly on training folds before transforming validation sets.
* **Dimensionality Reduction:** Pearson correlation matrix filtering ($|r| > 0.8$) to eliminate multicollinearity and extreme variance artifacts ($10^{17}$) across synthetic channel operations.
* **Critical Evaluation:** Ablation analysis removing temporal duration (`Length`) to expose experimental bias in the benchmark dataset.

## Repository Structure
* `data/` — Original and correlation-filtered datasets.
* `src/config.py` — Global hyperparameters and file paths.
* `src/01_data_correlation.py` — Z-score standardization and correlation analysis.
* `src/02_mlp_experiments.py` — Custom `MLP_Numpy` class and K-Fold training pipeline.
* `src/main.py` — End-to-end execution script.
* `fig/` — Distributions, correlation heatmaps, loss curves, and confusion matrices.

## Results Summary
| Model / Experiment | Features | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Benchmark (Decision Tree w/ Length) | 19 | 83.4% | — | — | — |
| Custom MLP (Baseline w/o Length - Phase 2) | 29 | **57.0%** (±6.6%) | **59.0%** | **57.0%** | **55.7%** |
| Custom MLP (Correlation Filtered \|r\|>0.8 - Phase 4) | 20 | **44.5%** (±14.7%) | **45.1%** | **44.5%** | **43.6%** |