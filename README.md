# CDS535 Group Project

Comparative data mining on the UCI Mushroom and Ionosphere datasets.

## Reproduce

Use Python 3.12 and install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
python run_experiments.py
python demo.py
```

The raw data are included. Running the experiment overwrites generated results and figures.

## Study design

- Mushroom: 8,124 categorical records.
- Ionosphere: 351 raw records, 350 after removing one exact duplicate.
- Four classifiers: CART, random forest, logistic regression and RBF SVM.
- Stratified 20% holdout, with ten-fold cross-validation on the development set.
- All learned preprocessing is fitted inside training folds.
- Controlled missing-value and scaling comparisons, plus K-means exploration.

## Files

- `data/`: original data, source notes, converted CSVs and schemas.
- `results/`: fixed splits, fold scores, holdout predictions and comparisons.
- `figures/`: generated charts.
- `run_experiments.py`: complete reproducible experiment.
- `demo.py`: short radar classification demonstration.

## Data provenance

Mushroom: UCI ID 73, https://doi.org/10.24432/C5959T. Original donor Jeff Schlimmer (1987), based on a 1981 field-guide source.

Ionosphere: UCI ID 52, https://doi.org/10.24432/C5W01B. Sigillito, Wing, Hutton and Baker (1989).

Both UCI pages specify CC BY 4.0. Original source notes remain in `data/`. Changes include duplicate removal for Ionosphere, named columns and binary label mapping. See `results/data_sha256.json` for data hashes and `results/environment.json` for the execution environment.

These are benchmark results and do not establish real-world deployment safety or universal model superiority.

## Weka demonstration

See [Weka_Demonstration](Weka_Demonstration) for actual Weka 3.8.7 experiments, ARFF data, saved models, reproducible commands and a real Explorer launcher. Weka results are separate from the Python results above.
