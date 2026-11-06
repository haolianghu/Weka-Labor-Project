# Labor contract feature selection

The repository includes the original 57-row labor-negotiation dataset in
`labor_data_before_feature_selection.arff` and a reduced dataset produced by
Weka's supervised wrapper selection in
`labor_data_after_feature_selection.arff`. Both contain the same examples and
class labels. The included Weka model and result buffer document the original
run.

## Reproducible comparison

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python labor_baseline.py
python -m unittest discover -s tests -v
```

The comparison uses identical stratified five-fold splits for both feature
sets, fits imputation, scaling, and one-hot encoding separately inside each
training fold, and reports ROC AUC and balanced accuracy from out-of-fold
predictions. The dataset has 57 observations and substantial missing fields;
the classifier and split seed are fixed for reproducibility. This is a new
scikit-learn benchmark, not an attempt to reproduce Weka's J48 wrapper output.

## Interpretation

This small, historical sample is not representative evidence for current labor
negotiations. Feature selection itself used the same dataset, so evaluation of
the selected set is optimistic unless selection is repeated inside each
training fold (nested validation). Class imbalance, missing values, and
selection uncertainty limit conclusions. These scores are exploratory and
should not be used to predict or judge real contracts.
