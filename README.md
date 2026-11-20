# Credit Risk Default Prediction

Predicts whether a borrower will **default on a loan**, using 32,581 loan applications with 11 borrower and loan features. It also includes an interactive dashboard that shows which factors are linked to default.

## Results

A scikit-learn pipeline (mean imputation, standard scaling, Random Forest) evaluated on a 30% hold-out set of 9,775 loans:

| Metric | Value |
|---|---|
| Accuracy | **93.1%** |
| Precision on defaults | 0.97 |
| Recall on defaults | 0.71 |
| Precision / recall on repaid loans | 0.92 / 0.99 |

The data is imbalanced: 78% of loans were repaid and 22% defaulted. That is why precision and recall on defaults are reported, not accuracy alone.

## What the notebook covers

1. **Cleaning:** imputing 895 missing employment lengths and 3,116 missing interest rates.
2. **Encoding:** mapping home ownership, loan purpose, ordinal loan grade (A–G) and prior default history to numbers.
3. **Exploration:** class balance and a correlation heatmap.
4. **Modelling:** a scikit-learn `Pipeline` with a `ColumnTransformer`, so preprocessing is fitted only on training data.
5. **Evaluation:** a classification report, a confusion matrix and an interactive ROC curve.
6. **Dashboard:** linked Bokeh charts with hover tooltips showing default rate by:
   - loan grade
   - loan amount as a percentage of income
   - whether the borrower has defaulted before

## Run it

1. Download `credit_risk_dataset.csv` from the [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).
2. Place it next to the notebook.
3. Run:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn bokeh jupyter
jupyter notebook credit_risk_prediction.ipynb
```

## Built with

Python · pandas · NumPy · scikit-learn · Bokeh · Matplotlib · seaborn · Google Colab
