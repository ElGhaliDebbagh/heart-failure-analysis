# Heart Failure Analysis

Python course project — November 2025.

**Authors:** El Ghali Debbagh and Mariane Modena

## Objective

Explore factors associated with mortality during follow-up among 299 patients with heart failure.

## Methods

- Data quality checks and descriptive statistics.
- Visualisation of distributions and relationships between variables.
- Student’s t-tests and Fisher’s exact tests.
- Decision tree classification with stratified splits and grid search.
- Sensitivity analysis across ten random splits.

## Results

The dataset contains no missing values; 96 patients (32.1%) died during follow-up. The selected decision tree achieved approximately 72% accuracy on the initial test split. Across ten splits, accuracy ranged from 65% to 78.3%. The majority-class baseline is approximately 68% overall. The small sample and varying follow-up durations limit interpretation. Follow-up duration is excluded from the predictors because it is not available at baseline. This is an exploratory academic analysis, not a clinical prediction tool.

## Reproduce

Place the original `heart_failure.csv` next to the script, then run:

```bash
pip install -r requirements.txt
python heart_failure.py
```

The script generates figures and prints statistical and classification results. The individual-level data are not included; use the original course dataset with appropriate permission.
