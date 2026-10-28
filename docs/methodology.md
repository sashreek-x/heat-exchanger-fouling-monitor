# Methodology

## Research question
Can operating measurements and physically motivated features estimate a documented fouling-resistance label or identify sustained thermal-performance deterioration?

## Data and label
Record the dataset URL/citation, license, exchanger type, sampling frequency, sensor names, units, cleaning rules and target-label derivation. A target inferred from thermal calculations is an estimate with uncertainty, not a direct measurement. If the label is synthetic, describe the work as a pipeline demonstration only.

## Preprocessing
- Convert numeric measurements explicitly; treat implausible values as data-quality flags for investigation.
- Sort timestamps and avoid duplicate observations where appropriate.
- Impute numeric missing values inside a training-only pipeline.
- Use chronological validation when observations are time ordered.
- Check data leakage: target-derived columns, post-cleaning measurements and future observations must not be predictors for past prediction.

## Models
Compare Linear Regression, Random Forest, Gradient Boosting, HistGradientBoosting and MLP regression. Keep a simple baseline. Tune hyperparameters only within training data; report the selected protocol.

## Evaluation
Report MAE, RMSE, R² and MAPE only where the target scale makes percentage error meaningful. Include holdout size, CV RMSE, residual plots and errors by operating regime. Do not compare metrics across different targets or splits as if directly equivalent.

## Engineering interpretation
Use duty balance and U-value trends as diagnostics, not unquestionable labels. Thermal resistance calculations depend on area, flow arrangement, properties, heat loss, fouling location and baseline comparability. Pressure drop may provide complementary hydraulic evidence. Feature importance is not causation.

## Deployment and safety
Show timestamps, model version, data-quality warnings and prediction context. Set alarm thresholds from engineering review and operating history; do not hard-code universal thresholds. A model output must not directly trigger safety-critical shutdown or cleaning without approved controls and human review.
