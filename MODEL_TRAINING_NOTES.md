# Tuned Isolation Forest Training Notes

Main model:

`models/isolation_forest_lmd2023_tuned.joblib`

Training code:

`tune_isolation_forest_model.py`

Testing code:

`test_isolation_forest_model.py`

## Feature Selection

The model uses the eight features selected in the paper:

- `Computer`
- `DestinationPortName`
- `EventID`
- `Initiated`
- `SourceIsIpv6`
- `EventRecordID`
- `Execution_ProcessID`
- `ProcessId`

In code, these are defined in `tune_isolation_forest_model.py`:

- categorical features: lines 21-24
- selected CSV columns: line 43

The script loads only `Label` plus those eight features from the CSV to reduce memory use.

## Label Handling

The LMD CSV contains labels that may be multiclass. For this binary model:

- `0` means normal
- any positive value means attack

This is done in `tune_isolation_forest_model.py` line 48.

The label is not used to fit the Isolation Forest. It is used only for:

- stratified train/validation/test splitting
- choosing the validation threshold
- reporting metrics

## Preprocessing

Preprocessing is defined in `build_preprocessor()` in `tune_isolation_forest_model.py` lines 52-64.

Categorical fields:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)`

Numeric fields:

- `SimpleImputer(strategy="median")`
- `MinMaxScaler()`

The fitted preprocessor is saved inside the `.joblib` model artifact, so deployment uses the same preprocessing.

## Split

The model uses the requested:

- 60% train
- 20% validation
- 20% test

The split is stratified to preserve the normal/attack ratio.

## Tuning

A small low-resource search is used because the machine has 8 GB RAM.

The search is in `tune_isolation_forest_model.py` lines 128-136.

The best configuration was:

- `n_estimators=300`
- `contamination=0.3`
- `max_features=0.5`
- `max_samples="auto"`
- `warm_start=True`

## Threshold

Isolation Forest produces anomaly scores. The code converts scores as:

`score = -model.decision_function(X)`

Higher score means more suspicious.

The threshold is selected on the validation split to maximize F1.

Final threshold:

`0.10984904755792074`

## Results

Held-out test results:

- AUC: `0.9739`
- Precision: `0.6435`
- Recall: `0.9559`
- F1: `0.7692`
- Accuracy: `0.9538`

Use this model as the ML layer after your rule layer.
