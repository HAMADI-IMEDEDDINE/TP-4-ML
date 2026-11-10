"""
National Higher School of Advanced Technologies
Academic year 2026–2027 | First-year Master's degree
Machine Learning – Practical Session 4
Overfitting, data leakage and model generalization
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def section_header(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

# ==============================================================================
# 2. DATASET AND TASK / ENCODINGS
# ==============================================================================
section_header("2. DATASET AND TASK - LOADING & ENCODING")
csv_path = "hybrid_student_performance_1200.csv" if os.path.exists("hybrid_student_performance_1200.csv") else "student_performance.csv"
raw_df = pd.read_csv(csv_path)
print(f"Dataset loaded: {csv_path} with shape {raw_df.shape}")

def clean_str(val):
    if pd.isna(val):
        return val
    return str(val).replace('\u2013', '-').replace('–', '-').strip()

df = raw_df.copy()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].apply(clean_str)

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Encodings as specified in Section 2
cgpa_map = {'5.0 - 6.9': 5.95, '7.0 - 8.4': 7.70, '8.5 - 9.4': 8.95, '9.5 - 10.0': 9.75}
df['final_cgpa'] = df['cgpa_category'].map(cgpa_map)

df['study_hours'] = df['study_hours_daily'].map({'Less than 1 hour': 0.5, '1-2 hours': 1.5, 'More than 2 hours': 2.5})
df['sleep_hours'] = df['sleep_hours'].map({'4-5 hours': 4.5, '6-7 hours': 6.5, 'More than 8 hours': 8.5})
df['attendance_rate'] = df['attendance_percentage'].map({
    'Less than 50%': 40.0, '50% - 65%': 57.5, '66% - 75%': 70.5, '76% - 85%': 80.5, 'Above 85%': 92.5
})
df['study_consistency'] = df['study_consistency'].map({'Rarely': 0, 'Sometimes': 1, 'Mostly consistent': 2})
df['tasks_on_time'] = df['tasks_on_time'].map({'Rarely': 0, 'Sometimes': 1, 'Often': 2, 'Always': 3})
df['assignments_on_time'] = df['assignments_on_time'].map({'Rarely': 0, 'Sometimes': 1, 'Often': 2, 'Always': 3})
df['revision_frequency'] = df['revision_frequency'].map({'Never': 0, 'Rarely': 1, 'Few times a week': 2, 'Daily': 3})
df['career_goal_clarity'] = df['career_goal_clarity'].map({'Not clear': 0, 'Somewhat clear': 1, 'Very clear': 2})

df['daily_productivity'] = pd.to_numeric(df['daily_productivity'])
df['stress_level'] = pd.to_numeric(df['stress_level'])
df['age'] = pd.to_numeric(df['age'])

df['cohort_code'] = df['program_stream'].astype(str) + " | " + df['year_class'].astype(str)

effort_components = pd.DataFrame({
    'sh': df['study_hours'] / 2.5,
    'att': df['attendance_rate'] / 100.0,
    'sc': df['study_consistency'] / 2.0,
    'tot': df['tasks_on_time'] / 3.0,
    'aot': df['assignments_on_time'] / 3.0,
    'rf': df['revision_frequency'] / 3.0
})
df['effort_index'] = effort_components.mean(axis=1, skipna=True)

# Sort once chronologically
df_chrono = df.sort_values('timestamp').reset_index(drop=True)
t0 = df_chrono['timestamp'].min()
df_chrono['submission_day'] = (df_chrono['timestamp'] - t0).dt.total_seconds() / (24 * 3600.0)

# ==============================================================================
# PART A. DATA INSPECTION AND POTENTIAL LEAKAGE
# ==============================================================================
section_header("PART A. DATA INSPECTION AND POTENTIAL LEAKAGE")

# A.1
print("A.1 Missing values report:")
miss_cnt = df.isna().sum()
miss_report = pd.DataFrame({'Missing': miss_cnt[miss_cnt > 0], 'Pct': (miss_cnt[miss_cnt > 0] / len(df)) * 100})
print(miss_report.to_string())

# A.2
dup_all = df.duplicated().sum()
dup_no_id = df.drop(columns=['student_id']).duplicated().sum()
dup_no_ts = df.drop(columns=['timestamp']).duplicated().sum()
print(f"\nA.2 Duplicates: with student_id={dup_all}, without student_id={dup_no_id}, ignoring timestamp={dup_no_ts}")

# A.3
num_cols = ['effort_index', 'attendance_rate', 'study_consistency', 'study_hours',
            'daily_productivity', 'career_goal_clarity', 'tasks_on_time',
            'assignments_on_time', 'revision_frequency', 'stress_level',
            'sleep_hours', 'age']
corrs = df_chrono[num_cols].apply(lambda c: c.corr(df_chrono['final_cgpa'])).sort_values(ascending=False)
print("\nA.3 Feature correlations with target (final_cgpa):")
print(corrs.to_string())

# A.4
print("\nA.4 Cross-tabulation: performance_risk_level vs cgpa_category:")
print(pd.crosstab(df['performance_risk_level'], df['cgpa_category']))
print("Can this label be used as a predictive feature? No, it causes target leakage.")

# A.5
cohort_counts = df['cohort_code'].value_counts()
print(f"\nA.5 Cohort code total categories: {len(cohort_counts)}, categories with < 10 rows: {(cohort_counts < 10).sum()}")

# A.6
print(f"\nA.6 Is raw file in chronological order? {df['timestamp'].is_monotonic_increasing}")

# ==============================================================================
# PART B. IMPACT OF DATA LEAKAGE AND EVALUATION STRATEGY
# ==============================================================================
section_header("PART B. IMPACT OF DATA LEAKAGE AND EVALUATION STRATEGY")

part_b_features = ['study_hours', 'sleep_hours', 'attendance_rate', 'study_consistency',
                   'tasks_on_time', 'assignments_on_time', 'revision_frequency',
                   'career_goal_clarity', 'daily_productivity', 'stress_level', 'age', 'submission_day']

split_idx = int(0.8 * len(df_chrono))
train_df = df_chrono.iloc[:split_idx].copy()
test_df = df_chrono.iloc[split_idx:].copy()

X_train = train_df[part_b_features]
y_train = train_df['final_cgpa']
X_test = test_df[part_b_features]
y_test = test_df['final_cgpa']

# B.1 Splitting strategies
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
dummy_tr = root_mean_squared_error(y_train, dummy.predict(X_train))
dummy_te = root_mean_squared_error(y_test, dummy.predict(X_test))

tree_chrono = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(random_state=RANDOM_STATE))]).fit(X_train, y_train)
b1_chrono_tr = root_mean_squared_error(y_train, tree_chrono.predict(X_train))
b1_chrono_te = root_mean_squared_error(y_test, tree_chrono.predict(X_test))

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(df_chrono[part_b_features], df_chrono['final_cgpa'], test_size=0.2, random_state=RANDOM_STATE)
tree_rand = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(random_state=RANDOM_STATE))]).fit(Xr_tr, yr_tr)
b1_rand_tr = root_mean_squared_error(yr_tr, tree_rand.predict(Xr_tr))
b1_rand_te = root_mean_squared_error(yr_te, tree_rand.predict(Xr_te))

print("B.1 Splitting Strategies Summary:")
print(pd.DataFrame([
    {'Strategy': 'Baseline (DummyRegressor)', 'Training RMSE': dummy_tr, 'Testing RMSE': dummy_te},
    {'Strategy': 'DecisionTree (Random Split 80/20)', 'Training RMSE': b1_rand_tr, 'Testing RMSE': b1_rand_te},
    {'Strategy': 'DecisionTree (Chronological Split 80/20)', 'Training RMSE': b1_chrono_tr, 'Testing RMSE': b1_chrono_te}
]).to_string(index=False))

# B.2 Duplicate leakage
dup_sample = df.sample(frac=0.15, random_state=RANDOM_STATE).copy()
min_ts, max_ts = df['timestamp'].min(), df['timestamp'].max()
tot_sec = (max_ts - min_ts).total_seconds()
np.random.seed(RANDOM_STATE)
dup_sample['timestamp'] = [min_ts + pd.Timedelta(seconds=s) for s in np.random.uniform(0, tot_sec, size=len(dup_sample))]

aug_df = pd.concat([df, dup_sample], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
aug_t0 = aug_df['timestamp'].min()
aug_df['submission_day'] = (aug_df['timestamp'] - aug_t0).dt.total_seconds() / (24 * 3600.0)

aug_tr = aug_df.iloc[:int(0.8 * len(aug_df))]
aug_te = aug_df.iloc[int(0.8 * len(aug_df)):]
tree_b2_1 = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(random_state=RANDOM_STATE))]).fit(aug_tr[part_b_features], aug_tr['final_cgpa'])
b2_1_tr = root_mean_squared_error(aug_tr['final_cgpa'], tree_b2_1.predict(aug_tr[part_b_features]))
b2_1_te = root_mean_squared_error(aug_te['final_cgpa'], tree_b2_1.predict(aug_te[part_b_features]))

cols_dedup = [c for c in aug_df.columns if c not in ['timestamp', 'submission_day']]
dedup_df = aug_df.drop_duplicates(subset=cols_dedup).sort_values('timestamp').reset_index(drop=True)
dedup_df['submission_day'] = (dedup_df['timestamp'] - dedup_df['timestamp'].min()).dt.total_seconds() / (24 * 3600.0)
dedup_tr = dedup_df.iloc[:int(0.8 * len(dedup_df))]
dedup_te = dedup_df.iloc[int(0.8 * len(dedup_df)):]
tree_b2_2 = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(random_state=RANDOM_STATE))]).fit(dedup_tr[part_b_features], dedup_tr['final_cgpa'])
b2_2_tr = root_mean_squared_error(dedup_tr['final_cgpa'], tree_b2_2.predict(dedup_tr[part_b_features]))
b2_2_te = root_mean_squared_error(dedup_te['final_cgpa'], tree_b2_2.predict(dedup_te[part_b_features]))

print("\nB.2 Duplicate Leakage Summary:")
print(pd.DataFrame([
    {'Strategy': 'Splitting before removing duplicates (Leaked)', 'Training RMSE': b2_1_tr, 'Testing RMSE': b2_1_te},
    {'Strategy': 'Removing duplicates before splitting (Clean)', 'Training RMSE': b2_2_tr, 'Testing RMSE': b2_2_te}
]).to_string(index=False))

# B.3 Preprocessing leakage
imp_full = SimpleImputer(strategy='median')
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(imp_full.fit_transform(df_chrono[part_b_features]))
knn_leaked = KNeighborsRegressor(n_neighbors=10).fit(X_full_scaled[:split_idx], y_train)
b3_leak_tr = root_mean_squared_error(y_train, knn_leaked.predict(X_full_scaled[:split_idx]))
b3_leak_te = root_mean_squared_error(y_test, knn_leaked.predict(X_full_scaled[split_idx:]))

knn_clean = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('m', KNeighborsRegressor(n_neighbors=10))]).fit(X_train, y_train)
b3_clean_tr = root_mean_squared_error(y_train, knn_clean.predict(X_train))
b3_clean_te = root_mean_squared_error(y_test, knn_clean.predict(X_test))

print("\nB.3 Preprocessing Leakage Summary:")
print(pd.DataFrame([
    {'Strategy': 'StandardScaler on entire dataset (Leaked)', 'Training RMSE': b3_leak_tr, 'Testing RMSE': b3_leak_te},
    {'Strategy': 'StandardScaler inside Pipeline (Clean)', 'Training RMSE': b3_clean_tr, 'Testing RMSE': b3_clean_te}
]).to_string(index=False))

# ==============================================================================
# PART C. UNDERSTANDING AND DIAGNOSING OVERFITTING
# ==============================================================================
section_header("PART C. UNDERSTANDING AND DIAGNOSING OVERFITTING")

# C.1 Model complexity
depths = [1, 2, 3, 5, 10, 20, None]
c1_records = []
for d in depths:
    p = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE))]).fit(X_train, y_train)
    tr = root_mean_squared_error(y_train, p.predict(X_train))
    te = root_mean_squared_error(y_test, p.predict(X_test))
    c1_records.append({'max_depth': str(d), 'Train RMSE': tr, 'Test RMSE': te, 'Gen Gap': te - tr})
print("C.1 Tree Depth vs Errors:")
print(pd.DataFrame(c1_records).to_string(index=False))

# C.2 Feature complexity
degrees = [1, 2, 3, 5, 9, 15]
tscv = TimeSeriesSplit(n_splits=5)
def eval_tscv(pipe, X_data, y_data):
    scores = []
    for tr_i, val_i in tscv.split(X_data):
        pipe.fit(X_data.iloc[tr_i], y_data.iloc[tr_i])
        scores.append(root_mean_squared_error(y_data.iloc[val_i], pipe.predict(X_data.iloc[val_i])))
    return np.mean(scores)

c2_records = []
for deg in degrees:
    p = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('sc', StandardScaler()),
        ('m', LinearRegression())
    ])
    c2_records.append({'degree': deg, 'CV RMSE': eval_tscv(p, train_df[['effort_index']], y_train)})
print("\nC.2 Polynomial Degree vs CV RMSE:")
print(pd.DataFrame(c2_records).to_string(index=False))

# ==============================================================================
# PART D. IMPROVING MODEL GENERALIZATION
# ==============================================================================
section_header("PART D. IMPROVING MODEL GENERALIZATION")

# D.1 High-cardinality cohort_code
X_tr_d1 = train_df[part_b_features + ['cohort_code']]
ct_rare = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), part_b_features),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False))]), ['cohort_code'])
])
cv_rare = eval_tscv(Pipeline([('pre', ct_rare), ('m', DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE))]), X_tr_d1, y_train)

ct_target = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), part_b_features),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('te', TargetEncoder(cv=5, shuffle=False))]), ['cohort_code'])
])
cv_target = eval_tscv(Pipeline([('pre', ct_target), ('m', DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE))]), X_tr_d1, y_train)
cv_removal = eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE))]), train_df[part_b_features], y_train)

print(f"D.1 Rare-category grouping CV RMSE: {cv_rare:.4f}")
print(f"D.1 Target encoding CV RMSE:        {cv_target:.4f}")
print(f"D.1 Feature removal CV RMSE:        {cv_removal:.4f}")

# D.2 Feature selection: submission_day
feats_no_sub = [f for f in part_b_features if f != 'submission_day']
print("\nD.2 Impact of removing submission_day:")
for name, m in [('LinearRegression', LinearRegression()), ('DecisionTree(d=5)', DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)), ('GradientBoosting', GradientBoostingRegressor(n_estimators=60, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE))]:
    cv_with = eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('m', clone(m))]), train_df[part_b_features], y_train)
    cv_wo = eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('m', clone(m))]), train_df[feats_no_sub], y_train)
    print(f"  {name:20s}: With={cv_with:.4f}, Without={cv_wo:.4f}, Diff={cv_with - cv_wo:+.4f}")

# D.3 Regularization on polynomial degree 5
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = [eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('poly', PolynomialFeatures(degree=5, include_bias=False)), ('sc', StandardScaler()), ('m', Ridge(alpha=a, random_state=RANDOM_STATE))]), train_df[['effort_index']], y_train) for a in alphas]
lasso_scores = [eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('poly', PolynomialFeatures(degree=5, include_bias=False)), ('sc', StandardScaler()), ('m', Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=10000))]), train_df[['effort_index']], y_train) for a in alphas]

best_a_ridge = alphas[int(np.argmin(ridge_scores))]
best_a_lasso = alphas[int(np.argmin(lasso_scores))]
print(f"\nD.3 Best Ridge alpha: {best_a_ridge} (CV={min(ridge_scores):.4f}), Best Lasso alpha: {best_a_lasso} (CV={min(lasso_scores):.4f})")

# D.4 Final Models Comparison
final_features = feats_no_sub
X_tr_final = train_df[final_features]
y_tr_final = train_df['final_cgpa']
X_te_final = test_df[final_features]
y_te_final = test_df['final_cgpa']

# 1. Baseline
base_p = DummyRegressor(strategy="mean").fit(X_tr_final, y_tr_final)
base_tr, base_te, base_r2 = root_mean_squared_error(y_tr_final, base_p.predict(X_tr_final)), root_mean_squared_error(y_te_final, base_p.predict(X_te_final)), r2_score(y_te_final, base_p.predict(X_te_final))

# 2. Linear Regression
lr_p = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', LinearRegression())]).fit(X_tr_final, y_tr_final)
lr_tr, lr_te, lr_r2 = root_mean_squared_error(y_tr_final, lr_p.predict(X_tr_final)), root_mean_squared_error(y_te_final, lr_p.predict(X_te_final)), r2_score(y_te_final, lr_p.predict(X_te_final))

# 3. Ridge / Lasso
ridge_cv_final = [eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('m', Ridge(alpha=a, random_state=RANDOM_STATE))]), X_tr_final, y_tr_final) for a in alphas]
lasso_cv_final = [eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('m', Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=10000))]), X_tr_final, y_tr_final) for a in alphas]
use_ridge = min(ridge_cv_final) <= min(lasso_cv_final)
best_reg_alpha = alphas[int(np.argmin(ridge_cv_final))] if use_ridge else alphas[int(np.argmin(lasso_cv_final))]
reg_p = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('m', Ridge(alpha=best_reg_alpha, random_state=RANDOM_STATE) if use_ridge else Lasso(alpha=best_reg_alpha, random_state=RANDOM_STATE))]).fit(X_tr_final, y_tr_final)
reg_tr, reg_te, reg_r2 = root_mean_squared_error(y_tr_final, reg_p.predict(X_tr_final)), root_mean_squared_error(y_te_final, reg_p.predict(X_te_final)), r2_score(y_te_final, reg_p.predict(X_te_final))

# 4. Decision Tree
dt_scores = [eval_tscv(Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE))]), X_tr_final, y_tr_final) for d in depths]
best_depth = depths[int(np.argmin(dt_scores))]
dt_p = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE))]).fit(X_tr_final, y_tr_final)
dt_tr, dt_te, dt_r2 = root_mean_squared_error(y_tr_final, dt_p.predict(X_tr_final)), root_mean_squared_error(y_te_final, dt_p.predict(X_te_final)), r2_score(y_te_final, dt_p.predict(X_te_final))

# 5. Gradient Boosting with early stopping
gb_split = int((1 - 0.15) * len(train_df))
imp_gb = SimpleImputer(strategy='median')
X_gb_tr = imp_gb.fit_transform(train_df.iloc[:gb_split][final_features])
X_gb_val = imp_gb.transform(train_df.iloc[gb_split:][final_features])
y_gb_tr = train_df.iloc[:gb_split]['final_cgpa']
y_gb_val = train_df.iloc[gb_split:]['final_cgpa']

gb_search = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE).fit(X_gb_tr, y_gb_tr)
staged_rmses = [root_mean_squared_error(y_gb_val, p) for p in gb_search.staged_predict(X_gb_val)]
best_trees = int(np.argmin(staged_rmses)) + 1

gb_final = Pipeline([('imp', SimpleImputer(strategy='median')), ('m', GradientBoostingRegressor(n_estimators=best_trees, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE))]).fit(X_tr_final, y_tr_final)
gb_tr, gb_te, gb_r2 = root_mean_squared_error(y_tr_final, gb_final.predict(X_tr_final)), root_mean_squared_error(y_te_final, gb_final.predict(X_te_final)), r2_score(y_te_final, gb_final.predict(X_te_final))

section_header("FINAL DELIVERABLE SUMMARY TABLE (D.4)")
summary_df = pd.DataFrame([
    {'Model': 'Baseline (training mean)', 'Training RMSE': round(base_tr, 4), 'Testing RMSE': round(base_te, 4), 'Testing R²': round(base_r2, 4)},
    {'Model': 'Linear Regression', 'Training RMSE': round(lr_tr, 4), 'Testing RMSE': round(lr_te, 4), 'Testing R²': round(lr_r2, 4)},
    {'Model': f'Ridge / Lasso ({"Ridge" if use_ridge else "Lasso"} a={best_reg_alpha})', 'Training RMSE': round(reg_tr, 4), 'Testing RMSE': round(reg_te, 4), 'Testing R²': round(reg_r2, 4)},
    {'Model': f'Decision Tree (max_depth={best_depth})', 'Training RMSE': round(dt_tr, 4), 'Testing RMSE': round(dt_te, 4), 'Testing R²': round(dt_r2, 4)},
    {'Model': f'Gradient Boosting (n_trees={best_trees})', 'Training RMSE': round(gb_tr, 4), 'Testing RMSE': round(gb_te, 4), 'Testing R²': round(gb_r2, 4)}
])
print(summary_df.to_string(index=False))
print("\nSession completed successfully.")
