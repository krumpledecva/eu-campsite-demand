"""Run this once to generate classifier.pkl, regressor.pkl, campsite_data.csv."""
import warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
import eurostat
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

SEED = 42

print("Downloading Eurostat data...")
raw = eurostat.get_data_df('tour_occ_nim')
mask = (raw['nace_r2']=='I553') & (raw['c_resid']=='TOTAL') & (raw['unit']=='NR')
df_camp = raw[mask].copy()
geo_col = 'geo\\TIME_PERIOD'
time_cols = [c for c in df_camp.columns if str(c).startswith('20')]
df_long = df_camp.melt(id_vars=geo_col, value_vars=time_cols, var_name='time', value_name='nights_spent')
df_long = df_long.rename(columns={geo_col: 'geo'})
df_long['year']  = df_long['time'].str[:4].astype(int)
df_long['month'] = df_long['time'].str[5:7].astype(int)
df_long = df_long[df_long['geo'].str.len()==2].dropna(subset=['nights_spent'])
df_long = df_long[df_long['nights_spent']>0][df_long['year']>=2010]
df = df_long[['geo','year','month','nights_spent']].sort_values(['geo','year','month']).reset_index(drop=True)
print(f"Raw data: {df.shape}")

full_grid = pd.MultiIndex.from_product(
    [df['geo'].unique(), range(df['year'].min(), df['year'].max()+1), range(1,13)],
    names=['geo','year','month']
).to_frame(index=False)
df_coverage = full_grid.merge(df, on=['geo','year','month'], how='left')
df_clean = df_coverage.sort_values(['geo','year','month']).copy()
df_clean['nights_spent'] = df_clean.groupby('geo')['nights_spent'].transform(lambda s: s.ffill().bfill())
df_clean = df_clean.dropna(subset=['nights_spent']).reset_index(drop=True)
df_clean['is_covid'] = df_clean['year'].isin([2020,2021]).astype(int)

df_feat = df_clean.sort_values(['geo','year','month']).copy()
df_feat['lag_1m']  = df_feat.groupby('geo')['nights_spent'].shift(1)
df_feat['lag_12m'] = df_feat.groupby('geo')['nights_spent'].shift(12)
df_feat = df_feat.dropna(subset=['lag_1m','lag_12m']).reset_index(drop=True)
df_feat['is_summer'] = df_feat['month'].isin([6,7,8]).astype(int)
df_feat['is_july']   = (df_feat['month']==7).astype(int)
df_feat['year_rel']  = df_feat['year'] - 2010
df_feat['month_sin'] = np.sin(2*np.pi*df_feat['month']/12)
df_feat['month_cos'] = np.cos(2*np.pi*df_feat['month']/12)
print(f"Feature dataset: {df_feat.shape}")

FEATURES = ['geo','year_rel','month','month_sin','month_cos','lag_1m','lag_12m','is_summer','is_july','is_covid']
num_cols  = ['year_rel','month','month_sin','month_cos','lag_1m','lag_12m','is_summer','is_july','is_covid']
cat_cols  = ['geo']

df_sorted = df_feat.sort_values(['year','month','geo']).reset_index(drop=True)
cutoff = int(len(df_sorted)*0.80)
df_train = df_sorted.iloc[:cutoff].copy()
df_test  = df_sorted.iloc[cutoff:].copy()

low_thresh  = df_train.groupby('geo')['nights_spent'].quantile(0.333)
high_thresh = df_train.groupby('geo')['nights_spent'].quantile(0.667)
def assign_demand_level(row):
    lo = low_thresh.get(row['geo'], low_thresh.median())
    hi = high_thresh.get(row['geo'], high_thresh.median())
    if row['nights_spent'] <= lo: return 0
    elif row['nights_spent'] <= hi: return 1
    else: return 2
df_train['demand_level'] = df_train.apply(assign_demand_level, axis=1)
df_test['demand_level']  = df_test.apply(assign_demand_level, axis=1)

X_train = df_train[FEATURES]; X_test = df_test[FEATURES]
y_train_clf = df_train['demand_level']; y_test_clf = df_test['demand_level']
y_train_reg = df_train['nights_spent']; y_test_reg = df_test['nights_spent']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
])
def clf_pipeline(m): return Pipeline([('prep', preprocessor), ('clf', m)])
def reg_pipeline(m): return Pipeline([('prep', preprocessor), ('reg', m)])

print("Training XGBoost classifier...")
xgb_search = GridSearchCV(
    clf_pipeline(XGBClassifier(eval_metric='mlogloss', random_state=SEED, verbosity=0)),
    {'clf__n_estimators':[100,300], 'clf__max_depth':[3,5], 'clf__learning_rate':[0.05,0.1]},
    cv=3, scoring='f1_macro', n_jobs=-1)
xgb_search.fit(X_train, y_train_clf)
xgb_clf = xgb_search.best_estimator_
print(f"  Best params: {xgb_search.best_params_}")

print("Training XGBoost regressor...")
xgb_reg_search = GridSearchCV(
    reg_pipeline(XGBRegressor(random_state=SEED, verbosity=0)),
    {'reg__n_estimators':[100,300], 'reg__max_depth':[3,5], 'reg__learning_rate':[0.05,0.1]},
    cv=3, scoring='r2', n_jobs=-1)
xgb_reg_search.fit(X_train, y_train_reg)
xgb_reg = xgb_reg_search.best_estimator_
print(f"  Best params: {xgb_reg_search.best_params_}")

joblib.dump(xgb_clf, 'classifier.pkl')
joblib.dump(xgb_reg, 'regressor.pkl')
df_feat[['geo','year','month','nights_spent','lag_1m','lag_12m']].to_csv('campsite_data.csv', index=False)

print("\nSaved: classifier.pkl")
print("Saved: regressor.pkl")
print("Saved: campsite_data.csv")

# Smoke test
sample = X_test.iloc[[0]]
print(f"\nSmoke test: geo={sample['geo'].values[0]}, month={sample['month'].values[0]}, year_rel={sample['year_rel'].values[0]}")
print(f"  Demand level : {['Low','Medium','High'][xgb_clf.predict(sample)[0]]}")
print(f"  Nights est.  : {xgb_reg.predict(sample)[0]:,.0f}")
print("\nDone. All artifacts ready.")
