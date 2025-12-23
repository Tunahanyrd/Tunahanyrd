import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit 
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring
import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score, balanced_accuracy_score, roc_auc_score, roc_curve
from scipy.special import logit, expit
import numpy as np
from collections import defaultdict
eps=1e-7
path = pathlib.Path("data")

train = pd.read_csv(path/"train.csv", index_col="id")
test = pd.read_csv(path/"test.csv", index_col="id")
submission = pd.read_csv(path/"sample_submission.csv")

y = train["Personality"]

y = (y == "Introvert").astype(int)


train["Stage_fear"] = train["Stage_fear"].apply(lambda row: True if row == "Yes"
                                                else (False if row == "No" else np.nan))
train["Drained_after_socializing"] = train["Drained_after_socializing"].apply(
                                                lambda row: True if row == "Yes"
                                                else (False if row == "No" else np.nan))
X = train.drop(columns=["Personality"], axis=1)

test["Stage_fear"] = test["Stage_fear"].apply(lambda row: True if row == "Yes"
                                                else (False if row == "No" else np.nan))
test["Drained_after_socializing"] = test["Drained_after_socializing"].apply(
                                                lambda row: True if row == "Yes"
                                                else (False if row == "No" else np.nan))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_ffnn = np.zeros(len(X), dtype=np.float32)
oof_lgb  = np.zeros(len(X), dtype=np.float32)
oof_cat  = np.zeros(len(X), dtype=np.float32)
oof_xgb  = np.zeros(len(X), dtype=np.float32)

prep_tree = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ]), X.columns.tolist()),
    ],
    remainder="drop"
)
prep_nn = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), X.columns.tolist()),
    ],
    remainder="drop"
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLU(nn.Module):
    # Gated GELU (Shazeer, 2020) — kapılamalı doğrusal + GELU
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return F.gelu(a) * b

class ResidualBlock(nn.Module):
    # Residual MLP bloğu: GEGLU -> Dropout -> Linear -> LayerNorm + Residual
    def __init__(self, d_in, d_hidden, drop=0.25):
        super().__init__()
        self.gelu = GEGLU(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_in)
        self.ln = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(drop)
        # LayerScale (küçük gamma, residual’ı dengeler)
        self.gamma = nn.Parameter(torch.ones(d_in) * 0.1)

    def forward(self, x):
        h = self.gelu(x)
        h = self.drop(h)
        h = self.lin2(h)
        return self.ln(x + self.gamma * h)

class FFNN(nn.Module):
    """
    Tabular için sağlam MLP:
    - Input dropout (feature dropout)
    - 3 adet ResidualBlock (GEGLU kapılı)
    - Sonunda küçük bir head
    Not: BatchNorm yerine LayerNorm kullandım -> küçük batch’te daha stabil.
    """
    def __init__(self, in_dim, width=384, depth=3, drop=0.25, in_drop=0.05):
        super().__init__()
        self.in_drop = nn.Dropout(in_drop)
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(drop),
        )
        blocks = [ResidualBlock(width, width, drop=drop) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x):
        x = self.in_drop(x)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)  # raw logit

        
fold_auc = defaultdict(list)   
lgb_test_preds = []
cat_test_preds = []
ffnn_test_preds = []
xgb_test_preds = []        
for fold,(tr_idx,va_idx) in enumerate(kf.split(X, y), 1):
    print("Fold", fold)
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pos_weight = (y_tr.shape[0] - y_tr.sum()) / (y_tr.sum() + 1e-12)
    # FNNN      
    X_tr_nn = prep_nn.fit_transform(X_tr)
    X_va_nn = prep_nn.transform(X_va)
    test_nn = prep_nn.transform(test)
    auc = EpochScoring(
        scoring="roc_auc",
        lower_is_better=False,
        on_train=False,
        name="valid_auc"
    )
    net = NeuralNetBinaryClassifier(
        module=FFNN,
        module__in_dim=X_tr_nn.shape[1],
        max_epochs=50,
        batch_size=256,
        optimizer=optim.Adam,
        optimizer__lr=1e-3,
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32)),
        train_split=ValidSplit(0.1, stratified=True),
        device='cuda',
        callbacks=[
            auc,
            EarlyStopping(monitor='valid_loss', patience=5, load_best=True),
            LRScheduler(policy=optim.lr_scheduler.ReduceLROnPlateau,
                        monitor='valid_loss', patience=2, factor=0.5)
        ],
        verbose=0
    )
    
    net.fit(X_tr_nn.astype(np.float32), y_tr.values.astype(np.float32))
    p_va_ff = net.predict_proba(X_va_nn.astype(np.float32))[:, 1]
    p_te_ff = net.predict_proba(test_nn.astype(np.float32))[:, 1]
    oof_ffnn[va_idx] = logit(p_va_ff)
    ffnn_test_preds.append(logit(p_te_ff))

    # LightGBM
    X_tr_tree = prep_tree.fit_transform(X_tr)
    X_va_tree = prep_tree.transform(X_va)
    test_tree = prep_tree.transform(test)
    
    lgbm = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",          
    n_estimators=20000,
    learning_rate=0.03,
    num_leaves=31,                
    max_depth=-1,
    min_data_in_leaf=10,           
    min_sum_hessian_in_leaf=1e-3,  
    min_gain_to_split=0.0,         

    max_bin=255,
    min_data_in_bin=1,             
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    scale_pos_weight=pos_weight,         
    device_type="cpu",             
    force_col_wise=True,          
    random_state=42,
)
    lgbm.fit(X_tr_tree, y_tr,
           eval_set=[(X_va_tree, y_va)],
           eval_metric="auc",
           callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    
    oof_lgb[va_idx] = np.asarray(lgbm.predict(X_va_tree, raw_score=True))
    lgb_test_preds.append(np.asarray(lgbm.predict(test_tree, raw_score=True)))
    
    # CatBoost  
    catb = cat.CatBoostClassifier(
        task_type="GPU", devices="0",
        loss_function="Logloss", eval_metric="AUC",
        iterations=10000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, bagging_temperature=1.0,
        random_strength=0.2, border_count=128,
        early_stopping_rounds=300, verbose=False, random_state=42,
        scale_pos_weight=pos_weight
    )
        
    catb.fit(X_tr_tree, y_tr, eval_set=(X_va_tree, y_va),
           use_best_model=True)
    
    oof_cat[va_idx] = np.asarray(catb.predict(X_va_tree, 
                                                    prediction_type="RawFormulaVal"))
    cat_test_preds.append(np.asarray(catb.predict(test_tree, 
                                                        prediction_type="RawFormulaVal")))
    
    dtr= xgb.DMatrix(X_tr_tree, label=y_tr)
    dva= xgb.DMatrix(X_va_tree, label=y_va)
    dte= xgb.DMatrix(test_tree)
    
    params = dict(
        objective="binary:logitraw",
        eval_metric="auc",
        device="cuda",
        tree_method="hist",
        max_depth=6,
        eta=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=42,
        scale_pos_weight=pos_weight
    )
    
    xb = xgb.train(params, dtr, num_boost_round=20000,
                    evals=[(dva, "valid")], early_stopping_rounds=300, verbose_eval=False)
    
    p_va = xb.predict(dva, iteration_range=(0, xb.best_iteration+1))
    p_te = xb.predict(dte, iteration_range=(0, xb.best_iteration+1))
    oof_xgb[va_idx] = p_va
    xgb_test_preds.append(p_te)
    
    
    auc_ff = roc_auc_score(y_va, p_va_ff)

    auc_lgb_fold = roc_auc_score(y_va, expit(oof_lgb[va_idx]))

    auc_cat_fold = roc_auc_score(y_va, expit(oof_cat[va_idx]))

    auc_xgb_fold = roc_auc_score(y_va, expit(oof_xgb[va_idx]))

    fold_auc["ffnn"].append(auc_ff)
    fold_auc["lgb"].append(auc_lgb_fold)
    fold_auc["cat"].append(auc_cat_fold)
    fold_auc["xgb"].append(auc_xgb_fold)

    print(f"[Fold {fold}] AUC -> FFNN: {auc_ff:.5f} | LGBM: {auc_lgb_fold:.5f} | Cat: {auc_cat_fold:.5f} | XGB: {auc_xgb_fold:.5f}")

oof_auc_ff = roc_auc_score(y, expit(np.clip(oof_ffnn, eps, 1-eps)))  
oof_auc_lg = roc_auc_score(y, expit(np.clip(oof_lgb, eps, 1-eps)))  
oof_auc_ct = roc_auc_score(y, expit(np.clip(oof_cat, eps, 1-eps)))   
oof_auc_xg = roc_auc_score(y, expit(np.clip(oof_xgb, eps, 1-eps)))   

print("\n=== OOF AUC (base) ===")
print(f"FFNN: {oof_auc_ff:.6f} | LGBM: {oof_auc_lg:.6f} | Cat: {oof_auc_ct:.6f} | XGB: {oof_auc_xg:.6f}")

ffnn_test_pred = np.mean(np.vstack(ffnn_test_preds), axis=0)
lgb_test_pred  = np.mean(np.vstack(lgb_test_preds),  axis=0)
cat_test_pred  = np.mean(np.vstack(cat_test_preds),  axis=0)
xgb_test_pred  = np.mean(np.vstack(xgb_test_preds), axis=0)

Z_oof = np.vstack([oof_ffnn, oof_cat, oof_lgb, oof_xgb]).T
Z_test = np.vstack([ffnn_test_pred, cat_test_pred, lgb_test_pred, xgb_test_pred]).T

"""
meta = LogisticRegression(
    penalty="l2", C=1., solver="lbfgs", max_iter=1000,
    class_weight="balanced", random_state=42
)
meta.fit(Z_oof, y)
meta = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=7,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
meta.fit(Z_oof, y)
"""
del X_tr, X_va, y_tr, y_va, X_tr_nn, X_va_nn
Z_oof_df  = pd.DataFrame(Z_oof,  index=X.index,  columns=[f"m{i}" for i in range(Z_oof.shape[1])])
Z_test_df = pd.DataFrame(Z_test, index=test.index, columns=[f"m{i}" for i in range(Z_test.shape[1])])

X_meta_train = pd.concat([Z_oof_df, X], axis=1)
X_meta_test  = pd.concat([Z_test_df, test], axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_meta_train, y, test_size=0.1, stratify=y, random_state=42
)
auc = EpochScoring(
    scoring="roc_auc",
    lower_is_better=False,
    on_train=False,
    name="valid_auc"
)
meta = NeuralNetBinaryClassifier(
    module=FFNN,
    module__in_dim=X_train.shape[1],
    max_epochs=100,
    batch_size=256,
    optimizer=optim.AdamW,
    optimizer__lr=1e-3,
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32)),
    train_split=ValidSplit(0.1, stratified=True),
    device='cuda',
    callbacks=[
        auc,
        EarlyStopping(monitor='valid_loss', patience=5, load_best=True),
        LRScheduler(policy=optim.lr_scheduler.ReduceLROnPlateau,
                    monitor='valid_loss', patience=2, factor=0.5)
    ],
    verbose=0
)
imp = SimpleImputer(strategy="median")
sc  = StandardScaler()

X_tr_imp = imp.fit_transform(X_train)
X_va_imp = imp.transform(X_val)
X_te_imp = imp.transform(X_meta_test)

X_train_nn = sc.fit_transform(X_tr_imp)
X_val_nn = sc.transform(X_va_imp)
X_test_nn = sc.transform(X_te_imp)

pos_weight = (y_train.shape[0] - y_train.sum()) / (y_train.sum() + 1e-12)

meta.fit(X_train_nn.astype(np.float32), y_train.values.astype(np.float32))
p_val = meta.predict_proba(X_val_nn.astype(np.float32))[:, 1]
print("META valid AUC:", roc_auc_score(y_val, p_val))

p_tr  = meta.predict_proba(X_train_nn.astype(np.float32))[:, 1]
print("META train(resub) AUC:", roc_auc_score(y_train, p_tr))

prec, rec, thr = precision_recall_curve(y_val, p_val)
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_idx = np.nanargmax(f1)
best_thr_f1 = thr[best_idx]
print("Best oof f1:", np.nanmax(f1), "@ thr=", best_thr_f1)

fpr, tpr, thr2 = roc_curve(y_val, p_val)
youden = tpr - fpr
best_thr_bacc = thr2[np.argmax(youden)]
print("Best oof balancedacc thr:", best_thr_bacc)
best_thr = best_thr_f1

p_te = meta.predict_proba(X_test_nn.astype(np.float32))[:, 1]
y_test_hat = (p_te >= best_thr).astype(int)

sub = submission.copy()
sub["Personality"] = np.where(y_test_hat == 1, "Introvert", "Extrovert")
sub.to_csv("submission.csv", index=False)
print("saved -> submission.csv")


