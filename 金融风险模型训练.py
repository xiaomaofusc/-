import pandas as pd
import numpy as np
import gc
import time
import re
import datetime
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

def process_employment_length(value):
    if pd.isna(value) or str(value).lower() in ['nan', 'n/a', 'null']:
        return np.nan
    if isinstance(value, (int, float)):
        return min(max(value, 0), 50)
    value = str(value).lower()
    #快速值映射
    mapping = {
        '<1 year': 0.5, '< 1 year': 0.5,
        '1 year': 1.0, '2 years': 2.0,
        '3 years': 3.0, '4 years': 4.0,
        '5 years': 5.0, '6 years': 6.0,
        '7 years': 7.0, '8 years': 8.0,
        '9 years': 9.0, '10+years': 11.0,
        '10+ years': 11.0, '15+ years': 16.0,
        '20+ years': 21.0
    }
    if value in mapping:
        return mapping[value]
    #尝试提取数字
    match = re.search(r'(\d+)', value)
    return float(match.group(1)) if match else np.nan

def extract_year(value):
    try:
        value_str = str(value)
        match = re.search(r'(\d{4})', value_str)
        return int(match.group(1)) if match else np.nan
    except:
        return np.nan
class RobustFeatureEngineer:
    def __init__(self):
        self.imputation_values = {}
        self.scaler = None
        self.categorical_cols = ['grade', 'subGrade', 'purpose', 'regionCode', 'homeOwnership']
        self.label_encoders = {}

    def fit(self, df):
        #数值列处理
        numeric_cols = self._get_numeric_cols(df)
        for col in numeric_cols:
            if df[col].isna().any():
                self.imputation_values[col] = df[col].median()
        #为类别列准备编码器
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str).fillna('unknown'))
                self.label_encoders[col] = le

    def add_stat_features(self, df):
        # 增加统计特征
        if 'annualIncome' in df and 'loanAmnt' in df:
            df['income_loan_ratio'] = df['annualIncome'] / (df['loanAmnt'] + 1)
        if 'revolBal' in df and 'annualIncome' in df:
            df['revol_income_ratio'] = df['revolBal'] / (df['annualIncome'] + 1)
        if 'openAcc' in df and 'totalAcc' in df:
            df['open_total_ratio'] = df['openAcc'] / (df['totalAcc'] + 1)
        if 'dti' in df and 'annualIncome' in df:
            df['dti_income'] = df['dti'] * df['annualIncome']
        # 分箱特征
        if 'interestRate' in df:
            df['interest_bin'] = pd.qcut(df['interestRate'], 15, labels=False, duplicates='drop')
        # 分组统计特征
        for col in ['regionCode', 'grade', 'purpose']:
            if col in df and 'annualIncome' in df:
                group_mean = df.groupby(col)['annualIncome'].transform('mean')
                df[f'{col}_income_diff'] = df['annualIncome'] - group_mean
            if col in df and 'loanAmnt' in df:
                group_mean = df.groupby(col)['loanAmnt'].transform('mean')
                df[f'{col}_loan_diff'] = df['loanAmnt'] - group_mean
        # 统计排名特征
        if 'annualIncome' in df:
            df['income_rank'] = df['annualIncome'].rank(pct=True)
        if 'loanAmnt' in df:
            df['loan_rank'] = df['loanAmnt'].rank(pct=True)
        return df

    def add_interaction_features(self, df):
        # 增加交互特征
        if 'grade_enc' in df and 'purpose_enc' in df:
            df['grade_purpose'] = df['grade_enc'] * 10 + df['purpose_enc']
        if 'regionCode_enc' in df and 'homeOwnership_enc' in df:
            df['region_home'] = df['regionCode_enc'] * 10 + df['homeOwnership_enc']
        # 多项式交互
        if 'fico_avg' in df and 'interestRate' in df:
            df['fico_interest'] = df['fico_avg'] * df['interestRate']
        if 'annualIncome' in df and 'dti' in df:
            df['income_dti'] = df['annualIncome'] * df['dti']
        # 新增：高阶交互
        if 'income_loan_ratio' in df and 'dti' in df:
            df['income_loan_dti'] = df['income_loan_ratio'] * df['dti']
        if 'revol_income_ratio' in df and 'fico_avg' in df:
            df['revol_fico'] = df['revol_income_ratio'] * df['fico_avg']
        return df

    def transform(self, df):
        df = df.copy().drop(columns=['id'], errors='ignore')
        df = self._process_dates(df)
        #特殊列处理
        if 'employmentLength' in df.columns:
            df['employmentLength'] = df['employmentLength'].apply(process_employment_length)
            df['employmentLength'] = df['employmentLength'].fillna(df['employmentLength'].median())
        #关键特征组合
        if 'dti' in df and 'annualIncome' in df:
            df['debt_burden'] = df['dti'] * df['annualIncome'] / 1000
        if 'annualIncome' in df and 'loanAmnt' in df:
            df['income_to_loan'] = df['annualIncome'] / (df['loanAmnt'] + 1)
        if 'ficoRangeLow' in df and 'ficoRangeHigh' in df:
            df['fico_avg'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2
        if 'revolUtil' in df:
            df['high_utilization'] = (df['revolUtil'] > 80).astype(int)
        if 'openAcc' in df and 'totalAcc' in df:
            df['closed_account_ratio'] = (df['totalAcc'] - df['openAcc']) / (df['totalAcc'] + 1)

        #缺失值处理
        for col, value in self.imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        #类别编码
        for col in self.categorical_cols:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    df[col] = df[col].astype(str).fillna('unknown')
                    #处理未知类别
                    unknown_mask = ~df[col].isin(le.classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = 'unknown'
                        if 'unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'unknown')
                    df[f'{col}_enc'] = le.transform(df[col])
                    df.drop(col, axis=1, inplace=True)
        #异常值处理
        numeric_cols = self._get_numeric_cols(df)
        for col in numeric_cols:
            #填充缺失值
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
            #基于IQR的鲁棒异常值处理
            q1, q3 = df[col].quantile([0.05, 0.95])
            if q3 > q1:  #确保IQR有效
                iqr = q3 - q1
                lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = np.clip(df[col], lb, ub)
        #特征标准化
        if len(numeric_cols) > 0:
            if self.scaler is None:
                self.scaler = StandardScaler()
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        #确保没有object类型列
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            #尝试转换到数值类型
            for col in object_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = pd.factorize(df[col])[0]
        #最终检查-移除任何非数值列
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            df = df.drop(columns=non_numeric)

        # 增加统计特征
        df = self.add_stat_features(df)
        # 类别编码后增加交互特征
        df = self.add_interaction_features(df)

        return df.fillna(0)

    def _get_numeric_cols(self, df):
        #获取数值列，排除目标列
        return [col for col in df.select_dtypes(include=np.number).columns
                if col not in ['isDefault', 'y']]

    def _process_dates(self, df):
        #处理所有日期列
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        #已经是日期类型，提取特征
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df.drop(col, axis=1, inplace=True)
                    else:
                        #尝试转换为日期
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df.drop(col, axis=1, inplace=True)
                except:
                    #无法转换日期，尝试提取年份
                    df[col] = df[col].apply(extract_year)
        #处理包含"credit"关键字的列
        for col in df.columns:
            if 'credit' in col.lower():
                df[col] = df[col].apply(extract_year)
                if col.lower() in ['earliestcreditline', 'earliescreditline']:
                    current_year = datetime.datetime.now().year
                    df['credit_history_years'] = current_year - df[col]
                    df.drop(col, axis=1, inplace=True)

        return df

    def select_important_features(self, X, y, threshold='median'):
        # 用随机森林筛选重要特征
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features

def train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight):
    #训练和评估LightGBM模型（启用GPU）    pip uninstall xgboost -y
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1,
        'device': 'gpu',  # 启用GPU
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    model = lgb.LGBMClassifier(**params, n_estimators=2000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    return model

def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    #训练和评估XGBoost模型（启用GPU）
    params = {
        'objective': 'binary:logistic',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # 新版XGBoost推荐
        'device': 'cuda',       # 显式指定GPU
        'eval_metric': 'auc',
        'use_label_encoder': False
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=0
    )
    return model

def train_catboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    #训练和评估CatBoost模型（启用GPU）
    params = {
        'iterations': 2000,
        'learning_rate': 0.05,
        'depth': 7,
        'l2_leaf_reg': 3,
        'random_state': 42,
        'silent': True,
        'auto_class_weights': 'Balanced',
        'task_type': 'GPU'  # 启用GPU
    }
    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=0
    )
    return model

def robust_cross_validation(X, y, test_df, n_folds=5):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_folds}")
        start_time = time.time()
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fe = RobustFeatureEngineer()
        fe.fit(X_train)
        X_train_fe = fe.transform(X_train)
        X_val_fe = fe.transform(X_val)
        test_fold_fe = fe.transform(test_df)
        selected_features = fe.select_important_features(X_train_fe, y_train, threshold='median')
        X_train_fe = X_train_fe[selected_features]
        X_val_fe = X_val_fe[selected_features]
        test_fold_fe = test_fold_fe[selected_features]
        # stacking集成
        base_models = [
            ('lgb', lgb.LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=800, learning_rate=0.05, random_state=42)),
            ('xgb', xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=800, learning_rate=0.05, random_state=42, use_label_encoder=False)),
            ('cb', cb.CatBoostClassifier(task_type='GPU', iterations=800, learning_rate=0.05, random_state=42, silent=True))
        ]
        stack_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=200),
            n_jobs=1,  # 避免CatBoost多进程争抢GPU
            passthrough=True
        )
        stack_model.fit(X_train_fe, y_train)
        val_preds = stack_model.predict_proba(X_val_fe)[:, 1]
        oof_preds[val_idx] = val_preds
        test_fold_preds = stack_model.predict_proba(test_fold_fe)[:, 1]
        test_preds += test_fold_preds / n_folds
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}, Acc: {fold_acc:.5f}, Time: {time.time() - start_time:.1f}s")
    oof_auc = roc_auc_score(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    print(f"\nOverall OOF AUC: {oof_auc:.5f}, Acc: {oof_acc:.5f}")
    return test_preds, oof_auc, oof_acc


def main():
    print("金融风险预测模型")
    start_time = time.time()
    #加载数据
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('testA.csv')
        print(f"数据加载完成: 训练集 {train.shape}, 测试集 {test.shape}")
        print(f"训练集列名: {train.columns.tolist()}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    #准备数据
    target_col = 'isDefault' if 'isDefault' in train.columns else 'y'
    y = train[target_col]
    X = train.drop(columns=[target_col, 'id'], errors='ignore')
    test_id = test['id']
    #交叉验证训练
    test_preds, oof_auc, oof_acc = robust_cross_validation(X, y, test, n_folds=5)
    #概率校准
    test_preds = np.clip(test_preds, 0.01, 0.99)
    alpha, beta = 1.5, 1.5
    calibrated_preds = (test_preds * alpha) / (test_preds * alpha + (1 - test_preds) * beta)
    #生成提交
    submission = pd.DataFrame({'id': test_id, 'isDefault': calibrated_preds})
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f'submission_auc_{oof_auc:.5f}_{timestamp}.csv'
    submission.to_csv(submission_file, index=False)
    #性能报告
    mins = (time.time() - start_time) / 60
    print(f"最终AUC: {oof_auc:.5f}")
    print(f"准确率: {oof_acc:.5f}")
    print(f"提交文件: {submission_file}")

if __name__ == "__main__":
    main()

# Optuna调参增强
import optuna

def optuna_objective(trial, X, y):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'scale_pos_weight': np.sum(y == 0) / np.sum(y == 1),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = lgb.LGBMClassifier(**params, n_estimators=1500)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

def run_optuna_tuning(X, y, n_trials=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=n_trials)
    print('Best params:', study.best_params)
    print('Best AUC:', study.best_value)
    return study.best_params

    #交叉验证训练前可选自动调参
    #best_params = run_optuna_tuning(X, y, n_trials=30)
    #print('Optuna最优参数:', best_params)