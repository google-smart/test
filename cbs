import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 读取数据
data = pd.read_csv('cc.csv')

# 特征工程
# 添加更多的技术指标
data['balance_to_credit_limit_ratio'] = data['LIMIT_BAL'] / data['BILL_AMT1']
data['payment_to_bill_ratio'] = data['PAY_AMT1'] / data['BILL_AMT1']
data['credit_utilization'] = (data['BILL_AMT1'] - data['PAY_AMT1']) / data['LIMIT_BAL']

# 特征选择
X = data[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
          'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'PAY_AMT1',
          'balance_to_credit_limit_ratio', 'payment_to_bill_ratio', 'credit_utilization']]
y = data['default']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建随机森林分类器
clf = RandomForestClassifier(random_state=42)

# 交叉验证评估模型性能
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(f'Cross Validation Scores: {scores}')
print(f'Mean Accuracy: {np.mean(scores)}')

# 拟合模型
clf.fit(X_train, y_train)

# 预测测试集结果
y_pred = clf.predict(X_test)

# 输出模型性能
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 使用交叉验证评估模型性能
scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
print("Cross-validation F1 score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 使用自动化调参来优化模型
params = {'n_estimators': [50, 100, 200, 400],
          'max_depth': [5, 10, 20, None],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)
print("Best parameters found on training set:")
print(grid_search.best_params_)
print("Grid scores on training set:")
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
# 训练使用最佳超参数的模型
best_model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                     max_depth=grid_search.best_params_['max_depth'],
                                     min_samples_split=grid_search.best_params_['min_samples_split'],
                                     min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
best_model.fit(X_train, y_train)

# 用测试集对最佳模型进行评估
y_pred = best_model.predict(X_test)
print("Best model performance on test set:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 计算并打印出特征的重要性排名
feature_importances = best_model.feature_importances_
sorted_idx = feature_importances.argsort()
barh(y_train.columns[sorted_idx], feature_importances[sorted_idx])
xlabel("Random Forest Feature Importance")

# 绘制ROC曲线和计算AUC
y_pred_proba = best_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, roc_auc)
