
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

# Load dataset
lung = pd.read_csv("lung_disease.csv")

# Crosstab
print(pd.crosstab(lung['Smoking'], lung['LungDisease']))

# Chi-square test
ct = pd.crosstab(lung['Smoking'], lung['LungDisease'])
chi = chi2_contingency(ct)
print(chi)

# Logistic Regression
lung['LD'] = lung['LungDisease'].map({'Yes':1,'No':0})

model = smf.logit(
    'LD ~ C(Smoking) + Age + C(Pollution) + C(Income)',
    data=lung
).fit()

print(model.summary())

# AUC
pred = model.predict(lung)
auc = roc_auc_score(lung['LD'], pred)
print("AUC =", auc)
