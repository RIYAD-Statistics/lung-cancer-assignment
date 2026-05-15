
# Load Dataset
lung <- read.csv("lung_disease.csv")

# Crosstab
table(lung$Smoking, lung$LungDisease)

# Chi-square test
chisq.test(table(lung$Smoking, lung$LungDisease))

# Logistic Regression
lung$LD <- ifelse(lung$LungDisease=="Yes",1,0)

model <- glm(
LD ~ Smoking + Age + Pollution + Income,
data=lung,
family=binomial
)

summary(model)

# ROC & AUC
library(pROC)

pred <- predict(model, type="response")
roc_obj <- roc(lung$LD, pred)
auc(roc_obj)
