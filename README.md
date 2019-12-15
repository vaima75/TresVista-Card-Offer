## TresVista - Card-Offer-Promotion


### Chi Square Test Results

The data given is of credit records of individuals with certain attributes. Please go through following to understand the variables involved:

#### Var-Var Matrix

The below matrix represented in the following format:
df, X-squared, p-value where,

- df = degrees of freedom
- X-squared = Chi-Square Value, and
- P-Value of Chi statistic

| Var/Var | demographic_slice | country_reg | ad_exp | card_offer |
|:------------- |:------------- |:------------- |:-------------|:-------------|
| demographic_slice | . | 3 , 3.2525 , 0.3543 | 3 , 7.6743 , 0.05325 | 3 , 487.11 , < 2.2e-16 | 
| country_reg | 3 , 3.2525 , 0.3543 | . | 1 , 0.010034 , 0.9202 | 1 , 176.85 , < 2.2e-16 |
| ad_exp | 3 , 7.6743 , 0.05325 |  1 , 0.010034 , 0.9202 | . |  1 , 0.018046 , 0.8931 |
| card_offer | 3 , 487.11 , < 2.2e-16 | 1 , 176.85 , < 2.2e-16 | 1 , 0.018046 , 0.8931 | . |

#### Confusion Matrix Statistics

For details and definition refer [here](https://github.com/vaima75/TresVista-Card-Offer/blob/master/ConfMat.md)

| Parameters | RandomForest | BAGGING | GBM | C50 | logisticReg | Naive bayes |
|:------------- |:------------- |:------------- |:------------- |:------------- |:------------- |:------------- |
| Accuracy	| 0.9824 | 0.978 | 0.9832 |  0.9848 | 0.9684 | 0.8896 |
| 95% CI | 0.9764 , 0.9872 | 0.9714 , 0.9834 | 0.9773 , 0.9879 | 0.9792 , 0.9892 | 0.9608 , 0.9749 | 0.8766 , 0.9016 |
| No Information Rate | 0.8471	| 0.8471 | 0.8471 | 0.8471 | 0.8471 | 0.8471 |
| P-Value (Acc > NIR) | <2e-16	| <2e-16 | < 2e-16 | < 2e-16 | < 2e-16 | 4.93e-10 |
| Kappa | 0.9313 |  0.9144 | 0.9343 | 0.941 | 0.8759 | 0.3942 |
| Mcnemar's Test P-Value | 0.1748 | 0.4185 | 0.08963 | 0.6265 | 0.1152 | < 2.2e-16 |
| Sensitivity | 0.9920 | 0.9887 | 0.9929 | 0.9920 | 0.9849 | 1.0000 |
| Specificity | 0.9293 | 0.9188 | 0.9293 | 0.9450 | 0.8770 | 0.2775 |
| Pos Pred Value | 0.9873 | 0.9854 | 0.9873 | 0.9901 | 0.9780 | 0.8847 |
| Neg Pred Value | 0.9543 | 0.9360 | 0.9595 | 0.9550 | 0.9128 | 1.0000 |
| Prevalence | 0.8471 |  0.8471 | 0.8471 | 0.8471 | 0.8471 | 0.8471 |
| Detection Rate | 0.8403 |  0.8375 | 0.8411 | 0.8403 | 0.8343 | 0.8471 |
| Detection Prevalence | 0.8511 | 0.8499 | 0.8519 | 0.8487 | 0.8531 | 0.9576 |
| Balanced Accuracy | 0.9606 | 0.9538 | 0.9611 | 0.9685 | 0.9309 | 0.6387 |
| ROC – AUC Results | 0.9978 | 0.9955 | 0.9983 | 0.9988 | 0.994 | 0.9906 |
| 'Positive' Class | No | No | No | No | No | No |


### Evaluation Metric

- Accuracy
- Area under Curve (AUC)