getwd()
rm(list = ls())

library(caret)
library(randomForest)
train = read.csv("Test 1.csv")
head(train)
colSums((is.na(train))) # No null Values
train$card_offer = ifelse(train$card_offer == TRUE, "Yes","No")
train$card_offer = as.factor(train$card_offer)

train$customer_id = NULL
names(train)
### EDA ###

names(train)
str(train)
summary(train)

## Income
# Histogram
ggplot(train, aes(x = est_income, fill = demographic_slice)) + 
  geom_histogram(show.legend = F, bins = 20) + 
  facet_wrap(~demographic_slice) + 
  labs(x = "Demography",y = "Freq") + 
  ggtitle("Income Distribution across Geo locations")

ggplot(train, aes(x = est_income)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~card_offer+~demographic_slice) +
  ggtitle("Income Distribution w.r.t Geo locations and Card Offer")

ggplot(train, aes(x = est_income)) + 
  geom_histogram(bins = 20) + 
  facet_wrap(~country_reg+~ad_exp) +
  ggtitle("Income Distribution w.r.t Region and ad expense")


# Boxplot
ggplot(train, aes(x = train$demographic_slice,y = train$est_income)) + 
  geom_boxplot() +
  facet_wrap(~train$card_offer)


# scatter Plot 
cval_cscore_plot = ggplot(train, aes(x = imp_crediteval,y = imp_cscore,
                  colour = card_offer)) + geom_point(show.legend = T) + labs(x = "Credit Value", y = "Credit Score") + ggtitle("Credit Value V/S Score")
cval_cscore_plot + facet_wrap(~demographic_slice) + ggtitle("Geo-wise Credit Value V/S Score")
cval_cscore_plot + facet_wrap(~ad_exp) + ggtitle("Ad-Expense-wise Credit Value V/S Score")
cval_cscore_plot + facet_wrap(~country_reg) + ggtitle("Region-wise Credit Value V/S Score")
cval_cscore_plot + facet_wrap(~ad_exp + ~country_reg) + ggtitle("Ad & Region Wise Credit Value V/S Score")


bal_cscore_plot = ggplot(train, aes(x = hold_bal,y = imp_cscore,
                                     colour = card_offer)) + geom_point(show.legend = T) + labs(x = "Hold Balance", y = "Credit Score") 
bal_cscore_plot + facet_wrap(~demographic_slice) + ggtitle("Geo-wise Credit Balance V/S Score")
bal_cscore_plot + facet_wrap(~ad_exp) + ggtitle("Ad-Expense-wise Credit Balance V/S Score")
bal_cscore_plot + facet_wrap(~country_reg) + ggtitle("Region-wise Credit Balance V/S Score")
bal_cscore_plot + facet_wrap(~ad_exp + ~country_reg) + ggtitle("Ad & Region Wise Credit Balance V/S Score")


#### Correlation Matrix ####
corr_data = data.frame(a = NA)
for (i in 1:ncol(train)) {
  if (class(train[,i]) == "numeric" | class(train[,i]) == "integer") {
    corr_data = cbind(corr_data, train[i])
  }
}
corr_data$a = NULL
names(corr_data)
write.csv(cor(corr_data)*100, "Correlation Matrix.csv")
View(cor(corr_data)*100)
# From correlation matrix, among ratio variables, we observed that
# "imp_crediteval" Vs "imp_cscore" 
# are highly corelated variables (92.6908%)


corr_oth_data = data.frame(a = NA)
for (i in 1:ncol(train)) {
  if (class(train[,i]) == "factor") {
    corr_oth_data = cbind(corr_oth_data, train[i])
  }
}
corr_oth_data$a = NULL
names(corr_oth_data)

for (c1 in 1:ncol(corr_oth_data)) {
  for (c2 in 1:ncol(corr_oth_data)) {
    if (c1 != c2) {
      cat("c1 --> ",names(corr_oth_data[c1]),"\nc2 --> ",names(corr_oth_data[c2]),"\n")
      print(chisq.test(corr_oth_data[,c1], corr_oth_data[,c2], correct = F))
    }
  }
}

# From Chisq Test we observered that  
# variables "country_reg" and "demographic_slice" 
# have dependencies on varaible "card_offer"
# Whereas variable "ad_exp" is independent 
# of "card_offer"
# Also, apart from "card offer", 
# all variables are independent to each other 
# Based on this insight we can omit "ad_exp" from dataset
# NOTE: this omission process will be done by model itself
# Refer Importance graph and comment at end

####### Data Split ###############
set.seed(112018) # For result consistancy
ind = createDataPartition(train$card_offer, p = 3/4,list = F)
tr = train[ind,]
te = train[-ind,]
prop.table(table(tr$card_offer))*100
prop.table(table(te$card_offer))*100

########## Multiple Model Developments ###########
names(train)

model_control = trainControl(method = 'repeatedcv',
                           number = 10,
                           repeats = 3,
                           classProbs = T,
                           savePredictions = T,
                           summaryFunction = twoClassSummary,
                           allowParallel = T)

seed = 112018
metric = "ROC"
pp = c('center','scale')

str(train)
##### Generalized Linear Model
## Logistic Regression 
set.seed(seed)
fit_logit = train(card_offer ~.,
                      data = tr,
                      method = "glm",
                      family="binomial",
                      metric = metric,
                      trControl = model_control,
                      preProc = pp)

## Naive Bayes
set.seed(seed)
fit_bayes = train(card_offer ~.,
                      data = tr,
                      method = "nb",
                      metric = metric,
                      trControl = model_control,
                      preProc = pp,
                      importance = T)

##### Decision tree Algo
## C5.0
set.seed(seed)
fit_c50 = train(card_offer ~.,
                    data = tr,
                    method = "C5.0",
                    metric = metric,
                    trControl = model_control,
                    preProc = pp,
                    importance = T)

##### Boosting Algo
## Stochastic Gradient Boosting (GBM)
set.seed(seed)
fit_gbm = train(card_offer ~.,
                    data = tr,
                    method = "gbm",
                    metric = metric,
                    trControl = model_control,
                    preProc = pp)

##### Bagging Algo
## Bagged CART
set.seed(seed)
fit_bag = train(card_offer ~.,
                    data = tr,
                    method = "treebag",
                    metric = metric,
                    trControl = model_control,
                    preProc = pp,
                    importance = T)

## Random Forest
set.seed(seed)
fit_rf = train(card_offer ~.,
                    data = tr,
                    method = "rf",
                    metric = metric,
                    trControl = model_control,
                    preProc = pp)

#Summary (ROC - Based)
model_results = resamples(list(C5.0 = fit_c50, 
                               GBM = fit_gbm,
                               RF = fit_rf,
                               TreeBag = fit_bag,
                               NaiveBayes = fit_bayes,
                               Logistic = fit_logit))

summary(model_results)
dotplot(model_results, main = "Model Comparison")

###### Prediction on test sample (Hold-Out Data) #########
{
  pred_rf = predict(fit_rf, te)
  acc_rf = mean(pred_rf == te$card_offer)*100
  
  pred_bag = predict(fit_bag, te)
  acc_bag = mean(pred_bag == te$card_offer)*100
  
  pred_gbm = predict(fit_gbm, te)
  acc_gbm = mean(pred_gbm == te$card_offer)*100
  
  pred_c50 = predict(fit_c50, te)
  acc_c50 = mean(pred_c50 == te$card_offer)*100
  
  pred_logit = predict(fit_logit, te)
  acc_logit = mean(pred_logit == te$card_offer)*100
  
  pred_bayes = predict(fit_bayes, te)
  acc_bayes = mean(pred_bayes == te$card_offer)*100
}

# Accuracy Result on Hold-out Data
acc_res = c(RandomForest = acc_rf, 
  BAGGING = acc_bag, 
  GBM = acc_gbm, 
  C50 = acc_c50,
  logisticReg = acc_logit,
  NaiveBayes = acc_bayes)

pred_res = list(pred_rf, 
                pred_bag,
                pred_gbm,
                pred_c50,
                pred_logit,
                pred_bayes)

##### Confusion Matrix for Each Model ####
for (z in 1:length(pred_res)) {
  cat("\n::::::::::::::::: ",names(acc_res[z])," :::::::::::::::::\n")
  print(confusionMatrix(pred_res[[z]], te$card_offer))
}

# From Confusion Matix, it can be seen that
# C50 is having higest Kappa Value among all
# Following is the distribution for C50
# :::::::::::::::::  C50  :::::::::::::::::
#   Confusion Matrix and Statistics
# 
#             Reference
# Prediction   No  Yes
# No         2100   21
# Yes          17  361
# 
# Accuracy : 0.9848          
# 95% CI : (0.9792, 0.9892)
# No Information Rate : 0.8471          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.941           
# Mcnemar's Test P-Value : 0.6265          
#                                           
#             Sensitivity : 0.9920          
#             Specificity : 0.9450          
#          Pos Pred Value : 0.9901          
#          Neg Pred Value : 0.9550          
#              Prevalence : 0.8471          
#          Detection Rate : 0.8403          
#    Detection Prevalence : 0.8487          
#       Balanced Accuracy : 0.9685          
#                                           
#        'Positive' Class : No              


#### Plot ROC - AUC 
library(pROC)
{
  probs_gbm = predict(fit_gbm, te, type = "prob")
  ROC_gbm = roc(predictor = probs_gbm$Yes,
                response = te$card_offer)
  
  probs_rf = predict(fit_rf, te, type = "prob")
  ROC_rf = roc(predictor = probs_rf$Yes,
               response = te$card_offer)
  
  probs_bag = predict(fit_bag, te, type = "prob")
  ROC_bag = roc(predictor = probs_bag$Yes,
                    response = te$card_offer)
  
  probs_bayes = predict(fit_bayes, te, type = "prob")
  ROC_bayes = roc(predictor = probs_bayes$Yes,
                response = te$card_offer)
  
  probs_c50 = predict(fit_c50, te, type = "prob")
  ROC_c50 = roc(predictor = probs_c50$Yes,
                  response = te$card_offer)
  
  probs_logit = predict(fit_logit, te, type = "prob")
  ROC_logit = roc(predictor = probs_logit$Yes,
                response = te$card_offer)
}

ROC_res = list(ROC_rf = ROC_rf,
            ROC_bag = ROC_bag,
            ROC_gbm = ROC_gbm,
            ROC_c50 = ROC_c50,
            ROC_logit = ROC_logit,
            ROC_bayes = ROC_bayes)

for (x in 1:length(ROC_res)) {
  cat("\n:::::: ",names(acc_res[x])," ::::::::::\n")
  print(ROC_res[[x]])
}

### Plot Graph #####
{
  plot(ROC_c50, type = "l", pch = 5, 
       col="#e6194b", main = "ROC",
       lwd = 1, lty = 2)
  lines(ROC_gbm, col = "#3cb44b", lty = 2)
  lines(ROC_rf, col = "#ffe119", lty = 3)
  lines(ROC_bag, col = "#4363d8", lty = 4)
  lines(ROC_bayes, col = "#f58231", lty = 5)
  lines(ROC_logit, col = "#911eb4", lty = 6)
  legend(0,1, legend = c('C50 AUC', 'GBM AUC', 'RF AUC', 'BAG AUC', 'Bayes AUC', 'Logistic AUC'), 
         col = c('#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4'), lty = 1:6, cex = 0.7)
  text(0.2,0.6, cex = 0.7,labels = "::AUC::")
  text(0.2,0.5, cex = 0.7,labels = paste("C50 : ",round(ROC_c50$auc,5)))
  text(0.2,0.4, cex = 0.7,labels = paste("GBM : ",round(ROC_gbm$auc,5)))
  text(0.2,0.3, cex = 0.7,labels = paste("Rf : ",round(ROC_rf$auc,5)))
  text(0.2,0.2, cex = 0.7,labels = paste("Bag : ",round(ROC_bag$auc,5)))
  text(0.2,0, cex = 0.7,labels = paste("Logistic Regression : ",round(ROC_logit$auc,5)))
  text(0.2,0.1, cex = 0.7,labels = paste("Naive Bayes : ",round(ROC_bayes$auc,5)))
}

#### Important Variables ####
plot(varImp(fit_c50), main = "Important Variables Used For Model Development")
# As you can see "ad_exp" has been opted out from fitted model
# Also proved above using chi-square test 
##################  XXXXXXXXXXXXXXXX  #########################


############ Make Predictions on Test Data using Best Model ###############
test = read.csv("Test 2.csv")
sum(is.na(test))
cust_id = test$customer_id 

test$card_offer = NULL
test$customer_id = NULL

# Using GBM
test_pred = predict(fit_c50, test)
test_pred = ifelse(test_pred == "Yes", "TRUE", "FALSE")

test$customer_id = cust_id 
test$card_offer = test_pred

test = test[,c(ncol(test)-1, ncol(test))]
View(test)
write.csv(test, "Prediction.csv")
