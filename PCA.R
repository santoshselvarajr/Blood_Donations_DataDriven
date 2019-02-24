#Importing necessary packages
library(tidyverse)
library(caTools)
library(caret)
library(e1071)
library(kernlab)
library(ElemStatLearn)

#Performing PCA
#DATA PRE-PROCESSING
#Importing Model and Predict Data into R
model_data = read.csv("model_data.csv")
predict_data = read.csv("predict_data.csv")
#Remove Person ID and total volume blood columns
model_data = model_data[-c(1,4)]
predict_data = predict_data[-c(1,4)]
#Encoding the dependent variable as a factor
model_data$donation = factor(model_data$donation, levels = c(0,1))

#Splitting Model Data into training and test data

split = sample.split(model_data$donation, SplitRatio = 0.75)
training_set = subset(model_data, split == TRUE)
test_set = subset(model_data, split == FALSE)

#Feature Scaling
training_set[-4] = scale(training_set[-4])
test_set[-4] = scale(test_set[-4])

# #Performing Kernel PCA Analysis
# pca = preProcess(training_set[-4], method = "pca", pcaComp = 2)
# training_set = predict(pca, training_set)
# training_set = training_set[c(2,3,1)]
# test_set = predict(pca, test_set)
# test_set = test_set[c(2,3,1)]

#LOGISTIC REGRESSION
#Fitting Logistic Regression to training set
# classifier = glm(formula = donation~., 
#                  data = training_set, 
#                  family = binomial)
# 
# #Predict Test Set results
# prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
# y_pred = as.factor(ifelse(prob_pred>0.5, 1, 0))
# 
# #Build the confusion matrix
# cm = table(test_set[,3],y_pred)
# cm
#(cm[1,1])/(cm[1,1]+cm[2,1])

#10 fold cross validation
control = trainControl(method = "cv", number = 10)
metric = "Accuracy"

#Linear Discriminant Analysis
set.seed(123)
fit.lda = train(donation~., data = training_set, method = "lda",
                metric = metric, trControl = control)
#CART
set.seed(123)
fit.cart = train(donation~., data = training_set, method = "rpart",
                metric = metric, trControl = control)
#KNN
set.seed(123)
fit.knn = train(donation~., data = training_set, method = "knn",
                metric = metric, trControl = control)

#SVM
set.seed(123)
fit.svm = train(donation~., data = training_set, method = "svmRadial",
                metric = metric, trControl = control)

#Random Forest
set.seed(123)
fit.rf = train(donation~., data = training_set, method = "rf",
                metric = metric, trControl = control)

#logistic Regression
set.seed(123)
fit.glm = train(donation~., data = training_set, method = "glm",
                metric = metric, trControl = control)

#Print results of resamples
results = resamples(list(lda=fit.lda,cart=fit.cart,knn=fit.knn,svm=fit.svm,rf=fit.rf,glm=fit.glm))
summary(results)
dotplot(results)

#Prediction: LDA
predict.lda = predict(fit.lda, test_set)
confusionMatrix(predict.lda, test_set$donation)

#Prediction: CART
predict.cart = predict(fit.cart, test_set)
confusionMatrix(predict.cart, test_set$donation)

#Prediction: KNN
predict.knn = predict(fit.knn, test_set)
confusionMatrix(predict.knn, test_set$donation)

#Prediction: SVM
predict.svm = predict(fit.svm, test_set)
confusionMatrix(predict.svm, test_set$donation)

#Prediction: RF
predict.rf = predict(fit.rf, test_set)
confusionMatrix(predict.rf, test_set$donation)

#Prediction: GLM
predict.glm = predict(fit.glm, test_set)
confusionMatrix(predict.glm, test_set$donation)

PredTable = as.data.frame(cbind(donation = test_set[,4], 
                          glm = predict.glm,
                          rf = predict.rf,
                          svm = predict.svm,
                          knn = predict.knn,
                          cart = predict.cart,
                          lda = predict.lda))
