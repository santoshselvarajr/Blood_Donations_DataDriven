#Importing necessary packages
library(tidyverse)
library(caTools)
library(caret)
library(e1071)
library(usdm)
library(Boruta)

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

#Visualize data
#Scatter Plots for the variables
#ScatPlot = pairs(donation~., data = training_set)
#featurePlot(x = training_set[,1:3], y = training_set[,4], plot = "box")
#featurePlot(x = training_set[,1:3], y = training_set[,4], plot = "density")

#Correlation matrix
#CorrMatrix = cor(training_set[-4])

#Feature Selection (Using Boruta)
#boruta.train = Boruta(donation ~., data = training_set, doTrace = 2)
#print(boruta.train)

#Feature Scaling
training_set[-4] = scale(training_set[-4])
test_set[-4] = scale(test_set[-4])

#Performing PCA Analysis
pca = preProcess(x = training_set[-4], method = "pca", pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2,3,1)]
test_set = predict(pca,test_set)
test_set = test_set[c(2,3,1)]
###########################################################################
###########################################################################
#NAIVE BAYES
# #Fitting Naive Bayes to training set
# classifier = naiveBayes(x = training_set[-4], 
#                         y = training_set$donation)
# 
# #Predict Test Set results
# y_pred = round(predict(classifier, newdata = test_set[-4],"raw"),2)
# 
# #Build the confusion matrix
# cm = table(test_set[,4],y_pred)
# #(cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
# 
# #Applying k-Fold Cross-Validation
# folds = createFolds(training_set$donation, k = 10)
# cv = lapply(folds, function(x){
#   training_fold = training_set[-x,]
#   test_fold = training_set[x,]
#   classifier = naiveBayes(x = training_fold[-4], 
#                           y = training_fold$donation)
#   y_pred = predict(classifier, newdata = test_fold[-4])
#   cm = table(test_fold[,4],y_pred)
#   accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
#   return(accuracy)
# })
# Avg_accuracy = mean(as.numeric(cv))
# 
# #Applying Grid Search for optimum parameters
# classifier = train(form = donation ~.,
#                    data = training_set, 
#                    method = "naive_bayes")

###########################################################################
###########################################################################

#LOGISTIC REGRESSION
#Fitting Logistic Regression to training set
classifier = glm(formula = donation~., 
                 data = training_set, 
                 family = binomial)

#Predict Test Set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-4])
y_pred = ifelse(prob_pred>0.5, 1, 0)

#Build the confusion matrix
cm = confusionMatrix(test_set[,4],as.factor(y_pred))
cm
#(cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])

#Applying k-Fold Cross-Validation
folds = createFolds(training_set$donation, k = 10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x,]
  test_fold = training_set[x,]
  classifier = glm(formula = donation~., 
                   data = training_fold, 
                   family = binomial)
  prob_pred = predict(classifier, 
                      type = 'response', 
                      newdata = test_fold[-4])
  y_pred = ifelse(prob_pred>0.5, 1, 0)
  cm = table(test_fold[,4],y_pred)
  accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
  return(accuracy)
})
Avg_accuracy = mean(as.numeric(cv))


#Applying Grid Search for optimum parameters
classifier = train(form = donation ~.,
                   data = training_set, 
                   method = "glm")

######################################################################################
######################################################################################
#10 fold cross validation
control = trainControl(method = "cv", number = 10)
metric = "Accuracy"

#Linear Discriminant Analysis
set.seed(12)
fit.lda = train(donation~., data = model_data, method = "lda",
                metric = metric, trControl = control)
#CART
set.seed(12)
fit.cart = train(donation~., data = model_data, method = "rpart",
                 metric = metric, trControl = control)
#KNN
set.seed(12)
fit.knn = train(donation~., data = model_data, method = "knn",
                metric = metric, trControl = control)

#SVM
set.seed(12)
fit.svm = train(donation~., data = model_data, method = "svmRadial",
                metric = metric, trControl = control)

#Random Forest
set.seed(12)
fit.rf = train(donation~., data = model_data, method = "rf",
               metric = metric, trControl = control)

#logistic Regression
set.seed(12)
fit.glm = train(donation~., data = model_data, method = "glm",
                metric = metric, trControl = control)

#Print results of resamples
results = resamples(list(lda=fit.lda,cart=fit.cart,knn=fit.knn,svm=fit.svm,rf=fit.rf,glm=fit.glm))
summary(results)
dotplot(results)

#Prediction: LDA
predict.lda = predict(fit.lda, predict_data)

#Prediction: CART
predict.cart = predict(fit.cart, predict_data)

#Prediction: KNN
predict.knn = predict(fit.knn, predict_data)

#Prediction: SVM
predict.svm = predict(fit.svm, predict_data)

#Prediction: RF
predict.rf = predict(fit.rf, predict_data)

#Prediction: GLM
# predict.glm = predict(fit.glm, predict_data)

PredTable = as.data.frame(cbind(rf = predict.rf,
                                svm = predict.svm,
                                knn = predict.knn,
                                cart = predict.cart,
                                lda = predict.lda))