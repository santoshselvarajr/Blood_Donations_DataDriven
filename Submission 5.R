#DATA PRE-PROCESSING
#Importing Model and Predict Data into R
model_data = read.csv("model_data.csv")
predict_data = read.csv("predict_data.csv")
#Remove Person ID and total volume blood columns
model_data = model_data[-c(1,4)]
predict_data = predict_data[-c(1,4)]
#Encoding the dependent variable as a factor
#model_data$donation = factor(model_data$donation, levels = c(0,1))
#Feature Scaling
# model_data[-4] = scale(model_data[-4])
# predict_data = as.data.frame(scale(predict_data))

#Fit XGBoost
library(xgboost)
classifier = xgboost(objective = "reg:logistic", 
                     data = as.matrix(model_data[-4]),
                     label = model_data$donation,
                     nrounds = 300)

y_pred = predict(classifier, newdata = as.matrix(predict_data))
write.csv(y_pred, "XGBoost.csv", row.names = FALSE)

#Fit a ANN classifier
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = "donation",
                              training_frame = as.h2o(model_data),
                              activation = "Rectifier",
                              hidden = c(3,3),
                              epochs = 100,
                              train_samples_per_iteration = -2)
h2o.shutdown()
prob_pred = h2o.predict(classifier, newdata = as.h2o(predict_data))

prob_pred = as.data.frame(prob_pred)
write.csv(prob_pred,"ANNPred.csv")

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
classifier = glm(formula = donation~., 
                 data = model_data, 
                 family = binomial)

#Print results of resamples
results = resamples(list(lda=fit.lda,cart=fit.cart,knn=fit.knn,svm=fit.svm,rf=fit.rf))
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
predict.logreg = predict(classifier, type = 'response', newdata = predict_data)

PredTable = as.data.frame(cbind(rf = predict.rf,
                                svm = predict.svm,
                                knn = predict.knn,
                                cart = predict.cart,
                                lda = predict.lda,
                                glm = predict.logreg))

write.csv(PredTable, "Rough.csv")
