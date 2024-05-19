#Controls set
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)

#Train model (Decision Tree)---
fit.rpart <- train(target~., data = training, method="rpart", metric=metric, trControl=control)
#How well model1 predict
data.pred = predict(fit.rpart, newdata = testing)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(testing$target), mode = "prec_recall")
vi = varImp(fit.rpart, scale = FALSE) #most important features

#Decision Tree plot
plot(fit.rpart$finalModel, uniform=TRUE,
     main="Classification Tree")
text(fit.rpart$finalModel, use.n.=TRUE, all=TRUE, cex=.8)

#Print at this step
log_print("train/test decision tree....")
log_print(cm)
log_print(vi)

#Train model2 (RF)---
fit.rf <- train(target~., data = training, method="rf", metric=metric, trControl=control)
#How well model2 perform
data.pred = predict(fit.rf, newdata = testing)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(testing$target), mode = "prec_recall")
vi = varImp(fit.rf, scale = FALSE)#most important features

#Print at this step
log_print("train/test random forest....")
log_print(cm)
log_print(vi)


#Support Vector Machine---
fit.svm <- train(target~., data = training, method = "svmLinear", trControl=control, metric=metric)
#How well model2 perform
data.pred = predict(fit.rf, newdata = testing)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(testing$target), mode = "prec_recall")
vi = varImp(fit.rf, scale = FALSE)#most important features

#Print at this step
log_print("train/SVM....")
log_print(cm)
log_print(vi)