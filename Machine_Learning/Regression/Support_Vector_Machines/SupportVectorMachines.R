#SVM Linear in R using caret
#Use caret to implement SVM using a different kernels in R to fit regression type problems (Google close price)
library(caret)

#Load data into R enviorment
library(quantmod)
start <- as.Date(Sys.Date()-(365*5))
end <- as.Date(Sys.Date())
getSymbols("GOOG", src = "yahoo", from = start, to = end)

data = GOOG
colnames(data) = c("Open", "High", "Low", "Close", "Volume", "Adjusted")

#Visualize the Data 
plot(data[, "Close"], main = "Close Price") #Close Price is linear

#Create Test Harnesses
control <- trainControl(method="cv", number=10) #10 folds and R2 since regression problem
metric <- "Rsquared"

#Split the Data
split<-createDataPartition(y = data$Close, p = 0.7, list = FALSE)
train<-data[split,]
valid<-data[-split,]

#Build a SVM Model for Regression problem using svmLinear
set.seed(7)
fit.LM <- train(Close~., data = train, method = "svmLinear", trControl=control, metric=metric)

#Summarize the Results Briefly
fit.LM
#R2 is good. However RMSE is not ($39.89 off of the real close price which is a big error rate). Linear regression is better than SVM for regression problems

#Summarize the Entire Final Model
summary(fit.LM$finalModel)

#Create Prediction using Trained SVM #Post resample technique for regression problems
predictedValues<-predict(fit.LM, valid)
modelvalues<-data.frame(obs = valid$Close, pred=predictedValues)
postResample(pred = predictedValues, obs = valid$Close) #We got a good R2 (0.99959). RMSE and MAE are very off. It is not a good model to predict actuals.



##SVM Non-Linear in R using caret

#Use caret to implement SVM using different kernels in R to fit classification problems.
library(caret)

#Load data into R enviroment
data("iris")
iris = iris
str(iris)

#Visualization (EDA)

#Split the Data (70/30 Split)
index = createDataPartition(iris[,1], p =0.70, list = FALSE)
dim(index)

#Training Data
training = iris[index,]
dim(training)

#Validation Data
valid = iris[-index,]
dim(valid)

#Create Test Harnesses
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Build a SVM Model for Classification using svmLinear
set.seed(7)
#fit.LR <- train(Species~., data = training, method = "svmLinear", family=binomial(), trControl=control, metric=metric)
fit.LR <- train(Species~., data = training, method = "svmLinear", trControl=control, metric=metric) #family=binomial() used for binomial problems

#Summarize the Results Briefly
fit.LR #Good accuracy 0.9809091

#Summarize the Entire Final Model
summary(fit.LR$finalModel)

#Create Prediciton using Trained SVM
data.pred = predict(fit.LR, newdata = valid)
table(data.pred, valid$Species)

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm) #good 0.9302 with CI (0.8094, 0.9854)). Setosa is handled easily. Few errors done on versicolor and virginica


##Build a SVM Model for Classification using svmRadial
set.seed(7)
fit.svmRadial <- train(Species~., data = training, method="svmRadial", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.svmRadial #Doing penalization matrix. 0.5 is the best with accuracy of 0.9727273

#Summarize the Entire Final Model
summary(fit.svmRadial$finalModel)

#Create Prediciton using Trained Regression
data.pred = predict(fit.svmRadial, newdata = valid)
table(data.pred, valid$Species)

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm) # Accuracy : 0.907  &  95% CI : (0.7786, 0.9741)

##Build a SVM Model for Classification using svmPoly
set.seed(7)
fit.svmPoly <- train(Species~., data = training, method="svmPoly", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.svmPoly

#Summarize the Entire Final Model
summary(fit.svmPoly$finalModel)

#Create Prediciton using Trained Regression
data.pred = predict(fit.svmPoly, newdata = valid)
table(data.pred, valid$Species)

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm) #Accuracy : 0.9302  & 95% CI : (0.8094, 0.9854)

#Compare SVMs Models
results = resamples(list(svmLinear = fit.LR, svmRadial=fit.svmRadial, svmPoly = fit.svmPoly))
summary(results)


#Visualize Comparison
dotplot(results) #Poly trained the best. And handled versicolor and virginica classification the best.



