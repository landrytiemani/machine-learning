#Logistic regression handle the categorical responses
#Logistic Regression in R using caret
library(caret)

#Load data into R
data("iris")
iris = iris
str(iris)

#Visualize data (EDA)
with(iris, qplot(iris[,1], iris[,2], colour=iris[,ncol(iris)], cex=0.2)) #Setosa seems to be more predictable than the 2 others

#Split the Data (70/30 split)
index = createDataPartition(iris[,1], p =0.70, list = FALSE)
dim(index)

#Training data
training = iris[index,]
dim(training)

#Validation data
valid = iris[-index,]
dim(valid)

#Create Test Harnesses
control <- trainControl(method="cv", number=10)
metric <- "Accuracy" #Classification problem

#Build a Logistic Regression Model using multinom (we have 3 factors. So it is not a binary outcome but a multinominal)
set.seed(7)
#fit.LR <- train(Species~., data = training, method = "multinom", family=binomial(), trControl=control, metric=metric)
#In a multinominial problem, we shouldn't include family=binomial. It can cause a factor exclusion.
fit.LR <- train(Species~., data = training, method = "multinom", trControl=control, metric=metric)

#Summarize the Results Briefly
fit.LR
#decay rate tries to do right for the gradient descent and see if it can improve the result. In this case 1e-01  gives a great accuracy of 0.9900000.


#Summarize the Entire Final Model
summary(fit.LR$finalModel)


#Plot the Model
plot(fit.LR) #Check if increasing decay rate, increase/decrease accuracy

#Create Prediciton using Trained Regression 
data.pred = prediintervalcmct(fit.LR, newdata = valid)
table(data.pred, valid$Species) #Confusion matrix

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)
#No mistake on setosa and virginica. 2 errors on versicolor. 
#Accuracy equals 0.95. Confidence  interval (0.8419, 0.9943). It has good predictive capability


##Build a Logistic Regression Model using LogitBoost

#Import Library
library(caTools)

#Model using LogitBoost
set.seed(7)
fit.LogitBoost <- train(Species~., data = training, method="LogitBoost", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.LogitBoost
#It boost the number of iteraion to increase accuracy

#Summarize the Entire Final Model
summary(fit.LogitBoost$finalModel)

#Plot the Model
plot(fit.LogitBoost) #31 iterations got the best result of 0.978 accuracy

#Create Prediciton using Trained Regression
data.pred = predict(fit.LogitBoost, newdata = valid)
table(data.pred, valid$Species) #confusion matrix

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)
#Accuracy of 0.9535


##Build a Logistic Regression Model using plr
#Import library
library(stepPlr)

#Regression Model using plr
set.seed(7)
fit.plr <- train(Species~., data = training, method="plr", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.plr #Different penalization metrics are run to find the best lambda

#Summarize the Entire Final Model
summary(fit.plr$finalModel)

#Plot the Model
plot(fit.plr) #Nothing improve since dataset is so small that the model didn't have time to effectively do the penalty.

#Create Prediciton using Trained Regression
data.pred = predict(fit.plr, newdata = valid)
table(data.pred, valid$Species) #The model performed worse bc it couldn't learn anything with this penalization technique. Also Virginica had a complete wrong. 

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm) #0.72 accuracy with CI of (0.5633, 0.8467)

#Compare Logistic Regression Models
results = resamples(list(LogR = fit.LR, LogitBoost=fit.LogitBoost, PLR = fit.plr))
summary(results)

#Visualize Comparison
dotplot(results) #LogR performs the best (Accuracy is high and CI is smaller)


