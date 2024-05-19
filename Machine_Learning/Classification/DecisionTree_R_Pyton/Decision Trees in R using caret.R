#Decision Trees in R using caret
library(caret)

#Loaad data into R environement 
data(iris)
iris=iris
str(iris)

#Visualize the data to check if any insight
with(iris, qplot(iris[,1], iris[,2], color =iris[,ncol(iris)], cex=0.2) ) #The graph shows that setosa will be easy to identify using this features (sepal length vs sepal width). However, versicolor and virginica are overlapping and the model will have more difficulty to handle this.

#Split the data: Create a list of 80% of rows in Original dataset we can use for training. So 20% for testing
index = createDataPartition(iris[,1], p=0.8, list=FALSE)
dim(index)


#Training Data
#Use 80% of data to train the model
training = iris[index,]
dim(training)

#Validation Data (dim of 20% of my data)
valid =iris[-index,]
dim(valid)

#Create Test Harnesses (create a learning algo control)
#In this case we will use a robust method called cross-validation. It will create 10 folds from the traininf data randomly selected and train the model 10 times with those folds. SO we can get an average accuracy on the training. Since it is a classification problem we neded metric=accuracy
control = trainControl(method="cv", number=10)
metric = "Accuracy"

#Build a Decision Tree Model using rpart
set.seed(7)
fit.rpart = train(Species~., data=training, method="rpart", metric=metric, trControl=control)


#Summarize the Results Briefly
fit.rpart #cp is smaller meaning model is optimum with accuracy equals to 0.96

#Summarize the Entire Final Model
summary(fit.rpart$finalModel)

#Plot the Model
plot(fit.rpart$finalModel, uniform = TRUE,
     main = "Classification Tree")
text(fit.rpart$finalModel, use.n.=TRUE, all = TRUE, cex=.8)

#Fancier Plot
suppressMessages(library(rattle))
fancyRpartPlot(fit.rpart$finalModel) # Setosa is a perfect node while we got some errors on others nodes

#So far we have just look into the training model w/o doing anything. Now we have to predict 

#Create Prediction using Trained Decision Tree
data.pred = predict(fit.rpart, newdata = valid)
table(data.pred, valid$Species) #The pred table showed no error for setosa, 2 errors for versicolor and 1 error for virginica

#Check Error
error.rate = round(mean(data.pred != valid$Species, 2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)
#Accuracy = 0.8966. So it would be a good model to predict species using these inputs.


##Build a Decision Tree using rpart1SE
set.seed(7)
fit.rpart1SE = train(Species~., data=training, method="rpart1SE", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.rpart1SE #cp is smaller meaning model is optimum with accuracy equals to 0.96

#Summarize the Entire Final Model
summary(fit.rpart1SE$finalModel)

#Plot the Model
plot(fit.rpart1SE$finalModel, uniform = TRUE,
     main = "Classification Tree")
text(fit.rpart1SE$finalModel, use.n.=TRUE, all = TRUE, cex=.8)

#Fancier Plot
suppressMessages(library(rattle))
fancyRpartPlot(fit.rpart1SE$finalModel) # Setosa is a perfect node while we got some errors on others nodes

#So far we have just look into the training model w/o doing anything. Now we have to predict 

#Create Prediction using Trained Decision Tree
data.pred = predict(fit.rpart1SE, newdata = valid)
table(data.pred, valid$Species) #The pred table showed no error for setosa, 2 errors for versicolor and 1 error for virginica

#Check Error
error.rate = round(mean(data.pred != valid$Species, 2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)
#Accuracy = 0.8966. So this model rpart1SE works as same as the previous model (rpart)

##Build a Decision Tree using rpart2
set.seed(7)
fit.rpart2 = train(Species~., data=training, method="rpart2", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.rpart2 #cp is smaller meaning model is optimum with accuracy equals to 0.96

#Summarize the Entire Final Model
summary(fit.rpart2$finalModel)

#Plot the Model
plot(fit.rpart2$finalModel, uniform = TRUE,
     main = "Classification Tree")
text(fit.rpart2$finalModel, use.n.=TRUE, all = TRUE, cex=.8)

#Fancier Plot
suppressMessages(library(rattle))
fancyRpartPlot(fit.rpart2$finalModel) # Setosa is a perfect node while we got some errors on others nodes

#So far we have just look into the training model w/o doing anything. Now we have to predict 

#Create Prediction using Trained Decision Tree
data.pred = predict(fit.rpart2, newdata = valid)
table(data.pred, valid$Species) #The pred table showed no error for setosa, 2 errors for versicolor and 1 error for virginica

#Check Error
error.rate = round(mean(data.pred != valid$Species, 2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)
#Accuracy = 0.8966. So this model rpart2 works as same as the previous model (rpart1SE)


#Compare Decision Tree Models
results = resamples(list(rpart = fit.rpart, rpart1SE = fit.rpart1SE, rpart2 = fit.rpart2))
summary(results)

#Visualize comparison
dotplot(results) #All 3 models perform identically
