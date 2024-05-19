#Importing library
library(caret)

#Load data into R environment
data("iris")
iris = iris
str(iris)

#Visualize the data ~ Scatterplot Matrix (EDA)
library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
featurePlot(x = iris[, 1:4], #separate input (x)
            y = iris$Species, #separate output (y)
            plot = "pairs", #correlation matrix
            ## Add a key at the top (number of output factors)
            auto.key = list(columns = 3))
#Based on the matrix, setosa is easier to separate out of the others 2 species whatever input is taken. 
#In Contrary to versicolor and Virginica are overlapping. Hope RF can improve our overall decision model

#Visualize the data ~ Overlayed Density Plots
transparentTheme(trans = .9)
featurePlot(x = iris[, 1:4], 
            y = iris$Species,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 3))

#Visualize the data ~ Box Plots
featurePlot(x = iris[, 1:4], 
            y = iris$Species, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,1 ), 
            auto.key = list(columns = 2))

#Split the Data (Create a list of 80% of the rows in the Original dataset we can use for training)
index = createDataPartition(iris[,1], p =0.80, list = FALSE)
dim(index)

#Training Data (80% of data)
training = iris[index,]
dim(training)

#Validation Data
valid = iris[-index,]
dim(valid)

#Create Test Harnesses (set controls fir training and testing. "accurracy" for classification or "root mean squared error" or "R-squared" for regression)
control <- trainControl(method="cv", number=10) #This create 10 randomly split folds to we are randomizing anyway even if we accidentally created manual split in the beginning (train/test split). This is to get average accuracy accross those 10 splits
metric <- "Accuracy"

#Build a Random Forest Model using rf
set.seed(7)
fit.rf <- train(Species~., data = training, method="rf", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.rf

#Summarize the Entire Final Model
summary(fit.rf$finalModel)
#Bunch of metadata among them "importance" refers into feature engineering or feature selection bc we can use RF to select variables to them, train  better RF or any other model

#Plot the Models Performance
plot(fit.rf) #Performance is closed to 1

#Plot the Variable Importance
vi = varImp(fit.rf, scale = FALSE)
plot(vi, top = ncol(training)-1)
#This shows the Petal.Length and Petal.width are the most significant variable in the training process.
#From here, we can try another RF and see if it improves our result

#Create Prediciton using Trained Random Forest
data.pred = predict(fit.rf, newdata = valid)
table(data.pred, valid$Species)
#No error in this case so RF improve the accuracy comparing to DecisionTrees

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate
#0 error rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)
#accuracy is 1. Setosa, Versicolor & Virginica, so RF improve the accuracy comparing to DecisionTrees

#Build a Random Forest Model using extraTrees (New model)
set.seed(7)
fit.extraTrees <- train(Species~., data = training, method="extraTrees", metric=metric, trControl=control)

#Summarize the Results Briefly
fit.extraTrees

#Summarize the Entire Final Model
summary(fit.extraTrees$finalModel)

#Plot the Model
plot(fit.extraTrees)


#Create Prediciton using Trained Random Forest
data.pred = predict(fit.extraTrees, newdata = valid)
table(data.pred, valid$Species)

#Check Error
error.rate = round(mean(data.pred != valid$Species,2))
error.rate

#Confusion Matrix
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)

#Compare Random Forest Models
results = resamples(list(rf=fit.rf, extraTrees = fit.extraTrees))
summary(results)

#Visualize Comparison
dotplot(results)













