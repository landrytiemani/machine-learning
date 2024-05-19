#Regression ML to predict close price of Google

#Linear regression in R using caret
library(caret)

#Load data into R
library(quantmod) #Librry for API access
start <- as.Date(Sys.Date()-(365*5)) #5 years of data
end <- as.Date(Sys.Date())
getSymbols("GOOG", src = "yahoo", from = start, to = end) #Connect to yahoo API for stocks data imports
data = GOOG
colnames(data) = c("Open", "High", "Low", "Close", "Volume", "Adjusted") #Variables needed

#Visualize data (EDA)
plot(data[, "Close"], main = "Close Price") #Close Price...

#Create Test Harnesses
control <- trainControl(method="cv", number=10)
metric <- "Rsquared"  #Used for regression. Also RMSE can be used. R2 fits well and RMSE will obtained anyway


#Split the Data
split<-createDataPartition(y = data$Close, p = 0.7, list = FALSE) #70/30 split since I have a good number of data
train<-data[split,]
valid<-data[-split,]


#Build a Linear Regression Model using lm
set.seed(7)
fit.LM <- train(Close~., data = train, method = "lm", trControl=control, metric=metric)

#Summarize the Results Briefly
fit.LM #Rsquared =1. So it is overfited, RMSE=7.054381e-14 (error is very low. So my model trained and overfited which makes sense since Open high, low close and there is a multicolinarity among those variables.
#There is basically the same trend, same info. So there's really nothing that is fed into the model that gives learning capacity. This is why like our sample, I need to create a new
#variables and remove most if the variables because of high correlation btw and inputs and output.) 

#Summarize the Entire Final Model
summary(fit.LM$finalModel)


#Create Prediction using Trained Regression (here we don't use confusion matrix to check how our model predict, but we use post resample function)
predictedValues<-predict(fit.LM, valid)
modelvalues<-data.frame(obs = valid$Close, pred=predictedValues)
postResample(pred = predictedValues, obs = valid$Close) #This give us R2, RMSE, MAE
#In our example, the error is so low cause by overfitting our dataset (multicorrelarity).

