#Correlation
cor = cor(data[,c(2:5)]) #Can't correlate date which is the 1 vriable
corrplot(cor)

#Print text when this step is running
log_print("correlation analysis...")
log_print(cor)
