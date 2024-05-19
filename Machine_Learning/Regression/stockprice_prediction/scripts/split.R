index = createDataPartition(data[,1], p =0.80, list = FALSE)
training = data[index,]
testing = data[-index,]

#Print text at this step
log_print("split data....")
log_print(head(training))
log_print(head(testing))