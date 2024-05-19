processed = read.csv("data/processed.csv")
data = processed[,c(2:5)]
rule = diff(data$close) #Diff btw previous and current close
target = ifelse(rule>0, 1, 2) #1 means buy, 2 is sell

data = data.frame(data[2:1260,c(2:4)],target) #remove 1st row since diff eliminate the 1st row of targets
data$target=factor(data$target, levels =c(1,2), labels=c("buy", "sell"))

#Print text at this step
log_print("target creation....")
log_print(head(data))
