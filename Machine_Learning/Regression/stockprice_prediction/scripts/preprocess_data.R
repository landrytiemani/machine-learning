#Preprocess Data
# - [] - Impute Missing NA Values ----
missing = data %>% mice::mice(m=5,maxit=50,meth="sample",seed=500,print = FALSE)
missing <- mice::complete(missing, action=as.numeric(2))
data = na.omit(missing)
print(str(data))
# - [] - Impute Outliers ----
data = data %>% outlieR::impute(flag = NULL, fill = "mean", 
                                level = 0.1, nmax = NULL,
                                side = NULL, crit = "lof", 
                                k = 5, metric = "euclidean", q = 3)
print(str(data))
# - [] - Normalize the Data ----
preProClean <- preProcess(x = data, method = c("scale", "center"))
data <- predict(preProClean, data %>% na.omit)
print(str(data))
# - [] - Save the Processed Data ----
write.csv(data, "data/processed.csv")

#Print text at this step
log_print("preprocess data....")
log_print(head(data))
