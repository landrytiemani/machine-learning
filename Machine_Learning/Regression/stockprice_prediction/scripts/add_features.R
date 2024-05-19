#Add features to the Data Set ---
bb = BBands (data[,c(3:5)], n=20, sd=2) #dn is lower band, up is upper band, mavg is moving average, pctB takes moving average and then adds 2 std
bb_low = bb[,1]
bb_avg = bb[,2]
bb_high = bb[,3]

data = data.frame(close = data[,5], bb_low, bb_avg, bb_high)

#Print once at this step
log_print("added features...")
log_print(head(data))
