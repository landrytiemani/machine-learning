#Import data
start = as.Date(Sys.Date()-(365*5)) #start date
end=as.Date(Sys.Date())#current date
getSymbols("AMZ", src="yahoo", from=start, to = end) #Feed symbol from s&p500
data = GOOG #Call xts object into env.
colnames(data) = c("Open", "High", "Low", "Close", "Volume", "Adjusted")
#Save data----
write.csv(data, "data/original.csv", row.names = FALSE)

#Print when I download my data
log_print("Download data... ")
log_print(head(data))