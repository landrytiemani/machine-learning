#Build a system that tell when to buy or sell google stock

#setwd---
setwd("C:/stockprice_prediction")

#Load Libs ---
source("scripts/libs.r")
#Open Log---
fn = paste0("log_",toString(Sys.Date()),"_",toString(Sys.time()))
fn = gsub(":", "_",fn)
log_open(file_name = fn)

#Import data---
source("scripts/import_data.r") #allow to run other scripts

#Explore the data --- 
source("scripts/eda.r") #Correlation analysis

#Create variables in data---
source("scripts/add_features.r")

#Preprocess Data---
source("scripts/preprocess_data.r")

#Create Target variable---
source("scripts/create_target.r")

#Splits---
source("scripts/split.r")

#Modeling---
for (i in 1:1){
  source("scripts/modeling.r")
}


#Close Log---
log_close()
