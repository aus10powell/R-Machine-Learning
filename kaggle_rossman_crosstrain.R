
library(readr)
set.seed(123)
library(xgboost)

#my favorite seed^^



setwd("~/Dropbox/Kaggle/Rossman")
cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")
store <- read_csv("store.csv")



# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

# Begin Comment weather code
if(FALSE)  {  
	
	 
# Weather data
# may include at a later date
df = read.table("germany_weather2.txt",sep = ",",header=TRUE) 
# df[is.na(df)] <- 0 # don' think this is necessary with gbm

df$MXSPD[df$MXSPD==999.90] = median(df$MXSPD)
df$MXSPD = log(df$MXSPD) # is an improvement

df$VISIB[df$VISIB==999.90] = median(df$VISIB)
df$VISIB = log(df$VISIB) # is an improvement

# replace missing obs with media and take log
df$VISIB[df$VISIB == 999.900] <- median(df$VISIB)
df$VISIB = log(df$VISIB)



df$Date = format(df$YEARMODA)
df$Date = as.Date(df$Date,"%Y%m%d")
df.mxspd = aggregate(df$MXSPD,list(date=df$Date),median)
df.temperature = aggregate(df$TEMP, list(date = df$Date), median)
df.visib= aggregate(df$VISIB, list(date = df$Date), median)
names(df.temperature) = c("Date","median.temp")
names(df.mxspd) = c("Date","median.mxspd")
names(df.visib) = c("Date","median.visib")


# merge mean temp with train and test sets
train = merge(x=train,y=df.temperature,by="Date")
test = merge(x=test,y=df.temperature,by="Date")
# merge median max wind speed with train and test sets
train = merge(x=train,y=df.mxspd,by="Date")
test = merge(x=test,y=df.mxspd,by="Date")

train = merge(x=train,y=df.mxspd,by="Date")
test = merge(x=test,y=df.mxspd,by="Date")

# End comment out weather code
}



cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
#train <- train[ which(train$Open=='1'),]
#train <- train[ which(train$Sales!='0'),]
# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train = train[ , -which(names(train) %in% c("Date","StateHoliday"))] # train = train[,-c(3,8)]

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test = test[ , -which(names(test) %in% c("Date","StateHoliday"))] # previously was: test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,5:19)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)
h<-sample(nrow(train),10) # changed from 10,000

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.01, # changed from an optimum 0.02. 0.06, #0.01,
                max_depth           = 12, # changed from optimum 12, changed from default of 8
                subsample           = 0.90, # changed from optimum 0.9, 0.7
                colsample_bytree    = 0.7, # 0.7
                #num_parallel_tree   = 1000,
                alpha = 0.0002, # default alpha = 0.0001, 
                 #lambda = .9
                # gamma = .9
)


timer = proc.time() # start the timer


history <- xgb.cv(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 3000, # optimum 4000 #300, #280, #125, #250, # changed from 300
                    verbose             = 1, #changed from 0
                   early.stop.round    = 100, #changed from optimum 100
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE,
                    nfold=3,
                    nround=2,
                    prediction = TRUE # If TRUE data table results (history$dt) and predictions are provided
)
timer = proc.time() - timer  # end the timer
print(history$dt)

# allow results to be output to a text file with changes to information
out <- capture.output(history$dt)
cat(timer, 
    file="output_summary/summary_of_kaggle_rossman_ben_hammer_temperature_crosstrain.txt", sep="\n", append=TRUE)
cat("Note: no weather information   work13
          -using settings from working2
          -did not eliminate samples from training closed for repairs and whose sales were 0
         
                eta                 = 0.01, # changed from an optimum 0.02. 0.06, #0.01,
                max_depth           = 12, # changed from optimum 12, changed from default of 8
                subsample           = 0.90, # changed from optimum 0.9, 0.7
                colsample_bytree    = 0.7, # 0.7
                #num_parallel_tree   = 1000,
                alpha = 0.0002, # default alpha = 0.0001, 
                 #lambda = .9
                # gamma = .9
    
    nfold=3,
    nround=2 
    ", 
    out, 
    file="output_summary/summary_of_kaggle_rossman_ben_hammer_temperature_crosstrain.txt", sep="\n", append=TRUE)


clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 3000, # optimum 4000 #300, #280, #125, #250, # changed from 300
                    verbose             = 1, #changed from 0
                   early.stop.round    = 100, #changed from optimum 100
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1

# save model to R's raw vector


submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, "rf_17.csv")



library(caret)
library(xgboost)
library(readr)
library(dplyr)
library(tidyr)
history$dt %>%
  select(-contains("std")) %>%
  mutate(IterationNum = 1:n()) %>%
  gather(TestOrTrain, AUC, -IterationNum) %>%
  ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
  geom_line() + 
  theme_bw()