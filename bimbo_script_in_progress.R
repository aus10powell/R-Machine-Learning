
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(caret)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
setwd("~/Google Drive/kaggle_bimbo")


# Any results you write to the current directory are saved as output.
#Load data.table for fast reads and aggreagtions
library(data.table)

# Input data files are available in the "../input/" directory.

# Read in only required columns and force to numeric to ensure that subsequent 
# aggregation when calculating medians works
train <- fread('input/train.csv', 
               select = c('Semana', 'Agencia_ID', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil'),
               colClasses=c(Semana="numeric", Agencia_ID="numeric", Cliente_ID="numeric",Producto_ID="numeric",Demanda_uni_equil="numeric"))

train <- fread('input/train.csv', colClasses = "numeric")
test <- fread('input/test.csv', colClasses = "numeric")

#transform target variable to log(1 + demand) - this makes sense since we're 
#trying to minimize rmsle and the mean minimizes rmse:
train$log_demand = log1p(train$Demanda_uni_equil) 


###################### Start the feature modeling ########################
# set a table key to enable fast aggregations
#set a table key to enable fast aggregations
setkey(train, Producto_ID, Cliente_ID, Ruta_SAK)
setkey(test, Producto_ID, Cliente_ID, Ruta_SAK)

cat("Computing means")
mean_total <- mean(train$log_demand) #overall mean
mean_Prod <-  train[, .(mean_prod = mean(log_demand)), by = .(Producto_ID)]
mean_Prod_Ruta <- train[, .(mean_prod_ruta = mean(log_demand)),
                        by = .(Producto_ID, Ruta_SAK)] #mean by product and ruta
mean_Client_Prod_Agencia <- train[, .(mean_client_prod = mean(log_demand)),
                                  by = .(Producto_ID, Cliente_ID, Agencia_ID)] #mean by product, client, agencia




#  set up custom evaluation metric
RMSLE <- function(preds, cv_dtrain) {
  labels <- getinfo(cv_dtrain, "label")
  
  # elab<-exp(as.numeric(labels))-1
  elab <- log(as.numeric(labels) + 1)
  # epreds<-exp(as.numeric(preds))-1
  epreds <- log(as.numeric(preds) +1 )
  # err <- sqrt(mean((epreds/elab-1)^2))
  err <- sqrt(mean(( epreds -   elab  )^2))
  return(list(metric = "RMSLE", value = err))
}


# last chance set of parameters based off of grid
param <- list(objective           = "reg:linear", 
              booster = "gbtree",
              eta                 = 0.03, # changed from an optimum 0.01
              max_depth           = 10, # changed from optimum 12, changed from default of 8
              gamma = 0.2,
              colsample_bytree    = 0.8, # 0.7
              #scale_pos_weight = 87,
              min_child_weight = 1
)





###################### Begin training XGBOOOOOOOOOOST!!!
# set the training rounds for both cv and testing
nrounds_cv = 3000
timer = proc.time() # start the timer

history <- xgb.cv(   params              = param, 
                     data                = cv_dtrain, 
                     nrounds             = nrounds_cv, # optimum 4000 #300, #280, #125, #250, # changed from 300
                     verbose             = 1, #changed from 0
                     # early.stop.round    = 100, #changed from optimum 100
                     watchlist           = watchlist,
                     maximize            = FALSE,
                     eval.metric         = "auc",
                     feval=RMSLE,
                     nfold=2,
                     prediction = TRUE # If TRUE data table results (history$dt) and predictions are provided
)
timer = proc.time() - timer  # end the timer
print(history$dt)

############## Plot the training/testing history
plot(history$dt$train.auc.mean,col='darkred',type='l',lwd=2)
lines(history$dt$train.auc.mean+ history$dt$train.auc.std,type="l",lty=2,col="red")
lines(history$dt$train.auc.mean- history$dt$train.auc.std,type="l",lty=2,col="red")

lines(history$dt$test.auc.mean,col='darkblue',type='l',lwd=2)
lines(history$dt$test.auc.mean+ history$dt$test.auc.std,type="l",lty=2,col="blue")
lines(history$dt$test.auc.mean- history$dt$test.auc.std,type="l",lty=2,col="blue")
legend(20,.9,c("mean train","mean test"),col=c("red","blue"),lty=c(1,1),cex=.9)
# 




################################################
cv_train = train[, !("Demanda_uni_equil"), with=FALSE]

#Inspired by:
#R_medians https://www.kaggle.com/nigelcarpenter/grupo-bimbo-inventory-demand/r-medians/run/264606
#R mean-medians:https://www.kaggle.com/paulorzp/grupo-bimbo-inventory-demand/mean-median-lb-0-48

setwd("~/Google Drive/kaggle_bimbo")
library(data.table)
library(dplyr)

train <- fread('input/train.csv', 
               select = c('Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Demanda_uni_equil'))
test <- fread('input/test.csv', 
              select = c('id', 'Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK'))

#transform target variable to log(1 + demand) - this makes sense since we're 
#trying to minimize rmsle and the mean minimizes rmse:
train$log_demand = log1p(train$Demanda_uni_equil) 

#set a table key to enable fast aggregations
setkey(train, Producto_ID, Cliente_ID, Ruta_SAK)
setkey(test, Producto_ID, Cliente_ID, Ruta_SAK)

print("Computing means")
mean_total <- mean(train$log_demand) #overall mean
mean_Prod <-  train[, .(mean_prod = mean(log_demand)), by = .(Producto_ID)]
mean_Prod_Ruta <- train[, .(mean_prod_ruta = mean(log_demand)),
                        by = .(Producto_ID, Ruta_SAK)] #mean by product and ruta
mean_Client_Prod_Agencia <- train[, .(mean_client_prod = mean(log_demand)),
                                  by = .(Producto_ID, Cliente_ID, Agencia_ID)] #mean by product, client, agencia




#  set up custom evaluation metric
RMSLE <- function(preds, cv_dtrain) {
  labels <- getinfo(cv_dtrain, "label")
  
  # elab<-exp(as.numeric(labels))-1
  elab <- log(as.numeric(labels) + 1)
  # epreds<-exp(as.numeric(preds))-1
  epreds <- log(as.numeric(preds) +1 )
  # err <- sqrt(mean((epreds/elab-1)^2))
  err <- sqrt(mean(( epreds -   elab  )^2))
  return(list(metric = "RMSLE", value = err))
}


# last chance set of parameters based off of grid
param <- list(objective           = "reg:linear", 
              booster = "gbtree",
              eta                 = 0.03, # changed from an optimum 0.01
              max_depth           = 10, # changed from optimum 12, changed from default of 8
              gamma = 0.2,
              colsample_bytree    = 0.8, # 0.7
              #scale_pos_weight = 87,
              min_child_weight = 1
)





###################### Begin training XGBOOOOOOOOOOST!!!
# set the training rounds for both cv and testing
nrounds_cv = 3000
timer = proc.time() # start the timer

history <- xgb.cv(   params              = param, 
                     data                = cv_dtrain, 
                     nrounds             = nrounds_cv, # optimum 4000 #300, #280, #125, #250, # changed from 300
                     verbose             = 1, #changed from 0
                     # early.stop.round    = 100, #changed from optimum 100
                     watchlist           = watchlist,
                     maximize            = FALSE,
                     eval.metric         = "auc",
                     feval=RMSLE,
                     nfold=2,
                     prediction = TRUE # If TRUE data table results (history$dt) and predictions are provided
)
timer = proc.time() - timer  # end the timer
print(history$dt)

############## Plot the training/testing history
plot(history$dt$train.auc.mean,col='darkred',type='l',lwd=2)
lines(history$dt$train.auc.mean+ history$dt$train.auc.std,type="l",lty=2,col="red")
lines(history$dt$train.auc.mean- history$dt$train.auc.std,type="l",lty=2,col="red")

lines(history$dt$test.auc.mean,col='darkblue',type='l',lwd=2)
lines(history$dt$test.auc.mean+ history$dt$test.auc.std,type="l",lty=2,col="blue")
lines(history$dt$test.auc.mean- history$dt$test.auc.std,type="l",lty=2,col="blue")
legend(20,.9,c("mean train","mean test"),col=c("red","blue"),lty=c(1,1),cex=.9)
# 
