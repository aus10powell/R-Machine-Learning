## Notes:
# Prediction code used in predicting high-risk member within KP medical system


# - (Server 1) Running 21 vars based off gbm selection. Put gbm selection to default ntrees. RESULT: 0.907252828738396
# - (Server 2) Running 19 vars based off gbm selection. Put gbm selection to default ntrees. RESULT: 0.905411155412149

# - (Server 1) Running 22 vars based off saved gbm selection. Put gbm selectio: RESULT:  0.907779553477425
# - (server 1) Running 22 vars based off of only deep learning selection: RESULT: 0.898577232197662
# - (server 2) Running 22 vars based off 20 gbm vars the last 2 cg_2013 and cg_2012: RESULT: 0.906873452199543
# - (server 3) Running 22 vars based off of 20 gbm vars the last 13 from diff in deeplearning: RESULT: 0.908284456337447, adj 0.9022528
# - (server 1) Imputed missing vars using glrm then ran gbm var select. Using 21 from gbm var-select: RESULT:  0.906915207827353 un-adjusted
# - (server 1) Running 21 standard-selected gbm vars. Used glrm to replace ethnct_ds_tx (they are not continuous values though) RESULT: 0.907051235159444
# - (server 2) Running 21 standard-selected gbm vars. + artificially created rsk_grp var based off of er_stay_ct_pri: RESULT: 0.907567918131913
# - (server 2) Running 3 vars: c("pri_cst","cms_hcc_130_ct","cg_2014"): RESULT: 0.871496608745197
# - (servier 1) Running 12 standard gbm vars. RESULT: 0.904157930680199
# - (server 1) Running 12 gbm standard. scaled c("dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","rx_cncur_med_risk_qt","dx_cncur_med_risk_qt")
#              RESULT:  0.904329319545525 so...better
# - (server 1) Running 12 gbm standard scaled c("dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","rx_cncur_med_risk_qt","dx_cncur_med_risk_qt")
#              + "pri_cst" this time  RESULT: 0.904419503551068 BEST SO FAR
# - (server 2) Running 12 gbm standard scaled c("dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","rx_cncur_med_risk_qt","dx_cncur_med_risk_qt").
# - used kmeans to find centers by age and pri_cst to add new variable. RESULT: 0.903872644126164
# - (server 2) Running 12 gbm standard scaled AND centered c("dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","rx_cncur_med_risk_qt","dx_cncur_med_risk_qt","pri_cst)
# - RESULT:  0.904498818584977 BEST SO FAR
# - (server 2) Running 13 gbm standard scaled and centered for all continuous vars: 0.904301196833877
# - (server 1) Running grid search for xgboost on rstudio:
# - (server 3) Running standard 19var,scaled,centered and all continuous vars logged. RESULT: 0.905893038683869
# - (server 2) Running 11 gbm standard. First logged, then centered,scaled. RESULT:0.901893710149431
# - (server 2) Running 11 gbm standard. Only logged pri_cst, and removed h2o.gbm wrapper. RESULT: 0.900699306565771
# - (server )
##########################################################################################
library(h2o); library(e1071)
h2o.removeAll() # Clean slate - just in case the cluster was already running
h2o.init(nthreads=-1)

train <- h2o.importFile(path = normalizePath("hutrain.csv"))
test <- h2o.importFile(path = normalizePath("hutest.csv"))

temp_train = train
temp_test = test


# columns which are not in test also
# get columns exclusive training set
extra_cols = c("nxt_cst","hu","hu_01","hu_02","hu_05")

# set predictor
y = temp_train$hu_01
train = train[,-which(names(train) %in% extra_cols)]

# put test and train into one dataset for pre-processing
whole_hog = h2o.rbind(train,test)

# these columns have one value and also dropping member id
cols_to_delete = c("cms_mdl_ver_tx","cms_mcare_advntg_in_cd","cops_qt","rho2_qt","ablaps_qt",
                   "score_ver_nb","rho2_qt_p","ablaps_qt_p","clndr_mm_sk","fnd_cst","fnd_rsk","mbr_id")
whole_hog = whole_hog[,-which(colnames(whole_hog) %in% cols_to_delete)]


# Need to pay attention to below
cms_hcc_columns = colnames(whole_hog[,20:89])
var_factors = c("prmy_wrtn_lang_cd","gndr_cd","hispanic_in_cd","race_ds_tx","ethnct_ds_tx"
                ,"prmy_spkn_lang_cd","prmy_wrtn_lang_cd","paneled","ob_hs","ed_hs","admt_ct_pri",
                cms_hcc_columns,"cg_2014","cg_2013","cg_2012","paneled")
for (i in 1:length(var_factors)){
  whole_hog[,var_factors[i]] = as.factor(whole_hog[,var_factors[i]])
}


############
# Replacing Missing Values/ Recatagorizing 
###########
whole_hog[,"race_ds_tx"] <- ifelse(whole_hog[,"race_ds_tx"] == "?", "UNKNOWN", whole_hog[,"race_ds_tx"])
whole_hog[,"ethnct_ds_tx"] <- ifelse(whole_hog[,"ethnct_ds_tx"] == "?", NA, whole_hog[,"ethnct_ds_tx"])




#### scaling - does include "pri_cst" 
#vars_to_scale = c("dx_cncur_med_risk_qt","dx_prspct_med_risk_qt","rx_cncur_med_risk_qt",
#                  "rx_prspct_ttl_risk_qt","rx_inpat_prspct_ttl_risk_qt","mcare_cncur_med_risk_qt",
##                  "mcare_prspct_med_risk_qt","loh_prspct_qt","rx_prspct_ttl_risk_qt_p",
#                  "mcare_prspct_med_risk_qt_p","ndi")

vars_to_scale = c("dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt","rx_cncur_med_risk_qt","dx_cncur_med_risk_qt","pri_cst")
whole_hog[,vars_to_scale] = scale(whole_hog[,vars_to_scale],scale=TRUE, center=TRUE)


log_pri_cst = log(whole_hog$pri_cst + 2)

# 
#vars_to_scale2 = c("dx_cncur_med_risk_qt","dx_prspct_med_risk_qt","rx_cncur_med_risk_qt","rx_prspct_ttl_risk_qt","mcare_cncur_med_risk_qt","mcare_prspct_med_risk_qt","loh_prspct_qt","mcare_prspct_med_risk_qt_p")
#whole_hog[,vars_to_scale2] = scale(whole_hog[,vars_to_scale2],scale=TRUE,center=TRUE)

######### PCA
# looking at some variable importance and pca
#train.hex.pca <- h2o.prcomp(training_frame = whole_hog, transform = "STANDARDIZE",k = ncol(whole_hog))
#train.hex.pca

######## VARIABLE IMPORTANCE 
# X_train = h2o.cbind(whole_hog[1:1000000,],y)
# x1 <- setdiff(names(X_train), "hu_01")
#   X_train[,c("hu_01")] <- as.factor(X_train[,c("hu_01")]) 

## Variable importance using deep learning
#my.dl <- h2o.deeplearning(x=x1, y="hu_01",training_frame = X_train,  distribution = "bernoulli",variable_importances = TRUE) 
#dl.impvariables = h2o.varimp(my.dl)
#dl.impvariables = data.frame(dl.impvariables)
# saveRDS(dl.impvariables,"dl_import_vars.rds")



###### Variable importance using gbm
# Run GBM with variable importance
# my.gbm <- h2o.gbm(x = x1, y = "hu_01", distribution = "bernoulli", training_frame = X_train, ntrees =100, max_depth=2)

# Access Variable Importance from the built model
# gbm.VI = data.frame(my.gbm@model$variable_importances)
# gbm.var_importance = gbm.VI[1]$variable

# Plot variable importance from GBM
#barplot(t(gbm.VI[2]),main="VI from GBM",names.arg = gbm.VI[1]$variable)

#gbm_import_vars = head(gbm.VI[1],21)$variable

head(gbm.var_importance,21)
# saveRDS(gbm.VI[1]$variable,"gbm_import_vars.rds")
#gbm.var_importance=  readRDS("gbm_import_vars.rds")
###### Variable importance using Random forrest
#hu.rf = h2o.randomForest(y = "hu_01", x = x1, training_frame = X_train)

#Lets look at variables since we will get penalized if used all variables
#impvariables = h2o.varimp(hu.rf)

## which vectors the two DO have
#setdiff(head(gbm_import_vars,n=26), head(impvariables$variable,n=26))
## which vectors the two DO have
#important_vars = intersect(head(gbm_import_vars,n=26), head(impvariables$variable,n=26))


#impvariables = readRDS("rf_impvariables.rds")
#saveRDS(impvariables,"rf_impvariables_newest.rds")
# rf.impvariables = readRDS("rf_impvariables_newest.rds")

# get difference in variable importance from models
# rf_vs_gbm =  setdiff(head(rf.impvariables$variable,n=22),head(gbm.var_importance,n=22)   )
dl_vs_gbm = setdiff(c("cg_2014","cg_2013","admt_ct_pri","cg_2012","ie_outside_ct_pri","hedis_rdmt_indx_ct_pri","ae_elig_mm_ct",
                      "cms_hcc_105_ct","mcare_entl_elig_mm_ct","hosp_day_ct_pri","pri_cst","rx_elig_mm_ct",
                      "gndr_cd","cms_hcc_105_ct","age_yr_nb","mbrshp_ct","cmnty_score_qt","hedis_rdmt_r30_ct_pri",
                      "cmnty_score_qt_p","hedis_rdmt_r30_ct_pri","cmnty_score_qt_p","cms_hcc_27_ct")  ,head(gbm.var_importance,n=22) )

gbm.var_importance=  readRDS("gbm_import_vars.rds")
important_vars = head(gbm.var_importance,12)
temp = whole_hog
whole_hog = temp

whole_hog = whole_hog[,which(names(whole_hog) %in% important_vars)]

################# IMPUTATION ############################################################


## Impute missing values based on group-by on targets
targets <- colnames(whole_hog)
vars_miss = vector(); j = 0
for (i in 1:ncol(whole_hog)) {
  if (sum(is.na(whole_hog[,i]))==0 || sum(is.na(whole_hog[,i])) == nrow(whole_hog)) next
    j = j+ 1
    vars_miss[j] = colnames(whole_hog[,i])
}
vars_miss
head(whole_hog[,vars_miss])
tail(whole_hog[,vars_miss])
# temporarily setting up new variables i think would be good cindidates for imputation
vars_miss = c("ethnct_ds_tx")
cols_for_kmeans = c("pri_cst","dx_prspct_med_risk_qt","rx_prspct_ttl_risk_qt" ,"cms_hcc_130_ct","cg_2014")
############ BEGIN NEW FEATURE TESTING
## Impute missing values based on group-by on targets
num_col_missing =vector()
for (i in 1:length(vars_miss)){num_col_missing[i] = which(colnames(whole_hog) == vars_miss[i])}
num_col_missing


gait.glrm2 <- h2o.glrm(training_frame = whole_hog, 
                       k = ncol.H2OFrame(whole_hog) , init = "SVD", svd_method = "GramSVD",
                       loss = "Quadratic", regularization_x = "None", regularization_y = "None", 
                       max_iterations = 2000, min_step_size = 1e-6)

plot(gait.glrm2)
summary(gait.glrm2)
# Impute missing values in our training data from X and Y.
gait.pred2 <- predict(gait.glrm2, whole_hog[,num_col_missing])
head(gait.pred2); head(whole_hog[,num_col_missing])

whole_hog$ethnct_ds_tx = gait.pred2$reconstr_ethnct_ds_tx


whole_hog[,vars_miss] = ifelse(is.na(whole_hog[,vars_miss]),gait.pred2,whole_hog[,vars_miss])
sum(is.na(whole_hog))

# where, a.hex is the original h2o dataframe and pr.hex is the prediction h2oframe from glrm.

# saveRDS(as.data.frame(gait.pred2),"gait_pred2.rds")
# replaced_vars = readRDS("gait_pred2.rds)

###################### END IMPUTATION ###########################################

X_train = h2o.cbind(whole_hog[1:1000000,],temp_train$hu_01)



#############  END NEW FEATURE TESTING




##########################################################################################
# BEGIN Cross-Training Section
##########################################################################################
library(h2oEnsemble)
library(caret)


k_folds = 5
set.seed(1241)
trainIndex = createDataPartition(as.vector(X_train$hu_01),p = 0.6, list=FALSE, times = k_folds)

ptm <- proc.time()
for (fold in 1:k_folds){  
  training_frame = X_train[as.vector(trainIndex[,fold]),]
  validation_frame = X_train[-as.vector(trainIndex[,fold]),]
  
  
  family <- "binomial"
  x <- setdiff(names(X_train), "hu_01")
  training_frame[,c("hu_01")] <- as.factor(training_frame[,c("hu_01")]) 
  validation_frame[,c("hu_01")] <- as.factor(validation_frame[,c("hu_01")]) 
  
  h2o.randomForest.3 <- function(..., ntrees = 500, sample_rate = 0.85, seed = 1234) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
  h2o.gbm.1 <- function(..., ntrees = 100,max_depth = 6,learn_rate = .05, seed = 1234) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
  
  
  learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
               "h2o.gbm.wrapper", "h2o.deeplearning.wrapper",
               "h2o.randomForest.3","h2o.gbm.1")
  
  ptm <- proc.time() # start the clock
  fit <- h2o.ensemble(x = x, y = "hu_01", seed = 1234,
                      training_frame = training_frame, 
                      family = family, 
                      learner = learner, 
                      metalearner = "h2o.gbm.1",
                      cvControl = list(V = 5, shuffle = TRUE))
  proc.time() - ptm # how long did the code run?
  
  
  h2o.save_ensemble(fit,path = "./h2o-ensemble-model-save")
  
  # measure performance
  performance_ens = h2o.ensemble_performance(fit, newdata = validation_frame)
  performance_ens
  
  # Re-trains an existing H2O Ensemble fit using a new metalearning function.
 # h2o.glm_nn <- function(..., non_negative = TRUE) h2o.glm.wrapper(..., non_negative = non_negative)
 # newfit <- h2o.metalearn(fit, metalearner = "h2o.glm_nn")
 # h2o_glm_nn_performance_ens = h2o.ensemble_performance(newfit, newdata = validation_frame)
 # h2o_glm_nn_performance_ens
  

  # try with deep-learning
 # newfit_dl <- h2o.metalearn(fit, metalearner = "h2o.deeplearning.wrapper")
 # h2o_dl_performance_ens = h2o.ensemble_performance(newfit_dl, newdata = validation_frame)
 # h2o_dl_performance_ens
  
  # try with rf
 # newfit_rf <- h2o.metalearn(fit, metalearner = "h2o.randomForest.3")
 # h2o_rf_performance_ens = h2o.ensemble_performance(newfit_rf, newdata = validation_frame)
 # h2o_rf_performance_ens
  
  # evaluate performance
  # until I figure out the data structure to average performance more easily
  cross_performance = list(0)
  if (fold == 1){ 
    cross_performance[[1]] = performance_ens
  } else if(fold == 2) {
    cross_performance[[2]] = performance_ens
  } else if (fold == 3) {
    cross_performance[[3]] = performance_ens
  } else if (fold == 4) {
    cross_performance[[4]] = performance_ens
  } else if (fold == 5) {
    cross_performance[[5]] = performance_ens
  }
}
proc.time() - ptm


##########################################################################################
# END Cross-Training Section
##########################################################################################


##########################################################################################
# BEGIN Current working code
##########################################################################################
 
library(h2oEnsemble)
library(caret)


splits<-h2o.splitFrame(X_train, ratios = 0.6, destination_frames = c("training_frame","validation_frame"), seed = 1234)
training_frame <- splits[[1]]
validation_frame <- splits[[2]]
training_frame[,c("hu_01")] <- as.factor(training_frame[,c("hu_01")])
validation_frame[,c("hu_01")] <- as.factor(validation_frame[,c("hu_01")])


# check dimensions of different data frames
print("training_frame");dim(training_frame); print("validation_frame"); dim(validation_frame)

family <- "binomial"
x <- setdiff(names(X_train), "hu_01")

h2o.randomForest.3 <- function(..., ntrees = 500, sample_rate = 0.85, seed = 1234) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100,max_depth = 6,learn_rate = .05, seed = 1234) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 1000, sample_rate = 0.632,max_depth = 15,mtries=6,col_sample_rate_per_tree=.8, seed = 1234) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.glm.1 <- function(..., alpha = 0.0,lambda=0.1) h2o.glm.wrapper(..., alpha = alpha, lambda=lambda)
h2o.gbm.2 <- function(..., ntrees = 200,max_depth = 6,learn_rate = .05, seed = 1234) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)


# sample_rate ntrees max_depth mtries col_sample_rate_per_tree            model_ids               auc
#   0.632     50        15      6                      0.8 rf_grid_id_model_175 0.900259723798636

learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper",
             "h2o.randomForest.4","h2o.gbm.2","h2o.glm.1")

ptm <- proc.time() # start the clock
fit <- h2o.ensemble(x = x, y = "hu_01", seed = 1234,
                    training_frame = training_frame, 
                    #validation_frame = validation_frame,
                    family = family, 
                    learner = learner, 
                    metalearner = "h2o.gbm.2",
                    cvControl = list(V = 5, shuffle = TRUE))
proc.time() - ptm # how long did the code run?


h2o.save_ensemble(fit,path = "./h2o-ensemble-model-save")
# fit <- h2o.load_ensemble(path = "./h2o-ensemble-model-save")
# measure performance
performance_ens = h2o.ensemble_performance(fit, newdata = validation_frame)
performance_ens


# try with rf
newfit_rf <- h2o.metalearn(fit, metalearner = "h2o.randomForest.4")
h2o_rf_performance_ens = h2o.ensemble_performance(newfit_rf, newdata = validation_frame)
h2o_rf_performance_ens



submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid by replacing p624626 
  #filename = paste0("/hu/output/",filename,"p624626.csv")
  write.csv(submission, filename,row.names = FALSE)
}

proc.time() - timer


##########################################################################################
# END Testing New Features
##########################################################################################




##########################################################################################
# BEGIN xgboost implementation
##########################################################################################
library(xgboost)
head(X_train)
temp = whole_hog


whole_hog$paneled = as.numeric(whole_hog$paneled)
whole_hog$ethnct_ds_tx = as.numeric(whole_hog$ethnct_ds_tx)


library(h2oEnsemble)
library(caret)

X_train = h2o.cbind(whole_hog[1:1000000,],temp_train$hu_01)

splits<-h2o.splitFrame(X_train, 0.6, destination_frames = c("training_frame","validation_frame"), seed = 1234)
training_frame <- splits[[1]]
validation_frame <- splits[[2]]
training_frame[,c("hu_01")] <- as.factor(training_frame[,c("hu_01")])
validation_frame[,c("hu_01")] <- as.factor(validation_frame[,c("hu_01")])

training_frame[,c("hu_01")] <- as.numeric(training_frame[,c("hu_01")])
validation_frame[,c("hu_01")] <- as.numeric(validation_frame[,c("hu_01")])

test = as.data.frame(validation_frame)
train = as.data.frame(training_frame)

dval<-xgb.DMatrix(data=data.matrix(test[,-c(which(colnames(test) %in% "hu_01"))]),label=test$hu_01,missing=NaN) # 
dtrain<-xgb.DMatrix(data=data.matrix(train[,-c(which(colnames(train) %in% "hu_01"))]),label=train$hu_01,missing=NaN)
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "binary:logistic", 
                eval_metric = "auc",
                booster = "gbtree", # changed from gblinear
                eta                 = 0.02, # changed from an optimum 0.02. 0.06, #0.01,
                max_depth           = 6, # changed from optimum 12, changed from default of 8
                subsample           = 0.75, # changed from optimum 0.9, 0.7
                colsample_bytree    = 0.7, # 0.7
                num_parallel_tree   = 50,
                set.seed = 935,
                alpha = 0.02 # default alpha = 0.0001, 
                #lambda = .9
                # gamma = .9
)


timer = proc.time() # start the timer
setSessionTimeLimit(cpu = Inf, elapsed = Inf)
setTimeLimit(cpu = Inf, elapsed = Inf)
history <- xgb.train(   params              = param, 
                        data                = dtrain, 
                        nrounds             = 500, # optimum 4000 #300, #280, #125, #250, # changed from 300
                        verbose             = 1, #changed from 0
                        early.stop.round    = 300, #changed from optimum 100
                        watchlist           = watchlist,
                        maximize            = TRUE,
                        #feval=RMPSE,
                        nfold=5,
                        # nround=2,
                        prediction = TRUE # If TRUE data table results (history$dt) and predictions are provided
)
 proc.time() - timer  # end the timer


preds <- predict(history, data.matrix(test[,feature.names[c(-1,-2)]]))


preds = ifelse(preds < 0.5,0,1)
submission = data.frame(PassengerId = test$PassengerId, Survived = preds)

##########################################################################################
# END xgboost implementation
##########################################################################################


##########################################################################################
# BEGIN grid testing 
##########################################################################################

# Random Grid Search (e.g. 120 second maximum)
# This is set to run fairly quickly, increase max_runtime_secs 
# or max_models to cover more of the hyperparameter space.
# Also, you can expand the hyperparameter space of each of the 
# algorithms by modifying the hyper param code below.
ptm <- proc.time() # start the clock
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 120)
nfolds <- 5


# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.03) 
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt, 
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

gbm_grid <- h2o.grid("gbm", x = x, y = y,
                     training_frame = training_frame,
                     ntrees = 100,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)
gbm_models <- lapply(gbm_grid@model_ids, function(model_id) h2o.getModel(model_id))



# RF Hyperparamters
mtries_opt <- 8:20 
max_depth_opt <- c(5, 10, 15, 20, 25)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_per_tree_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(mtries = mtries_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate_per_tree = col_sample_rate_per_tree_opt)

rf_grid <- h2o.grid("randomForest", x = x, y = y,
                    training_frame = training_frame,
                    ntrees = 200,
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,                    
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
rf_models <- lapply(rf_grid@model_ids, function(model_id) h2o.getModel(model_id))



# Deeplearning Hyperparamters
activation_opt <- c("Rectifier", "RectifierWithDropout", 
                    "Maxout", "MaxoutWithDropout") 
hidden_opt <- list(c(10,10), c(20,15), c(50,50,50))
l1_opt <- c(0, 1e-3, 1e-5)
l2_opt <- c(0, 1e-3, 1e-5)
hyper_params <- list(activation = activation_opt,
                     hidden = hidden_opt,
                     l1 = l1_opt,
                     l2 = l2_opt)

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    training_frame = training_frame,
                    epochs = 15,
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,                    
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
dl_models <- lapply(dl_grid@model_ids, function(model_id) h2o.getModel(model_id))



# GLM Hyperparamters
alpha_opt <- seq(0,1,0.1)
lambda_opt <- c(0,1e-7,1e-5,1e-3,1e-1)
hyper_params <- list(alpha = alpha_opt,
                     lambda = lambda_opt)

glm_grid <- h2o.grid("glm", x = x, y = y,
                     training_frame = training_frame,
                     family = "binomial",
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,                    
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)
glm_models <- lapply(glm_grid@model_ids, function(model_id) h2o.getModel(model_id))


# Create a list of all the base models
models <- c(gbm_models, rf_models, dl_models, glm_models)

# Specify a defalt GLM as the metalearner
metalearner <- "h2o.glm.wrapper"

# Let's stack!
stack <- h2o.stack(models = models, 
                   response_frame = training_frame[,y],
                   metalearner = metalearner)

# Compute test set performance:
perf <- h2o.ensemble_performance(stack, newdata = validation_frame)
print(perf)

proc.time() - ptm # how long did the code run?

# save h2o Ensemble model to disk
h2o.save_ensemble(perf) 


# Now let's refit the metalearner using a DL and GLM-NN
stack2 <- h2o.metalearn(stack, metalearner = "h2o.deeplearning.wrapper")
perf2 <- h2o.ensemble_performance(stack2, newdata = validation_frame, score_base_models = FALSE)
print(perf2)

#0.854221823457336
#####################################################
# MAKE A PREDICTION
####################################################

X_test = h2o.cbind(whole_hog[1000001:dim(whole_hog)[1],])
#Predict
h2o_predictions = h2o.predict(object = hu.gbm , newdata = X_test)

# Convert to R
prediction = as.data.frame(h2o_predictions$p1)
head(prediction)
summary(prediction)

#Creating a submission frame
submitter(as.numeric(as.vector(test$mbr_id)),as.numeric(as.vector(prediction$p1)),"husubmission_gbm_wrapper.csv")


submitter <- function(id,predictions,filename)
{ 
  submission<-cbind(id,predictions)
  colnames(submission) <- c("mbr_id", "prediction")
  submission <- as.data.frame(submission)
  #add your nuid by replacing p624626 
  #filename = paste0("/hu/output/",filename,"p624626.csv")
  write.csv(submission, filename,row.names = FALSE)
}






