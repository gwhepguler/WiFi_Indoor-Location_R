# Title: Evaluate Techniques for Wifi Locationing
#
# Last update: 2018.07.31
# Updated by : Greg Hepguler

# File: Wifi-Locations_Classification_0731v1.R

# Q1: Investigate the feasibility of using "wifi fingerprinting" to determine 
#     a person's location in indoor spaces
# Q2: Evaluate multiple machine learning models to see which produces the best result

###############
# Project Notes
###############

# Summarize project: 
# Outdoor locationing problems can be solved accurately using signals of GPS sensors in the mobile devices
# However, indoor locationing is a different problem because the loss of GPS signal in indoor environments
# We evaluated the application of machine learning techniques to this problem, replacing the GPS signal with
# WAPS signal.

###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory #  ?getwd  # >> get help
getwd()     

# set working directory
setwd("D:/GH14/DATA-SCIENCE/RProjects/C3T3")
dir()

################
# Load packages
################

#caret
#http://topepo.github.io/caret/bytag.html
#model training: http://topepo.github.io/caret/bytag.html
#model measurement: http://topepo.github.io/caret/other.html

install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages("munsell")
install.packages("inum")
install.packages("corrplot")
install.packages("class")

install.packages("ggplot2")
install.packages("lubridate")
install.packages("dplyr")
install.packages("zoo")
install.packages("tidyr")
install.packages("plotly")
install.packages("ggfortify")
install.packages("forecast")

install.packages("ISLR")
install.packages("lattice")
install.packages("ggmap")
install.packages("caTools")
install.packages("gridExtra")
install.packages("ranger")
install.packages("e1071")

#load library
library(caret)
library(munsell)
library(inum)
library(corrplot)
library(class)

library(ggplot2)
library(lubridate) # work with dates
library(dplyr)     # data manipulation (filter, summarize, mutate)
library(zoo)
library(tidyr)
library(plotly)
library(ggfortify)
library(forecast)

library(ISLR)
library(lattice)
library(ggmap)
library(caTools)
library(gridExtra)
library(ranger)
library(e1071)

library(mlbench)
library(randomForest)
library(C50)

###############
# Import data
##############

# Load Traininig data 

Training <- read.csv("C3T3_TrainingData.csv", header=TRUE, sep=",",  stringsAsFactors=FALSE)

################
# Evaluate data
################

# --- Dataset 1: Training Data --- #

str(Training[,519:529])  # 19937 obs. of 529 variables
summary(Training[,519:529])
names(Training)

# plot location 
plot(Training$LONGITUDE, Training$LATITUDE, xlab = "LONGITUDE", ylab="LATITUDE", 
     main="UJIIndoorLoc Training Dataset")

# check for missing values 
anyNA(Training)   # FALSE
                  # is.na(Training)

# --- Universitat of Jaume I in real life  ---#

univ <- c(-0.067417, 39.992871)
map1 <- get_map(univ, zoom = 17, scale = 1)
# Reference
# https://maps.googleapis.com/maps/api/staticmap?center=39.992871,-0.067417&zoom=17&size=640x640&scale=1&maptype=terrain&language=en-EN
ggmap(map1)

#############
# Preprocess
#############

# Remove repeated rows in training dataset             
Training <- distinct(Training)             
str(Training[,519:529])   # 19300 obs. of 529 variables --------------------  # 19937 to 19300  -----
summary(Training[,521:529])

# Keep original dataset
data <- Training

# define LOCATION that unites BUILDINGID, FLOOR, SPACEID
data$LOCATION <- as.integer(group_indices(data, BUILDINGID, FLOOR, SPACEID))
head (data$LOCATION)
data$BUILDINGID

# data$LOCATION <- as.factor(data$LOCATION)
# Transform some variables to factor/numeric/datetime
# factors<-c("BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID")
# data[,factors] <-lapply(data[,factors], as.factor)
# rm(factors)

numeric<-c("LONGITUDE", "LATITUDE")
data[,numeric]<-lapply(data[,numeric], as.numeric)
rm(numeric)

data$TIMESTAMP <- as.POSIXct(data$TIMESTAMP,
                                   origin = "1970-01-01", tz="UTC")
summary(data[,1:10])
summary(data[,519:530])
str(data[,519:530])

##########################
# Feature Engineering (FE)
##########################

# unique(data$LOCATION) # 735 levels

check_ID<-data%>%
  arrange(BUILDINGID, FLOOR, SPACEID, LATITUDE, LONGITUDE)%>%
# distinct(BUILDINGID, FLOOR, SPACEID, LOCATION, LATITUDE, LONGITUDE)%>%
  select(BUILDINGID, FLOOR, SPACEID, LOCATION, LATITUDE, LONGITUDE)     #sIDs assigned sequentially

check_ID
tail(check_ID)

# rm(check_ID))

##################
# Train/test sets
##################

set.seed(998)

inTraining <- createDataPartition(data$LOCATION, p = .70, list = FALSE)
training <- data[inTraining,]
validation <- data[-inTraining,]
summary(validation)

# sample <- sample.split(data, SplitRatio = .70)
# training <- subset(data, sample ==TRUE)
# validation <- subset(data, sample == FALSE)

summary(training[,519:530])
str(training[,519:530])
str(validation[,519:530])

# Separate data by BUILDINGID for training & validation 
build.0 <- filter(training, BUILDINGID == 0)
build.1 <- filter(training, BUILDINGID == 1)
build.2 <- filter(training, BUILDINGID == 2)

build.0.v <- filter(validation, BUILDINGID == 0)
build.1.v <- filter(validation, BUILDINGID == 1)
build.2.v <- filter(validation, BUILDINGID == 2)
str(build.0.v[,519:529])

# Create a data frame for each feature
# TRAIN ----

# Build 0
build.0.loc <- data.frame(build.0$LOCATION, build.0[,1:520])
summary(build.0.loc[,510:521])
summary(build.0.loc[,1:10])
str(build.0.loc[,510:521])

#Build 1
build.1.loc <- data.frame(build.1$LOCATION, build.1[,1:520])

#Build 2
build.2.loc <- data.frame(build.2$LOCATION, build.2[,1:520])

# VALID ----

#Build 0
build.0.loc.v <- data.frame(build.0.v$LOCATION, build.0.v[,1:520])
str(build.0.loc.v[,1:10])

#Build 1
build.1.loc.v <- data.frame(build.1.v$LOCATION, build.1.v[,1:520])
str(build.1.loc.v[,1:10])

#Build 2
build.2.loc.v <- data.frame(build.2.v$LOCATION, build.2.v[,1:520])
str(build.2.loc.v[,1:10])


# Convert LOCATION feature to factor in validation (test) set
build.0.loc.v$build.0.v.LOCATION <- as.factor(build.0.loc.v$build.0.v.LOCATION)
build.1.loc.v$build.1.v.LOCATION <- as.factor(build.1.loc.v$build.1.v.LOCATION)
build.2.loc.v$build.2.v.LOCATION <- as.factor(build.2.loc.v$build.2.v.LOCATION)

# Sample TRAIN data set 2000 records ----
# Build 0 
sample.build.0.loc <- build.0.loc[sample(1:nrow(build.0.loc), 2000, replace = FALSE),]
#Build 1
sample.build.1.loc <- build.1.loc[sample(1:nrow(build.1.loc), 2000, replace = FALSE),]
#Build 2
sample.build.2.loc <- build.2.loc[sample(1:nrow(build.2.loc), 2000, replace = FALSE),]

# LOCATION is the  feature used in classification problem ---- convert to factor
# Convert LOCATION feature to factor in training set
sample.build.0.loc$build.0.LOCATION <- as.factor(sample.build.0.loc$build.0.LOCATION)
summary(sample.build.0.loc$build.0.LOCATION) 
summary(sample.build.0.loc[,1:10])
summary(sample.build.0.loc[,510:521])
str(sample.build.0.loc[,1:10])

sample.build.1.loc$build.1.LOCATION <- as.factor(sample.build.1.loc$build.1.LOCATION)
summary(sample.build.1.loc$build.1.LOCATION) 
summary(sample.build.1.loc[,1:10])
summary(sample.build.1.loc[,510:521])
str(sample.build.1.loc[,1:10])

sample.build.2.loc$build.2.LOCATION <- as.factor(sample.build.2.loc$build.2.LOCATION)
summary(sample.build.2.loc$build.2.LOCATION) 
summary(sample.build.2.loc[,1:10])
summary(sample.build.2.loc[,510:521])
str(sample.build.2.loc[,1:10])

# Not using droplevels to avoid dimension problems
# sample.build.0.loc$build.0.LOCATION  <- droplevels(sample.build.0.loc$build.0.LOCATION )

################
# Train control
################

set.seed(998)
metric1 <- "Accuracy"

# 10 fold cross validation
# fitCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
#                            verboseIter = TRUE)

# 10 fold cross validation, RF and C5.0 trContrl
# reduce repeats after initial runs
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                           verboseIter = TRUE)


# ########## Predict LOCATION (BUILDINGID, FLOOR, SPACEID) ###########

# Knn, Random Forest, C5.0 Algorithms have been implemented to predict 
# LOCATION at each building (0, 1, 2) :

#############
# BUILDING 0
#############

# kNN -------------------------------------------------------------
set.seed(998)

# The first (previous) run did not specify tunegrid
# See accompanying xls file for the results from the first run:
# The final value selected from the previous run was k = 5.

# This run is for k values 1 to 4
knnGrid <- expand.grid(.k=c(1:4))   # Last run
knn.loc.0 <- train(build.0.LOCATION ~ ., 
                     sample.build.0.loc,
                     method = "knn", metric=metric1,
                     tuneLength = 7, tuneGrid = knnGrid,
                     trControl = fitControl,
                     preProcess = c("zv", "medianImpute"))
knn.loc.0

pred.knn.loc.0 <- predict(knn.loc.0, build.0.loc.v)
summary(pred.knn.loc.0)

results.knn.loc.0 <- postResample(pred.knn.loc.0, build.0.loc.v$build.0.v.LOCATION)
results.knn.loc.0
accuracy.knn.0 <- results.knn.loc.0[1]
accuracy.knn.0
kappa.knn.0 <- results.knn.loc.0[2]
kappa.knn.0

conf.matrix.knn.loc.0 <- table(pred.knn.loc.0, build.0.loc.v$build.0.v.LOCATION)
conf.matrix.knn.loc.0[1:30,1:30]

# Test Set Results Max Kuhn
ConfusionMtrx <- confusionMatrix(pred.knn.loc.0, build.0.loc.v$build.0.v.LOCATION)
# CM$table
ConfusionMtrx$overall
ConfusionMtrx$byClass[1:20,]

# Random Forest ---------------------------------------------------
set.seed(998)

# dataframe for manual tuning of mtry
# rfGrid <- expand.grid(mtry=c(1,2,3,5))    # 1st Run: Fitting mtry = 5 on full training set
# rfGrid <- expand.grid(mtry=c(2,5))  
rfGrid <- expand.grid(mtry=c(16, 32, 48))   # Last run
rfor.loc.0 <- train(build.0.LOCATION ~ ., 
                  sample.build.0.loc,
                  method = "rf", tuneGrid=rfGrid, metric=metric1,
                  trControl = fitControl,
                  preProcess = c("zv", "medianImpute"))

# training results rfor
rfor.loc.0

# prediction
pred.rfor.loc.0 <- predict(rfor.loc.0, build.0.loc.v)
summary(pred.rfor.loc.0)
str(build.0.loc.v[,1:10])

results.rfor.loc.0 <- postResample(pred.rfor.loc.0, build.0.loc.v$build.0.v.LOCATION)
results.rfor.loc.0

accuracy.rfor.0 <- results.rfor.loc.0[1]
accuracy.rfor.0
kappa.rfor.0 <- results.rfor.loc.0[2]
kappa.rfor.0
  
conf.matrix.rfor.loc.0 <- table(pred.rfor.loc.0, build.0.loc.v$build.0.v.LOCATION)
conf.matrix.rfor.loc.0[1:20,1:20]

ConfusionMtrx <- confusionMatrix(pred.rfor.loc.0, build.0.loc.v$build.0.v.LOCATION)
ConfusionMtrx$overall
ConfusionMtrx$byClass[1:20,]


# C5.0 ------------------------------------------------------------

set.seed(998)

# First run
# grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(2,8,24), .model="tree" )
# Next run
# grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(32, 48, 64), .model = c("tree", "rules") )

# Last run
grid <- expand.grid( .winnow = FALSE, .trials=c(48, 96), .model = "rules" )
c50.loc.0 <- train(build.0.LOCATION ~ ., 
                    sample.build.0.loc,
                    method = "C5.0", tuneGrid=grid, metric=metric1,
                    tuneLength=2,
                    trControl = fitControl,
                    preProcess = c("zv", "medianImpute"))

# training results c50
c50.loc.0

# prediction
pred.c50.loc.0 <- predict(c50.loc.0, build.0.loc.v)
summary(pred.c50.loc.0)
str(build.0.loc.v[,1:10])

results.c50.loc.0 <- postResample(pred.c50.loc.0, build.0.loc.v$build.0.v.LOCATION)
results.c50.loc.0

accuracy.c50.0 <- results.c50.loc.0[1]
accuracy.c50.0
kappa.c50.0 <- results.c50.loc.0[2]
kappa.c50.0

conf.matrix.c50.loc.0 <- table(pred.c50.loc.0, build.0.loc.v$build.0.v.LOCATION)
conf.matrix.c50.loc.0[1:20,1:20]

ConfusionMtrx <- confusionMatrix(pred.c50.loc.0, build.0.loc.v$build.0.v.LOCATION)
ConfusionMtrx$overall
ConfusionMtrx$byClass[1:20,]

#############
# BUILDING 1
#############

# Predict LOCATION (BUILDINGID, FLOOR, SPACEID)

# kNN -------------------------------------------------------------
set.seed(998)

# NOTE : The first (previous) run did not specify tunegrid
#        Results from the first run are in the accompanying xls file

# This run is for k values 1 to 4
knnGrid <- expand.grid(.k=c(1:4))   # Last run
knn.loc.1 <- train(build.1.LOCATION ~ ., 
                   sample.build.1.loc,
                   method = "knn", metric=metric1,
                   tuneLength = 7, tuneGrid = knnGrid,
                   trControl = fitControl,
                   preProcess = c("zv", "medianImpute"))
knn.loc.1

nlevels(knn.loc.1)

pred.knn.loc.1 <- predict(knn.loc.1, build.1.loc.v)
summary(pred.knn.loc.1)
nrow(knn.loc.1)

results.knn.loc.1 <- postResample(pred.knn.loc.1, build.1.loc.v$build.1.v.LOCATION)
results.knn.loc.1
accuracy.knn.1 <- results.knn.loc.1[1]
accuracy.knn.1
kappa.knn.1 <- results.knn.loc.1[2]
kappa.knn.1

conf.matrix.knn.loc.1 <- table(pred.knn.loc.1, build.1.loc.v$build.1.v.LOCATION)
conf.matrix.knn.loc.1[1:20,1:20]

# Random Forest ---------------------------------------------------
set.seed(998)

# dataframe for manual tuning of mtry
# rfGrid <- expand.grid(mtry=c(1,2,3,5))  # 1st Run: Fitting mtry = 5 on full training set
# rfGrid <- expand.grid(mtry=c(5, 10)) 
rfGrid <- expand.grid(mtry=c(16, 32, 48)) # Last run
rfor.loc.1 <- train(build.1.LOCATION ~ ., 
                    sample.build.1.loc,
                    method = "rf", tuneGrid=rfGrid, metric=metric1,
                    trControl = fitControl,
                    preProcess = c("zv", "medianImpute"))

# training results rfor
rfor.loc.1

# prediction
pred.rfor.loc.1 <- predict(rfor.loc.1, build.1.loc.v)
summary(pred.rfor.loc.1)
str(build.1.loc.v[,1:10])

results.rfor.loc.1 <- postResample(pred.rfor.loc.1, build.1.loc.v$build.1.v.LOCATION)
results.rfor.loc.1

accuracy.rfor.1 <- results.rfor.loc.1[1]
accuracy.rfor.1
kappa.rfor.1 <- results.rfor.loc.1[2]
kappa.rfor.1

conf.matrix.rfor.loc.1 <- table(pred.rfor.loc.1, build.1.loc.v$build.1.v.LOCATION)
conf.matrix.rfor.loc.1[1:20,1:20]

# C5.0 ------------------------------------------------------------

set.seed(998)

# dataframe for manual tuning of mtry

# grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(2,8,24), .model="tree" )
# The final values used for the model were trials = 24, model = tree and winnow = FALSE.

# Number of boosting iterations must be between 1 and 100

grid <- expand.grid( .winnow = FALSE, .trials=c(48, 96), .model = "rules" )  # Last run
c50.loc.1 <- train(build.1.LOCATION ~ ., 
                   sample.build.1.loc,
                   method = "C5.0", tuneGrid=grid, metric=metric1,
                   tuneLength=2,
                   trControl = fitControl,
                   preProcess = c("zv", "medianImpute"))

# training results c50
c50.loc.1

# prediction
pred.c50.loc.1 <- predict(c50.loc.1, build.1.loc.v)
summary(pred.c50.loc.1)
str(build.1.loc.v[,1:10])

results.c50.loc.1 <- postResample(pred.c50.loc.1, build.1.loc.v$build.1.v.LOCATION)
results.c50.loc.1

accuracy.c50.1 <- results.c50.loc.1[1]
accuracy.c50.1
kappa.c50.1 <- results.c50.loc.1[2]
kappa.c50.1

conf.matrix.c50.loc.1 <- table(pred.c50.loc.1, build.1.loc.v$build.1.v.LOCATION)
conf.matrix.c50.loc.1[1:20,1:20]

#############
# BUILDING 2
#############

# Predict LOCATION (BUILDINGID, FLOOR, SPACEID)

# kNN -------------------------------------------------------------
set.seed(998)

# The first (previous) run did not specify tunegrid

knnGrid <- expand.grid(.k=c(1:4))  # Last run
knn.loc.2 <- train(build.2.LOCATION ~ ., 
                   sample.build.2.loc,
                   method = "knn", metric=metric1,
                   tuneLength = 7, tuneGrid = knnGrid,
                   trControl = fitControl,
                   preProcess = c("zv", "medianImpute"))
knn.loc.2

pred.knn.loc.2 <- predict(knn.loc.2, build.2.loc.v)
summary(pred.knn.loc.2)

results.knn.loc.2 <- postResample(pred.knn.loc.2, build.2.loc.v$build.2.v.LOCATION)
results.knn.loc.2
accuracy.knn.2 <- results.knn.loc.2[1]
accuracy.knn.2
kappa.knn.2 <- results.knn.loc.2[2]
kappa.knn.2

conf.matrix.knn.loc.2 <- table(pred.knn.loc.2, build.2.loc.v$build.2.v.LOCATION)
conf.matrix.knn.loc.2[1:20,1:20]

# Random Forest ---------------------------------------------------
set.seed(998)

# dataframe for manual tuning of mtry
# rfGrid <- expand.grid(mtry=c(1,2,3,5))  # 1st Run: Fitting mtry = 5 on full training set
# rfGrid <- expand.grid(mtry=c(2,5))
rfGrid <- expand.grid(mtry=c(16, 32, 48)) # Last run 
rfor.loc.2 <- train(build.2.LOCATION ~ ., 
                    sample.build.2.loc,
                    method = "rf", tuneGrid=rfGrid, metric=metric1,
                    trControl = fitControl,
                    preProcess = c("zv", "medianImpute"))

# training results rfor
rfor.loc.2

# prediction
pred.rfor.loc.2 <- predict(rfor.loc.2, build.2.loc.v)
summary(pred.rfor.loc.2)
str(build.2.loc.v[,1:10])

results.rfor.loc.2 <- postResample(pred.rfor.loc.2, build.2.loc.v$build.2.v.LOCATION)
results.rfor.loc.2

accuracy.rfor.2 <- results.rfor.loc.2[1]
accuracy.rfor.2
kappa.rfor.2 <- results.rfor.loc.2[2]
kappa.rfor.2

conf.matrix.rfor.loc.2 <- table(pred.rfor.loc.2, build.2.loc.v$build.2.v.LOCATION)
conf.matrix.rfor.loc.2[1:20,1:20]


# C5.0 ------------------------------------------------------------

set.seed(998)

grid <- expand.grid( .winnow = FALSE, .trials=c(48, 96), .model = "rules" )  # Last run
c50.loc.2 <- train(build.2.LOCATION ~ ., 
                   sample.build.2.loc,
                   method = "C5.0", tuneGrid=grid, metric=metric1,
                   tuneLength=2, 
                   trControl = fitControl,
                   preProcess = c("zv", "medianImpute"))

# training results c50
c50.loc.2

# prediction
pred.c50.loc.2 <- predict(c50.loc.2, build.2.loc.v)
summary(pred.c50.loc.2)
str(build.2.loc.v[,1:10])

results.c50.loc.2 <- postResample(pred.c50.loc.2, build.2.loc.v$build.2.v.LOCATION)
results.c50.loc.2

accuracy.c50.2 <- results.c50.loc.2[1]
accuracy.c50.2
kappa.c50.2 <- results.c50.loc.2[2]
kappa.c50.2

conf.matrix.c50.loc.2 <- table(pred.c50.loc.2, build.2.loc.v$build.2.v.LOCATION)
conf.matrix.c50.loc.2[1:20,1:20]



###############################
# Accuracy & Kappa for LOCATION 
###############################

# Weighted Average Accuracy & Kappa of each model for the LOCATION

# kNN ----
knn.Accuracy <- ( accuracy.knn.0 * nlevels(knn.loc.0) + accuracy.knn.1 * nlevels(knn.loc.1) +
                  accuracy.knn.2 * nlevels(knn.loc.2)) / 
                (nlevels(knn.loc.0) + nlevels(knn.loc.1) + nlevels(knn.loc.2))
knn.Accuracy
# Accuracy 
# 0.574934 

knn.Kappa <- ( kappa.knn.0 * nlevels(knn.loc.0) + kappa.knn.1 * nlevels(knn.loc.1) +
                kappa.knn.2 * nlevels(knn.loc.2)) / 
               (nlevels(knn.loc.0) + nlevels(knn.loc.1) + nlevels(knn.loc.2))
knn.Kappa
# Kappa 
# 0.572728 

# Random Forest ----
rfor.Accuracy <- ( accuracy.rfor.0 * nlevels(rfor.loc.0) + accuracy.rfor.1 * nlevels(rfor.loc.1) +
                    accuracy.rfor.2 * nlevels(rfor.loc.2)) / 
  (nlevels(rfor.loc.0) + nlevels(rfor.loc.1) + nlevels(rfor.loc.2))
rfor.Accuracy
# Accuracy 
# 0.7013552 

rfor.Kappa <- ( kappa.rfor.0 * nlevels(rfor.loc.0) + kappa.rfor.1 * nlevels(rfor.loc.1) +
                 kappa.rfor.2 * nlevels(rfor.loc.2)) / 
  (nlevels(rfor.loc.0) + nlevels(rfor.loc.1) + nlevels(rfor.loc.2))
# rfor.Kappa
# Kappa 
# 0.6997843 

# C5.0 ---
c50.Accuracy <- ( accuracy.c50.0 * nlevels(c50.loc.0) + accuracy.c50.1 * nlevels(c50.loc.1) +
                     accuracy.c50.2 * nlevels(c50.loc.2)) / 
  (nlevels(c50.loc.0) + nlevels(c50.loc.1) + nlevels(c50.loc.2))
c50.Accuracy
# Accuracy 
# 0.6016245

c50.Kappa <- ( kappa.c50.0 * nlevels(c50.loc.0) + kappa.c50.1 * nlevels(c50.loc.1) +
                  kappa.c50.2 * nlevels(c50.loc.2)) / 
  (nlevels(c50.loc.0) + nlevels(c50.loc.1) + nlevels(c50.loc.2))
c50.Kappa
# Kappa 
# 0.0.5995335
kappa.c50.0
kappa.c50.1
kappa.c50.2
nlevels(c50.loc.0)
nlevels(c50.loc.1)
nlevels(c50.loc.2)

CHECKLEVEL <- (nlevels(c50.loc.0) + nlevels(c50.loc.1) + nlevels(c50.loc.2))
CHECKLEVEL

###############################
# RESAMPLE & VISUALIZATION
###############################

set.seed(998)

# PLOT ACCURACY
loc.accuracy <- data.frame(metrics = c("KNN.0", "RFOREST.0", "C50.0", 
                                       "KNN.1", "RFOREST.1", "C50.1",
                                       "KNN.2", "RFOREST.2", "C50.2"),
                           values =  c(accuracy.knn.0, accuracy.rfor.0, accuracy.c50.0,
                                       accuracy.knn.1, accuracy.rfor.1, accuracy.c50.1,
                                       accuracy.knn.2, accuracy.rfor.2, accuracy.c50.2))

loc.accuracy$metrics <- as.character(loc.accuracy$metrics)
loc.accuracy$metrics <- factor(loc.accuracy$metrics, levels=unique(loc.accuracy$metrics))

# Plot unweighted
# ggplot() + stat_density(aes(x = loc.accuracy$values) )
                                                        
#loc plots
f <- loc.accuracy %>% 
  ggplot(aes(x = loc.accuracy$metrics, y = loc.accuracy$values)) + 
  geom_density(aes(x = loc.accuracy$metrics, weight = loc.accuracy$values / sum(loc.accuracy$values )), color = "green") +
  geom_density(aes(x = loc.accuracy$metrics), color = "blue") +
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), colour = "red") +
  coord_flip() +
  labs(x = "RESULTS for each BUILDING 0 1 2",
       y = "ACCURACY",
       title = "LOCATION") +
  theme_light() +
#  scale_fill_brewer(palette = "BuPu") +
   scale_fill_brewer(palette = "GnBu") +
  
    theme(legend.position="none")

# Accuracy bar plots
lat.long.plots <- grid.arrange(f, ncol = 1)

# PLOT KAPPA
loc.kappa <- data.frame(metrics = c("KNN.0", "RFOREST.0", "C50.0", 
                                       "KNN.1", "RFOREST.1", "C50.1",
                                       "KNN.2", "RFOREST.2", "C50.2"),
                           values =  c(kappa.knn.0, kappa.rfor.0, kappa.c50.0,
                                       kappa.knn.1, kappa.rfor.1, kappa.c50.1,
                                       kappa.knn.2, kappa.rfor.2, kappa.c50.2))

loc.kappa$metrics <- as.character(loc.kappa$metrics)
loc.kappa$metrics <- factor(loc.kappa$metrics, levels=unique(loc.kappa$metrics))

# Plot unweighted
# ggplot() + stat_density(aes(x = loc.kappa$values, xlab="KAPPA") )

#loc plots
k <- loc.kappa %>% 
  ggplot(aes(x = loc.kappa$metrics, y = loc.kappa$values)) + 
  geom_density(aes(x = loc.kappa$metrics, weight = loc.kappa$values / sum(loc.kappa$values )), color = "green") +
  geom_density(aes(x = loc.kappa$metrics), color = "red") +
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), colour = "green") +
  coord_flip() +
  labs(x = "RESULTS for each BUILDING 0 1 2",
       y = "KAPPA",
       title = "LOCATION") +
  theme_light() +
  scale_fill_brewer(palette = "OrRd") +
  theme(legend.position="none")

# Kappa plots
lat.long.plots <- grid.arrange(k, ncol = 1)
lat.long.plots

#All plots in one
lat.long.plots <- grid.arrange(f, k, ncol = 2)


# PLOT LOCATION ACCURACY
loc.accuracy <- data.frame(metrics = c("KNN", "RFOREST", "C50"),
                           values =  c(knn.Accuracy, rfor.Accuracy, c50.Accuracy))

loc.accuracy$metrics <- as.character(loc.accuracy$metrics)
loc.accuracy$metrics <- factor(loc.accuracy$metrics, levels=unique(loc.accuracy$metrics))

# Plot unweighted
# ggplot() + stat_density(aes(x = loc.accuracy$values) )

#loc plots
f <- loc.accuracy %>% 
  ggplot(aes(x = loc.accuracy$metrics, y = loc.accuracy$values)) + 
  geom_density(aes(x = loc.accuracy$metrics, weight = loc.accuracy$values / sum(loc.accuracy$values )), color = "green") +
  geom_density(aes(x = loc.accuracy$metrics), color = "blue") +
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), colour = "red") +
  coord_flip() +
  labs(x = "RESULTS for the LOCATION",
       y = "ACCURACY",
       title = "LOCATION") +
  theme_light() +
  #  scale_fill_brewer(palette = "BuPu") +
  scale_fill_brewer(palette = "BuGn") +
  
  theme(legend.position="none")

# Accuracy bar plots
lat.long.plots <- grid.arrange(f, ncol = 1)

# PLOT KAPPA for LOCATION
loc.kappa <- data.frame(metrics = c("KNN", "RFOREST", "C50"),
                        values =  c(knn.Kappa, rfor.Kappa, c50.Kappa))

loc.kappa$metrics <- as.character(loc.kappa$metrics)
loc.kappa$metrics <- factor(loc.kappa$metrics, levels=unique(loc.kappa$metrics))

# Plot unweighted
# ggplot() + stat_density(aes(x = loc.kappa$values, xlab="KAPPA") )

#loc plots
k <- loc.kappa %>% 
  ggplot(aes(x = loc.kappa$metrics, y = loc.kappa$values)) + 
  geom_density(aes(x = loc.kappa$metrics, weight = loc.kappa$values / sum(loc.kappa$values )), color = "green") +
  geom_density(aes(x = loc.kappa$metrics), color = "red") +
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), colour = "green") +
  coord_flip() +
  labs(x = "RESULTS for the LOCATION",
       y = "KAPPA",
       title = "LOCATION") +
  theme_light() +
  scale_fill_brewer(palette = "BuPu") +
  theme(legend.position="none")

# Kappa plots
lat.long.plots <- grid.arrange(k, ncol = 1)
lat.long.plots

#All plots in one
lat.long.plots <- grid.arrange(f, k, ncol = 2)


###############################
# Create Output File
################################ 

# -- Create csv file with results from RF selected as the best model --

# Add prediction of LOCATION and correponding BUILDINGID, FLOOR, SPACEID
OutputBLDS0 <-  build.0.v
OutputBLDS0$Actual_LOCATION0 <-  build.0.loc.v[,1]
OutputBLDS0$Predicted_LOCATION0 <- pred.rfor.loc.0

OutputBLDS1 <-  build.1.v
OutputBLDS1$Actual_LOCATION1 <-  build.1.loc.v[,1]
OutputBLDS1$Predicted_LOCATION1 <- pred.rfor.loc.1

OutputBLDS2 <-  build.2.v
OutputBLDS2$Actual_LOCATION2 <-  build.2.loc.v[,1]
OutputBLDS2$Predicted_LOCATION2 <- pred.rfor.loc.2


# Create a csv file with Validation / Testing Data Set 
# Predicted LOCATION and corresponding BUILDINGID, FLOOR, SPACEID
write.csv(OutputBLDS0, file="C3T3_BLDS0-OUTPUT_0801v1.csv", row.names = TRUE)
write.csv(OutputBLDS1, file="C3T3_BLDS1-OUTPUT_0801v1.csv", row.names = TRUE)
write.csv(OutputBLDS2, file="C3T3_BDLS2-OUTPUT_0801v1.csv", row.names = TRUE)

