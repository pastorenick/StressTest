
library('party')
 
library('e1071')
library('RWeka')

originalDF = read.csv('metrics.csv')

# Check if missing data
is.na.data.frame <- function(obj){
                        sapply(obj,FUN = function(x) all(is.na(x)))
                        }
                            
isnt.finite.data.frame <- function(obj){
                        sapply(obj,FUN = function(x) all(!is.finite(x)))
                        }
           
# List of bad fitted profiles (the sleeping region is bad constrained)
badFit_DF = read.csv('BadFit.csv')
        
# Remove useless columns for the learning algorithms
DF <- originalDF                         
DF$X <- NULL  
DF$ID <- NULL    
                            
# Extracting only DF rows with well fitted sleeping regions
DF_badfit <- DF[badFit_DF$BadFit == 'True', ]
DF_notfit <- DF[badFit_DF$BadFit == 'nan', ]
nDF <- DF[badFit_DF$BadFit == 'False', ]
                               
# Creating training and test datasets
set.seed(21351)
ratioTrainTest = 5.
splitSample = sample(rep(1:ratioTrainTest, length(nDF$Gini_diff)/ratioTrainTest)) 
nDF_test <- nDF[splitSample == 1,]
nDF <- nDF[splitSample != 1,] 
# nDF <- nDF
# nDF_test <- nDF                               
                            
# Removing potential empty levels in both test and training datasets
nDF$label <- factor(nDF$label)
nDF_test$label <- factor(nDF_test$label)

length(nDF$label)
length(nDF_test$label)
nDF_test$label

# Creating 10-fold training set
train <- createFolds(nDF$label, k=10)

C45Fit <- train(label ~ ., method = "J48", data = nDF,
    tuneLength = 5,
    trControl = trainControl(
        method = "cv", indexOut = train))
C45Fit
C45Fit$finalModel

# Check resulting predictions
predictedC45 <- predict(C45Fit, nDF_test)
check <- rep(0, nrow(nDF_test))
for(i in 1:nrow(nDF_test)){
    if(c(nDF_test$label[i])-c(predictedC45[i]) == 0) check[i] <- 1
}

fracMild = sum(check[nDF_test$label == 'mild'])/length(check[nDF_test$label == 'mild'])
fracMild  
fracModerate = sum(check[nDF_test$label == 'moderate'])/length(check[nDF_test$label == 'moderate'])
fracModerate
fracSevere = sum(check[nDF_test$label == 'severe'])/length(check[nDF_test$label == 'severe'])
fracSevere


# PART (Rule-based classifier)
rulesFit <- train(label ~ ., method = "PART", data = nDF,
  tuneLength = 5,
  trControl = trainControl(
    method = "cv", indexOut = train))
rulesFit
rulesFit$finalModel

# Check resulting predictions
predictedRules <- predict(rulesFit, nDF_test)
check <- rep(0, nrow(nDF_test))
for(i in 1:nrow(nDF_test)){
    if(c(nDF_test$label[i])-c(predictedRules[i]) == 0) check[i] <- 1
}

fracMild = sum(check[nDF_test$label == 'mild'])/length(check[nDF_test$label == 'mild'])
fracMild  
fracModerate = sum(check[nDF_test$label == 'moderate'])/length(check[nDF_test$label == 'moderate'])
fracModerate
fracSevere = sum(check[nDF_test$label == 'severe'])/length(check[nDF_test$label == 'severe'])
fracSevere

# Linear Support Vector Machines
svmFit <- train(label ~., method = "svmLinear", data = nDF,
    tuneLength = 5,
    trControl = trainControl(
        method = "cv", indexOut = train))
svmFit
svmFit$finalModel

# Check resulting predictions
predictedSVM <- predict(svmFit, nDF_test)
check <- rep(0, nrow(nDF_test))
for(i in 1:nrow(nDF_test)){
    if(c(nDF_test$label[i])-c(predictedSVM[i]) == 0) check[i] <- 1
}

fracMild = sum(check[nDF_test$label == 'mild'])/length(check[nDF_test$label == 'mild'])
fracMild  
fracModerate = sum(check[nDF_test$label == 'moderate'])/length(check[nDF_test$label == 'moderate'])
fracModerate
fracSevere = sum(check[nDF_test$label == 'severe'])/length(check[nDF_test$label == 'severe'])
fracSevere

# Artificial Neural Network
nnetFit <- train(label ~ ., method = "nnet", data = nDF,
    tuneLength = 5,
    trControl = trainControl(
        method = "cv"
        , indexOut = train))
nnetFit
nnetFit$finalModel

# Check resulting predictions
predictedNN <- predict(nnetFit, nDF_test)
check <- rep(0, nrow(nDF_test))
for(i in 1:nrow(nDF_test)){
    if(c(nDF_test$label[i])-c(predictedNN[i]) == 0) check[i] <- 1
}

fracMild = sum(check[nDF_test$label == 'mild'])/length(check[nDF_test$label == 'mild'])
fracMild  
fracModerate = sum(check[nDF_test$label == 'moderate'])/length(check[nDF_test$label == 'moderate'])
fracModerate
fracSevere = sum(check[nDF_test$label == 'severe'])/length(check[nDF_test$label == 'severe'])
fracSevere

# Random Forest radial
randomForestFit_SVMradial <- train(label ~ ., #method = "rf", data = nDF,
                         method = "svmRadial", data = nDF,
    tuneLength = 5,
    trControl = trainControl(
        method = "cv", indexOut = train))
randomForestFit_SVMradial
randomForestFit_SVMradial$finalModel

# Check resulting predictions
predictedRF_radial <- predict(randomForestFit_SVMradial, nDF_test)
check <- rep(0, nrow(nDF_test))
for(i in 1:nrow(nDF_test)){
    if(c(nDF_test$label[i])-c(predictedRF_radial[i]) == 0) check[i] <- 1
}

fracMild = sum(check[nDF_test$label == 'mild'])/length(check[nDF_test$label == 'mild'])
fracMild  
fracModerate = sum(check[nDF_test$label == 'moderate'])/length(check[nDF_test$label == 'moderate'])
fracModerate
fracSevere = sum(check[nDF_test$label == 'severe'])/length(check[nDF_test$label == 'severe'])
fracSevere

# Random Forest linear
randomForestFit_SVMlinear <- train(label ~ ., #method = "rf", data = nDF,
                         method = "svmLinear", data = nDF,
    tuneLength = 5,
    trControl = trainControl(
        method = "cv", indexOut = train))
randomForestFit_SVMlinear
randomForestFit_SVMlinear$finalModel

# Check resulting predictions
predictedRF_linear <- predict(randomForestFit_SVMlinear, nDF_test)
check <- rep(0, nrow(nDF_test))
for(i in 1:nrow(nDF_test)){
    if(c(nDF_test$label[i])-c(predictedRF_linear[i]) == 0) check[i] <- 1
}

fracMild = sum(check[nDF_test$label == 'mild'])/length(check[nDF_test$label == 'mild'])
fracMild  
fracModerate = sum(check[nDF_test$label == 'moderate'])/length(check[nDF_test$label == 'moderate'])
fracModerate
fracSevere = sum(check[nDF_test$label == 'severe'])/length(check[nDF_test$label == 'severe'])
fracSevere

# Random Forest standard
randomForestFit_rf <- train(label ~ ., #method = "rf", data = nDF,
                         method = "rf", data = nDF,
    tuneLength = 5,
    trControl = trainControl(
        method = "cv", indexOut = train))
randomForestFit_rf
randomForestFit_rf$finalModel

# Check resulting predictions
predictedRF_rf <- predict(randomForestFit_rf, nDF_test)
check <- rep(0, nrow(nDF_test))
for(i in 1:nrow(nDF_test)){
    if(c(nDF_test$label[i])-c(predictedRF_rf[i]) == 0) check[i] <- 1
}

fracMild = sum(check[nDF_test$label == 'mild'])/length(check[nDF_test$label == 'mild'])
fracMild  
fracModerate = sum(check[nDF_test$label == 'moderate'])/length(check[nDF_test$label == 'moderate'])
fracModerate
fracSevere = sum(check[nDF_test$label == 'severe'])/length(check[nDF_test$label == 'severe'])
fracSevere

# Comparing predictions for single cases
nDF_test$best_voted <- 99
for (index in 1:length(nDF_test$Gini_t)) {
    check <- c(predict(C45Fit, nDF_test[index,]),
#                predict(svmFit, nDF_test[index,]), 
               predict(rulesFit, nDF_test[index,]), 
               predict(nnetFit, nDF_test[index,]),
                predict(randomForestFit_SVMradial, nDF_test[index,]),
                predict(randomForestFit_SVMlinear, nDF_test[index,])
#                 predict(randomForestFit_rf, nDF_test[index,])
              )
    best_voted <- names(sort(table(check),decreasing=TRUE))[1]
    nDF_test$best_voted[index] <- best_voted
}

diff = as.numeric(nDF_test$best_voted)-c(nDF_test$label)
(length(which(diff!=0))/ length(nDF_test$Gini_t))*100

# ...work in progress
