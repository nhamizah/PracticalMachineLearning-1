


library(caret);
library(ggplot2);

naValues <- c("NA", "#DIV/0!")

# PART 1 (Data Input):read csv file
pmlTrainingData = read.csv("pml-training.csv", header=TRUE, na.strings=naValues, stringsAsFactors=FALSE, row.names=1)  

# PART 2 ( Basic Preprocessing of Data)
#Detect columns with near zero variance, as candidates to discard from training model
nsv <- nearZeroVar(pmlTrainingData,saveMetrics=TRUE)

pmlTrainingData2  <-   pmlTrainingData[, which(nsv$nzv==FALSE)]

#convert user_name to factor variables
pmlTrainingData2$user_name  <-  as.factor(pmlTrainingData2$user_name)

#convert cvtd_timestamp to friendly date format
pmlTrainingData2$cvtd_timestamp <- as.POSIXlt(pmlTrainingData2$cvtd_timestamp, format="%d/%m/%Y %H:%M") 

# Explore the variable, raw_timestamp_part_1
#  head(as.Date(as.POSIXlt(pmlTrainingData2$raw_timestamp_part_1, tz="", origin = "2011-02-12")))
#  hist(pmlTrainingData2$raw_timestamp_part_1, labels=TRUE) 

# From the histogram of raw_timestamp_part_1 falls into 1 of 4 total bins, 
# which may correspond to each of the 4 wearable devices - 
# the belt, the arm band, the glove, and the dumbbell. So,
# convert pmlTrainingData2$raw_timestamp_part_1  values into one of the four bins 
# as a factor variable

# 
# pmlTrainingData2$raw_timestamp_part_1 <- data.frame(pmlTrainingData2$raw_timestamp_part_1, 
#                                                     bin=cut(pmlTrainingData2$raw_timestamp_part_1,
#                                                     breaks=4, labels=FALSE))$bin 
# 
# pmlTrainingData2$raw_timestamp_part_1 <- as.factor(pmlTrainingData2$raw_timestamp_part_1)
# 

# Explore the variable, raw_timestamp_part_2

# > summary(pmlTrainingData2$raw_timestamp_part_2)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 294  252900  496400  500700  751900  998800 

# The raw_timestamp_part_2 most likely is a elapsed time value in ms, 
# so let's convert the values  to minutes for a better interpretation. 
# Divid the values by (1000 ms/s) * (60s/min) = 60,000 ms/min

pmlTrainingData2$raw_timestamp_part_2 <- pmlTrainingData2$raw_timestamp_part_2 / 60000

# REmove columns with excessive number of NA values

pmlTrainingData3  <- pmlTrainingData2[, -which(colnames(pmlTrainingData2)=="classe")]
pmlTrainingData3  <- pmlTrainingData3[, -which(colnames(pmlTrainingData3)=="user_name")]
pmlTrainingData3  <- pmlTrainingData3[, -which(colnames(pmlTrainingData3)=="cvtd_timestamp")]
pmlTrainingData3  <- pmlTrainingData3[, -which(colSums(is.na(pmlTrainingData3))/nrow(pmlTrainingData3) > 0)]


# Find if some columns are linear combinations of other columns within data

comboInfo <- findLinearCombos(pmlTrainingData3)
# $linearCombos
# list()
# 
# $remove
# NULL

# We find that the pmlTrainingData3 object does not contain columns that are linear combinations of other columns


# Now try to find and remove highly correlated descriptors with a threshhold above 0.75

descrCor <- cor(pmlTrainingData3)
summary(descrCor[upper.tri(descrCor)])

highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)

pmlTrainingData4 <- pmlTrainingData3[,-highlyCorDescr]

#PART 4:  Train model with PCA pre-processing and a k-nearest neighbor regression model
# USE PCA (Principal Component Analysis) to reduce number of covariates


                 #### ignore:  pcaTrainingDataCov  <- preProcess(pmlTrainingData4, method="pca") #yields 19 principal component variables

# define training control ; use k-fold cross validation with 10 data slices
pmlTrainingData5 <- cbind(pmlTrainingData4,pmlTrainingData$classe)
names(pmlTrainingData5)[names(pmlTrainingData5) == 'pmlTrainingData$classe'] <- 'classe'

train_control <- trainControl(method="cv", number=10)

modelFit <- train(classe ~., data=pmlTrainingData5, method = "knn", preProcess=c("pca"), 
                 trControl = trainControl(method = "cv", number=10))

# make predictions from the training data
predictions <- predict(modelFit, pmlTrainingData)


## Part 5. Now use the Training Prediction Model, modelFit, on the TEST set


pmlTestingData = read.csv("pml-testing.csv", header=TRUE, na.strings=naValues, stringsAsFactors=FALSE, row.names=1)  

testPC <- predict(modelFit, pmlTestingData)

confusionMatrix(pmlTestingData, predict(modelFit,testPC))
