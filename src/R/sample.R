library(Matrix)
source("EMC_IO.r")

train  <- EMC_ReadData("train_data.csv")
test  <- EMC_ReadData("test_data.csv")
dim(train)
dim(test)
labels = as.vector(t(read.csv("train_labels.csv", header= FALSE)))

library(randomForest)
forest_model <- randomForest(as.factor(labels) ~ ., data=trainset, ntree=150)
train_FOREST <- predict(forest_model, train, type="prob")
test_FOREST <- predict(forest_model, test, type="prob")

write.csv(test_FOREST, file ="../submission/submission.csv", row.names=F)
