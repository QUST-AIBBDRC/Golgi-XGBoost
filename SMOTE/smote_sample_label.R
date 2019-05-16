library(DMwR)
require(methods)
#require(xgboost)
setwd("E:/R_test")
data_train = read.csv("train_data_select.csv",header = F)
data_train$V1=factor(data_train$V1)
train_SMOTEdata <- SMOTE(V1~.,data_train,perc.over =200,perc.under=150)
jishu<-table(train_SMOTEdata$V1)


write.csv(train_SMOTEdata,file='train_smote_select200.csv')


data_test<- read.csv("test_data_select.csv",header = F)
data_test$V1=factor(data_test$V1)
test_SMOTEdata <- SMOTE(V1~.,data_test,perc.over =300,perc.under=135)
test_<-table(test_SMOTEdata$V1)
write.csv(test_SMOTEdata,file='test_smote_select200.csv')