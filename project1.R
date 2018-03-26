library(dplyr)
library(data.table)
library(foreach)
library(glmnet)

# error measure
mse = function(y_hat, y) {
  mse = mean((y - y_hat)^2)
  return(mse)
}

# Read training data
train <- fread("/home/maninya/MIS/ML/BlogFeedback/blogData_train.csv")

# Read test data, 2 test files for each of Feb and March
testf1 <- fread("/home/maninya/MIS/ML/BlogFeedback/blogData_test-2012.02.02.00_00.csv")
testf2 <- fread("/home/maninya/MIS/ML/BlogFeedback/blogData_test-2012.02.29.00_00.csv")
testm1 <- fread("/home/maninya/MIS/ML/BlogFeedback/blogData_test-2012.03.01.00_00.csv")
testm2 <- fread("/home/maninya/MIS/ML/BlogFeedback/blogData_test-2012.03.06.00_00.csv")


# log-transform column 281 into a new column
train[, V281_log := log(1 + V281)]
testf1[, V281_log := log(1+V281)]
testf2[, V281_log := log(1+V281)]
testm1[, V281_log := log(1+V281)]
testm2[, V281_log := log(1+V281)]

# Select basic features in train data
basic_log = subset(train, select = c(51:60, 282))
basic_non = subset(train, select = c(51:60, 281))

# Select basic features in test data
testf1basic_log = subset(testf1, select = colnames(basic_log))
testf1basic_non = subset(testf1, select = colnames(basic_non))

testf2basic_log = subset(testf2, select = colnames(basic_log))
testf2basic_non = subset(testf2, select = colnames(basic_non))

testm1basic_log = subset(testm1, select = colnames(basic_log))
testm1basic_non = subset(testm1, select = colnames(basic_non))

testm2basic_log = subset(testm2, select = colnames(basic_log))
testm2basic_non = subset(testm2, select = colnames(basic_non))


# Select textual features in train data
textual_log = subset(train, select = c(63:262, 282))
textual_non = subset(train, select = c(63:262, 281))

# Select textual features in test data
testf1textual_log = subset(testf1, select = colnames(textual_log))
testf1textual_non = subset(testf1, select = colnames(textual_non))

testf2textual_log = subset(testf2, select = colnames(textual_log))
testf2textual_non = subset(testf2, select = colnames(textual_non))

testm1textual_log = subset(testm1, select = colnames(textual_log))
testm1textual_non = subset(testm1, select = colnames(textual_non))

testm2textual_log = subset(testm2, select = colnames(textual_log))
testm2textual_non = subset(testm2, select = colnames(textual_non))


# Perform Linear Regression with basic features
libasic <- lm(formula=V281_log~.+1, data=basic_log)
summary(libasic)
mselibasic <- mean(libasic$residuals^2)

# Test for Feb data 1
predf1libasic = predict(libasic, testf1basic_log)
mse_testf1libasic <- mse(predf1libasic, testf1basic_log$V281_log)
compare_f1libasic = paste(exp(predf1libasic)-1, exp(testf1basic_log$V281_log)-1)

# Test for Feb data 2
predf2libasic = predict(libasic, testf2basic_log)
mse_testf2libasic <- mse(predf2libasic, testf2basic_log$V281_log)
compare_f2libasic = paste(exp(predf2libasic)-1, exp(testf2basic_log$V281_log)-1)

# Test for Mar data 1
predm1libasic = predict(libasic, testm1basic_log)
mse_testm1libasic <- mse(predm1libasic, testm1basic_log$V281_log)
compare_m1libasic = paste(exp(predm1libasic)-1, exp(testm1basic_log$V281_log)-1)

# Test for Mar data 2
predm2libasic = predict(libasic, testm2basic_log)
mse_testm2libasic <- mse(predm2libasic, testm2basic_log$V281_log)
compare_m2libasic = paste(exp(predm2libasic)-1, exp(testm2basic_log$V281_log)-1)

# Perform Logistic Regression on basic features

# Preprocess: Add levels for the column 281
train_5000 = train[1:5000]
train_5000$class[train_5000$V281 == 0] = 0
train_5000$class[train_5000$V281 == 1] = 1
train_5000$class[(train_5000$V281 > 1 & train_5000$V281 <= 5)] = 2
train_5000$class[(train_5000$V281 > 5 & train_5000$V281 <= 10)] = 3
train_5000$class[train_5000$V281 > 10] = 4

#Select the basic features
basic_class = subset(train_5000, select = c(51:60,283))

# Create the model
train_x_sparse = model.Matrix(class ~ ., data = basic_class, sparse = T)
lobasic <- glmnet(train_x_sparse, basic_class$class, family="multinomial", type.multinomial = "ungrouped")
plot(lobasic, xvar="lambda")

# Test for Feb data 1
testf1$class[testf1$V281 == 0] = 0
testf1$class[testf1$V281 == 1] = 1
testf1$class[(testf1$V281 > 1 & testf1$V281 <= 5)] = 2
testf1$class[(testf1$V281 > 5  & testf1$V281 <= 10)] = 3
testf1$class[testf1$V281 > 10] = 4
testf1basic_class = subset(testf1, select = c(51:60, 283))

predf1lobasic = predict(lobasic, newx=model.Matrix(class~., testf1basic_class), type="class")

# Confusion matrix
basicf1cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testf1basic_class)) {
  pred  = as.numeric(predf1lobasic[i]) + 1
  label = testf1basic_class[i]$class + 1
  basicf1cm[label, pred] <- basicf1cm[label, pred] + 1
}

# Test for Feb data 2
testf2$class[testf2$V281 == 0] = 0
testf2$class[testf2$V281 == 1] = 1
testf2$class[(testf2$V281 > 1 & testf2$V281 <= 5)] = 2
testf2$class[(testf2$V281 > 5  & testf2$V281 <= 10)] = 3
testf2$class[testf2$V281 > 10] = 4
testf2basic_class = subset(testf2, select = c(51:60, 283))

predf2lobasic = predict(lobasic, newx=model.Matrix(class~., testf2basic_class), type="class")

# Confusion matrix
basicf2cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testf2basic_class)) {
  pred  = as.numeric(predf2lobasic[i]) + 1
  label = testf2basic_class[i]$class + 1
  basicf2cm[label, pred] <- basicf2cm[label, pred] + 1
}


# Test for Mar data 1
testm1$class[testm1$V281 == 0] = 0
testm1$class[testm1$V281 == 1] = 1
testm1$class[(testm1$V281 > 1 & testm1$V281 <= 5)] = 2
testm1$class[(testm1$V281 > 5  & testm1$V281 <= 10)] = 3
testm1$class[testm1$V281 > 10] = 4
testm1basic_class = subset(testm1, select = c(51:60, 283))

predm1lobasic = predict(lobasic, newx=model.Matrix(class~., testm1basic_class), type="class")

# Confusion matrix
basicm1cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testm1basic_class)) {
  pred  = as.numeric(predm1lobasic[i]) + 1
  label = testm1basic_class[i]$class + 1
  basicm1cm[label, pred] <- basicm1cm[label, pred] + 1
}

# Test for Mar data 2
testm2$class[testm2$V281 == 0] = 0
testm2$class[testm2$V281 == 1] = 1
testm2$class[(testm2$V281 > 1 & testm2$V281 <= 5)] = 2
testm2$class[(testm2$V281 > 5  & testm2$V281 <= 10)] = 3
testm2$class[testm2$V281 > 10] = 4
testm2basic_class = subset(testm2, select = c(51:60, 283))

predm2lobasic = predict(lobasic, newx=model.Matrix(class~., testm2basic_class), type="class")

# Confusion matrix
basicm2cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testm2basic_class)) {
  pred  = as.numeric(predm2lobasic[i]) + 1
  label = testm2basic_class[i]$class + 1
  basicm2cm[label, pred] <- basicm2cm[label, pred] + 1
}



# Perform Linear Regression with textual features
litextual <- lm(formula=V281_log~.+1, data=textual_log)
summary(litextual)
mselitextual <- mean(litextual$residuals^2)

# Test for Feb data 1
predf1litextual = predict(litextual, testf1textual_log)
mse_testf1litextual <- mse(predf1litextual, testf1textual_log$V281_log)
compare_f1litextual = paste(exp(predf1litextual)-1, exp(testf1textual_log$V281_log)-1)

# Test for Feb data 2
predf2litextual = predict(litextual, testf2textual_log)
mse_testf2litextual <- mse(predf2litextual, testf2textual_log$V281_log)
compare_f2litextual = paste(exp(predf2litextual)-1, exp(testf2textual_log$V281_log)-1)

# Test for Mar data 1
predm1litextual = predict(litextual, testm1textual_log)
mse_testm1litextual <- mse(predm1litextual, testm1textual_log$V281_log)
compare_m1litextual = paste(exp(predm1litextual)-1, exp(testm1textual_log$V281_log)-1)

# Test for Mar data 2
predm2litextual = predict(litextual, testm2textual_log)
mse_testm2litextual <- mse(predm2litextual, testm2textual_log$V281_log)
compare_m2litextual = paste(exp(predm2litextual)-1, exp(testm2textual_log$V281_log)-1)


# Perform Logistic Regression on textual features

# Select the textual features
textual_class = subset(train_5000, select = c(63:262,283))


# Create the model
train_x_sparse = model.Matrix(class ~ ., data = textual_class, sparse = T)
lotextual <- glmnet(train_x_sparse, textual_class$class, family="multinomial", type.multinomial = "ungrouped")
plot(lotextual, xvar="lambda")

# Test for Feb data 1
testf1$class[testf1$V281 == 0] = 0
testf1$class[testf1$V281 == 1] = 1
testf1$class[(testf1$V281 > 1 & testf1$V281 <= 5)] = 2
testf1$class[(testf1$V281 > 5  & testf1$V281 <= 10)] = 3
testf1$class[testf1$V281 > 10] = 4
testf1textual_class = subset(testf1, select = c(63:262, 283))

predf1lotextual = predict(lotextual, newx=model.Matrix(class~., testf1textual_class), type="class")

# Confusion matrix
textualf1cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testf1textual_class)) {
  pred  = as.numeric(predf1lotextual[i]) + 1
  label = testf1textual_class[i]$class + 1
  textualf1cm[label, pred] <- textualf1cm[label, pred] + 1
}


# Test for Feb data 2
testf2$class[testf2$V281 == 0] = 0
testf2$class[testf2$V281 == 1] = 1
testf2$class[(testf2$V281 > 1 & testf2$V281 <= 5)] = 2
testf2$class[(testf2$V281 > 5  & testf2$V281 <= 10)] = 3
testf2$class[testf2$V281 > 10] = 4
testf2textual_class = subset(testf2, select = c(63:262, 283))

predf2lotextual = predict(lotextual, newx=model.Matrix(class~., testf2textual_class), type="class")

# Confusion matrix
textualf2cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testf2textual_class)) {
  pred  = as.numeric(predf2lotextual[i]) + 1
  label = testf2textual_class[i]$class + 1
  textualf2cm[label, pred] <- textualf2cm[label, pred] + 1
}


# Test for Mar data 1
testm1$class[testm1$V281 == 0] = 0
testm1$class[testm1$V281 == 1] = 1
testm1$class[(testm1$V281 > 1 & testm1$V281 <= 5)] = 2
testm1$class[(testm1$V281 > 5  & testm1$V281 <= 10)] = 3
testm1$class[testm1$V281 > 10] = 4
testm1textual_class = subset(testm1, select = c(63:262, 283))

predm1lotextual = predict(lotextual, newx=model.Matrix(class~., testm1textual_class), type="class")

# Confusion matrix
textualm1cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testm1textual_class)) {
  pred  = as.numeric(predm1lotextual[i]) + 1
  label = testm1textual_class[i]$class + 1
  textualm1cm[label, pred] <- textualm1cm[label, pred] + 1
}


# Test for Mar data 2
testm2$class[testm2$V281 == 0] = 0
testm2$class[testm2$V281 == 1] = 1
testm2$class[(testm2$V281 > 1 & testm2$V281 <= 5)] = 2
testm2$class[(testm2$V281 > 5  & testm2$V281 <= 10)] = 3
testm2$class[testm2$V281 > 10] = 4
testm2textual_class = subset(testm2, select = c(63:262, 283))

predm2lotextual = predict(lotextual, newx=model.Matrix(class~., testm2textual_class), type="class")

# Confusion matrix
textualm2cm = matrix(data=0, nrow=5, ncol=5)
for (i in 1:nrow(testm2textual_class)) {
  pred  = as.numeric(predm2lotextual[i]) + 1
  label = testm2textual_class[i]$class + 1
  textualm2cm[label, pred] <- textualm2cm[label, pred] + 1
}
