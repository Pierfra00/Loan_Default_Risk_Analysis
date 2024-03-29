#### Data preparation ####
# Load the necessary libraries
library(tidyverse)
library(readr)
library(dplyr)
library(tree)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(boot)
library(ggplot2)

# Load the Loan_default.csv dataset
loan_data <- read_csv("Data/Loan_default.csv")

# Check for missing values
apply(loan_data, 2, function(x) any(is.na(x))) # There are no missing values

# Convert relevant columns to factors
loan_data <- loan_data %>%
  mutate(across(c(Education, EmploymentType, MaritalStatus, LoanPurpose), as.factor))

# Create a new categorical variable for loan default status
loan_data$loan_default_cat <- factor(ifelse(loan_data$Default == 1, "Yes", "No"))

# View the modified dataset
View(loan_data)

#### FITTING THE MODELS ####
#### LOGISTIC REGRESSION ####
set.seed(12321)

# Function to fit a logistic regression model and calculate error rate
fit_logistic <- function(formula, data) {
  model <- glm(formula, data = data, family = "binomial", control = list(maxit = 1000))
  summary(model)
  
  pred <- predict(model, newdata = data, type = "response")
  pred <- ifelse(pred > 0.5, "Default", "Non-default")
  
  error <- sum(pred != data$Default) / nrow(data)
  
  return(list(model = model, pred = pred, error = error))
}


# Fit logistic models
logistic1 <- fit_logistic(Default ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose + HasCoSigner, loan_data)
logistic2 <- fit_logistic(Default ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose, loan_data)

# Create a dummy classifier
dummyClass <- factor(ifelse(loan_data$Default == 1, "Default", "Non-default"), levels = c("Default", "Non-default"))
dummy_error <- mean(dummyClass != loan_data$Default)

# Compare error rates
comparison <- data.frame(Model = c("Logistic Model 1", "Logistic Model 2", "Dummy Classifier"),
                         Error_Rate = c(logistic1$error, logistic2$error, dummy_error))
comparison

# Visualize Error Rates
barplot(comparison$Error_Rate, names.arg = comparison$Model, ylim = c(0, max(comparison$Error_Rate) + 0.1), col = c("lightblue", "lightgreen", "lightcoral"), main = "Error Rate Comparison", xlab = "Models", ylab = "Error Rate")

# Double check
# Check distribution of values in the target variable
table(loan_data$Default)
# Verify the creation of the dummy classifier
table(dummyClass)
# Check convergence warnings for logistic model 1
summary(logistic1$model)
# Check convergence warnings for logistic model 2
summary(logistic2$model)
# Print error rates
cat("Logistic Model 1 Error Rate:", logistic1$error, "\n")
cat("Logistic Model 2 Error Rate:", logistic2$error, "\n")
cat("Dummy Classifier Error Rate:", dummy_error, "\n")
# Visualize the distribution of the target variable
barplot(table(loan_data$Default), main = "Distribution of Target Variable", xlab = "Default", ylab = "Count")
# Visualize predicted values for logistic model 1
barplot(table(logistic1$pred), main = "Distribution of Predicted Values (Logistic Model 1)", xlab = "Predicted Value", ylab = "Count")
# Visualize predicted values for logistic model 2
barplot(table(logistic2$pred), main = "Distribution of Predicted Values (Logistic Model 2)", xlab = "Predicted Value", ylab = "Count")

#All 3 models have high error rate, cross validation will be employed later
# Let's try with other models:

##### DECISION TREE #####
# Convert Default to factor
loan_data$Default_factor <- as.factor(loan_data$Default)

# Fit decision tree using rpart, The attempt with tree was not so representative of the data
mytree_rpart <- rpart(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose + HasCoSigner, data = loan_data, method = "class", cp = 0.001, minsplit = 10, minbucket = 5)

# Plot the original tree
rpart.plot(mytree_rpart, box.palette = "RdBu", shadow.col = "gray90", cex = 0.8, nn = TRUE, type = 3, main = "Original Decision Tree")

## K-fold validation to choose subset and best subset size for the decision tree.
# Define the number of folds (k)
num_folds <- 10

# Define the range of subset sizes to test
subset_sizes <- c(500, 1000, 1500, 2000, 2500) # These numbers have been chosen after several attempts, larger numbers created problems in visualizing the pruned tree

# Create a data frame to store results
results <- data.frame(subset_size = numeric(length(subset_sizes)),
                      average_error = numeric(length(subset_sizes)),
                      std_dev_error = numeric(length(subset_sizes)))

# Iterate through each subset size
for (size_index in seq_along(subset_sizes)) {
  subset_size <- subset_sizes[size_index]
  
  # Create vector to store error rates
  error_rates <- numeric(num_folds)
  
  # Create folds using k-fold cross-validation
  folds <- createFolds(y = loan_data$Default_factor, k = num_folds)
  
  # Iterate through each fold
  for (fold_index in seq_along(folds)) {
    # Get the indices for the current fold
    fold_indices <- folds[[fold_index]]
    
    # Subset the data for the current fold
    fold_data <- loan_data[fold_indices, ]
    
    # Ensure the subset size is reasonable for visualization
    fold_size <- min(subset_size, nrow(fold_data))
    
    # Fit decision tree using rpart on the current fold
    mytree_rpart <- rpart(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose + HasCoSigner, data = fold_data[1:fold_size, ], method = "class", cp = 0.001, minsplit = 10, minbucket = 5)
    
    # Perform cross-validation on decision tree model
    cv_results <- rpart.control(xval = 10, method = "cv", cp = 0.01)
    mytree_cv <- rpart(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + InterestRate + EmploymentType, data = fold_data[1:fold_size, ], method = "class", control = cv_results, cp = 0.001, minsplit = 10, minbucket = 5)
    
    # Prune decision tree to optimal size
    final_tree <- prune(mytree_cv, cp = 0.0001)
    
    # Calculate error rate of pruned tree on the test set
    test_indices <- setdiff(1:nrow(loan_data), fold_indices)
    test_data <- loan_data[test_indices, ]
    predicted_default <- predict(final_tree, test_data, type = "class")
    error_rate <- sum(predicted_default != test_data$Default_factor) / nrow(test_data)
    
    # Store the error rate in the vector
    error_rates[fold_index] <- error_rate
  }
  
  # Store results in the data frame
  results[size_index, "subset_size"] <- subset_size
  results[size_index, "average_error"] <- mean(error_rates)
  results[size_index, "std_dev_error"] <- sd(error_rates)
}

# Print results
print(results)

# Choose the best subset based on lowest average error rate
best_subset <- subset_sizes[which.min(results$average_error)]
cat("\nBest Subset Size:", best_subset, "\n")

## Create a subset of data with results of k-fold validation
set.seed(123)  # Set a different seed for creating the subset
subset_indices <- sample(1:nrow(loan_data), size = best_subset)
subset_data <- loan_data[subset_indices, ]

# Fit decision tree using rpart on the selected subset
mytree_rpart_subset <- rpart(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose + HasCoSigner, data = subset_data, method = "class", cp = 0.001, minsplit = 10, minbucket = 5)

# Plot the tree on the selected subset
rpart.plot(mytree_rpart_subset, box.palette = "RdBu", shadow.col = "gray90", cex = 0.3, nn = TRUE, type = 3, main = "Decision Tree on Selected Subset", fallen.leaves = TRUE, tweak = 0.7)

# Perform cross-validation on decision tree model
cv_results <- rpart.control(xval = 10, method = "cv", cp = 0.01)
mytree_cv <- rpart(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + InterestRate + EmploymentType, data = subset_data, method = "class", control = cv_results, cp = 0.001, minsplit = 10, minbucket = 5)

# Prune decision tree to optimal size
final_tree <- prune(mytree_cv, cp = 0.0001)

# Plot cross-validation error
plotcp(final_tree)
print(final_tree)

# Plot pruned tree with adjusted parameters
prp(final_tree, box.palette = "RdBu", shadow.col = "gray90", cex = 0.5, nn = TRUE, type = 3, main = "Pruned Decision Tree", fallen.leaves = TRUE, tweak = 1.5)

# Print summary of pruned tree
summary(final_tree)

# Calculate error rate of pruned tree
predicted_default <- predict(final_tree, subset_data, type = "class")
error_rate <- sum(predicted_default != subset_data$Default_factor) / nrow(subset_data)
cat("Error Rate on the Selected Subset:", error_rate, "\n")

#### RANDOM FOREST ####
# Fit a random forest model with all predictors, this will take a bit of time to compute
RF <- randomForest(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose + HasCoSigner, data = loan_data)
# Print the random forest model
RF
#OOB error rate is 11,39%

# Plot variable importance
importance(RF)
varImpPlot(RF)

# Fit a random forest model with selected predictors
RF2 <- randomForest(Default_factor ~ Age + Income + CreditScore + InterestRate, data = loan_data)
# Print the random forest model with selected predictors
RF2
summary(RF2)
print(RF2)
# OOB error rate is 11.95%

##### Cross validation of the models #####

# Split the data into training and test set (80% in training set and 20% in test set)
set.seed(1)
ind <- sample(nrow(loan_data), nrow(loan_data)*0.2, replace = FALSE)
train <- loan_data[-ind,]
test <- loan_data[ind,]

# Fit the models on the training set
# Logistic regression with all predictors
logistic1_train <- glm(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose + HasCoSigner, data = train, family = "binomial")
# Logistic regression with significant predictors only
logistic2_train <- glm(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose, data = train, family = "binomial")
# Decision tree
mytree_train <- tree(Default_factor ~ Age + Income + CreditScore + LoanAmount + MonthsEmployed + NumCreditLines + InterestRate + LoanTerm + DTIRatio + Education + EmploymentType + MaritalStatus + HasMortgage + HasDependents + LoanPurpose + HasCoSigner, data = train)

# Random forest
RF_train <- randomForest(Default_factor ~ Age + Income + CreditScore + InterestRate, data = train)

# Calculate reference error rate
fail_proportion <- mean(loan_data$Default_factor == "1")
pass_proportion <- mean(loan_data$Default_factor == "0")
ref_error_rate <- min(fail_proportion, pass_proportion)
ref_error_rate
# 0.1161282

##### Predictions and error rates #######
# Test our models on test set and get the error rates

# Logistic models
# log1
prob_log1<-predict(logistic1_train,newdata = test, type = "response")
pred_log1<-ifelse(prob_log1<0.5, "Default","Non-default")
a<-table(pred_log1,test$Default_factor)
(a[1,1] +a[2,2]) /sum(a)
# 0.883863

# log2
prob_log2<-predict(logistic2_train,newdata = test, type = "response")
pred_log2<-ifelse(prob_log2<0.5, "Default","Non-default")
b<-table(pred_log2,test$Default_factor)
(b[1,1] +b[2,2]) /sum(b)
# 0.8833735

## Other approaches ##
# K-fold approach to estimate the cross validation error rate through 10-fold cross-validation using full dataset
set.seed(1)

# log1
modLogitCV1 <- cv.glm(train, logistic1_train, K = 10)
erLogit1 <- modLogitCV1$delta[1]
erLogit1
# 0.0921678

# log2
modLogitCV2 <- cv.glm(train, logistic2_train, K = 10)
erLogit2 <- modLogitCV2$delta[1]
erLogit2
# 0.09239473

# LOOCV error rate
# log1
#LOOCV.log1<-cv.glm(train, logistic1_train, K = nrow(train))
#er_LOOCV_log1<-LOOCV.log1$delta[1]
#er_LOOCV_log1
# log2
#LOOCV.log2<-cv.glm(train, logistic2_train, K = nrow(train))
#er_LOOCV_log2<-LOOCV.log2$delta[1]
#er_LOOCV_log2

# I chose to use K fold approach to estimate error rates of the logistic models because the LOOCV error rate is computationally really expensive, however it is ready above if it is want to be run

set.seed(1)
# Decision tree
pred_tree <- predict(final_tree, newdata = test, type = "class")
confmTree <- table(pred_tree, test$Default_factor)
erTree <- 1 - sum(diag(confmTree)) / sum(confmTree)
erTree
# 0.1179189

# Random forest
pred_rf <- predict(RF_train, newdata = test, type = "class")
erRF <- 1 - sum(diag(RF_train$confusion)) / sum(RF_train$confusion)
erRF
# 0.1192045

##### ERROR RATES COMPARISON #####
er_comparison <- data.frame(Model = c("Reference er", "Logistic Model1", "Logistic Model2", "Decision Tree", "Random Forest"),
                            Error_Rate = c(ref_error_rate * 100, erLogit1 * 100, erLogit2 * 100, erTree * 100, erRF * 100))

er_comparison

# Logistic Model 1 is the model with the lowest error rate, so Logistic Model 1 provides a more accurate prediction of loan default risk than the other models

# We can also visualize the error rates
ggplot(er_comparison, aes(x = Model, y = Error_Rate)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Error Rates Comparison", x = "Model", y = "Error Rate (%)") +
  theme_minimal()

##### MAE, MSE and RMSE for the models #####

# Convert predicted values to numeric
pred_log1 <- as.numeric(as.character(predict(logistic1_train, newdata = test, type = "response")))
pred_log2 <- as.numeric(as.character(predict(logistic2_train, newdata = test, type = "response")))
pred_tree <- as.numeric(as.character(pred_tree))
pred_rf <- as.numeric(as.character(pred_rf))

# Check the structure of pred_log1 and pred_log2
str(pred_log1)
str(pred_log2)

# Calculate MAE, MSE, and RMSE for Logistic Model 1
mae_log1 <- mean(abs(pred_log1 - as.numeric(as.character(test$Default_factor))))
mse_log1 <- mean((pred_log1 - as.numeric(as.character(test$Default_factor)))^2)
rmse_log1 <- sqrt(mse_log1)

# Calculate MAE, MSE, and RMSE for Logistic Model 2
mae_log2 <- mean(abs(pred_log2 - as.numeric(as.character(test$Default_factor))))
mse_log2 <- mean((pred_log2 - as.numeric(as.character(test$Default_factor)))^2)
rmse_log2 <- sqrt(mse_log2)

# Calculate MAE, MSE, and RMSE for Decision Tree
mae_tree <- mean(abs(pred_tree - as.numeric(as.character(test$Default_factor))))
mse_tree <- mean((pred_tree - as.numeric(as.character(test$Default_factor)))^2)
rmse_tree <- sqrt(mse_tree)

# Calculate MAE, MSE, and RMSE for Random Forest
mae_rf <- mean(abs(pred_rf - as.numeric(as.character(test$Default_factor))))
mse_rf <- mean((pred_rf - as.numeric(as.character(test$Default_factor)))^2)
rmse_rf <- sqrt(mse_rf)

# Create a data frame to store the MAE, MSE, and RMSE for each model
er_comparison_metrics <- data.frame(Model = c("Logistic Model1", "Logistic Model2", "Decision Tree", "Random Forest"),
                                    MAE = c(mae_log1, mae_log2, mae_tree, mae_rf),
                                    MSE = c(mse_log1, mse_log2, mse_tree, mse_rf),
                                    RMSE = c(rmse_log1, rmse_log2, rmse_tree, rmse_rf))

er_comparison_metrics

# Visualize the MAE, MSE and RMSE
ggplot(er_comparison_metrics, aes(x = Model, y = MAE)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "MAE Comparison", x = "Model", y = "MAE") +
  theme_minimal()

ggplot(er_comparison_metrics, aes(x = Model, y = MSE)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "MSE Comparison", x = "Model", y = "MSE") +
  theme_minimal()

ggplot(er_comparison_metrics, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "RMSE Comparison", x = "Model", y = "RMSE") +
  theme_minimal()
