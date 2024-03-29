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