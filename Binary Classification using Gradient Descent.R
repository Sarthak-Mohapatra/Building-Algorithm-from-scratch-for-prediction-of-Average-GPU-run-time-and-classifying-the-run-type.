##
## title: "Algorithms from scratch using Gradient Descent to predict average GPU Run Time & classify it's run type"
## author: "Sarthak Mohapatra"
## date: "1/29/2020"
##

options(scipen = 999)

##
## Loading the required packages.
##

pacman::p_load(data.table, forecast, leaps, tidyverse, caret, corrplot, glmnet, mlbench, ggplot2, gplots, pivottabler,MASS,
               e1071, fpp2, gains, pROC, knitr, gplots, FNN, RColorBrewer, viridis, cowplot, ggpubr, gridExtra, rlist, d3heatmap)


##
## Importing the dataset from the working directory
##

setwd('D:/Second Semester - MSBA - UTD/Applied Machine Learning/Assignment 1/sgemm_product_dataset')
gpu.df <- read.csv("sgemm_product.csv")
head(gpu.df)

##
## Renaming the last 4 column names
##

names(gpu.df)[15] = "Run1"
names(gpu.df)[16] = "Run2"
names(gpu.df)[17] = "Run3"
names(gpu.df)[18] = "Run4"
head(gpu.df)

##
## Creating a new feature Average. It will contain the average of Run1 through Run4 
##

gpu.df$Average <- (gpu.df$Run1 + gpu.df$Run2 + gpu.df$Run3 + gpu.df$Run4) / 4
head(gpu.df)

##
## Data Partioning
##

set.seed(16)

##
## randomly order the dataset
##

rows <- sample(nrow(gpu.df))
gpu  <- gpu.df[rows, -15:-18]

##
## find rows to split on
##

split <- round(nrow(gpu) * 0.7)
gpu.train.df <- gpu[1:split, ]
gpu.test.df  <- gpu[(split+1):nrow(gpu), ]

##
## confirm the size of the split
##

round(nrow(gpu.train.df)/nrow(gpu), digits = 3)
head(gpu.train.df)
head(gpu.test.df)

##
## Normalizing the dataset.
##

gpu_train_norm         <- gpu.train.df
gpu_test_norm          <- gpu.test.df
gpu_norm_df            <- gpu

norm.values            <- preProcess(gpu.train.df[, 1:15], method=c("center", "scale"))
gpu_train_norm[, 1:15] <- predict(norm.values, gpu.train.df[, 1:15])
gpu_test_norm[, 1:15]  <- predict(norm.values, gpu.test.df[, 1:15])
gpu_norm_df[, 1:15]    <- predict(norm.values, gpu[, 1:15])
new.gpu.norm.df        <- predict(norm.values, gpu)

##
## Creating the feature and target datasets ( X & Y)
##

x_gpu_train <- as.matrix(gpu_train_norm[c(1:14)])
y_gpu_train <- as.matrix(gpu_train_norm[c('Average')])

x_gpu_test  <- as.matrix(gpu_test_norm[c(1:14)])
y_gpu_test  <- as.matrix(gpu_test_norm[c('Average')])

x_gpu_train <- cbind(Intercept=1,x_gpu_train) 
head(x_gpu_train)
head(y_gpu_train)
x_gpu_test  <- cbind(Intercept=1, x_gpu_test)
head(x_gpu_test)
length(y_gpu_train)
length(y_gpu_test)


##
## Converting the problem statement to a binary class problem.  
##

##
## If the average run time of the record is less than Median value, it is given class 0 (low run type) and if greater or equal, it is termed as 1(high run type)
##

median.input <- median(gpu_norm_df$Average)
median.input

x.train.gpu.logit <- x_gpu_train
y.train.gpu.logit <- y_gpu_train

head(y.train.gpu.logit)
y.train.gpu.logit  <- ifelse(y.train.gpu.logit <= median.input, 0, 1)
head(y.train.gpu.logit)


x.test.gpu.logit  <- x_gpu_test
y.test.gpu.logit <- y_gpu_test
head(y.test.gpu.logit)
y.test.gpu.logit  <- ifelse(y.test.gpu.logit <= median.input, 0, 1)
head(y.test.gpu.logit)



##
## The Below code chunks is the implementation of the Gradient Descent method. Based on the experimentation performed, the best alpha selected for demonstration here
## is alpha = 0.0001 and the threshold as thold = 0.000001.
##

##
## Here, we are defining the Gradient Descent algorithm. First, we are declaring the variables to store cost, beta co-efficients, predicted target variable value and error.
##

gradient_descent <- function(x, y, alpha, m, beta, thold)
{
  cost_iter  <<- list()
  beta_iter  <<- matrix(0,nrow=m,ncol=15)
  yhat_iter  <<- list()
  error_iter <<- list()
  ##
  ## We are iterating over the matrices with the goal of minimizing the cost function value.
  ##
  for (i in 1:10000){
    
    yhat <- 1 / (1 + exp(-(as.matrix(x) %*% beta_value)))                                             ## Predictions of target variable.
    yhat_iter[i] <- yhat                                                                              ## Storing the predicted value.               
    
    error <- yhat - y                                                                                 ## Calculating the error value.
    error_iter[i] <- error                                                                            ## Storing the error value.
    
    cost <- -1 * (1/m) * sum( y*log(yhat) + (1-y)*log(1-yhat) )                                        ## Calculating the cost function value.
    cost_iter[i] <- cost                                                                              ## storing the cost function value.
    
    beta_value <- beta_value - (alpha * (1/m) * (t(x) %*% (yhat - y)))                                ## Calculating the new beta coefficinets values.
    beta_iter[i,1:15] <- t(beta_value)                                                                ## storing the beta coefficients value.
    
    
#    if ((i > 1) && ((cost_iter[[i-1]] - cost_iter[[i]]) < thold)) {
#      print('Threshold reached')
#      break
#    }
  }
  
  final_val <- list(cost_iter, beta_iter, yhat_iter, error_iter)                                      ## Storing the variables in a single variable so that it can be returned.
  return (final_val)                                                                                  ## Returning the values.
  
}


##
## Prediction function for the validation dataset.
##

linear_test_predict <- function(beta_conv_iter, x_gpu_test, y_gpu_test) 
{
  yhat_test  <- 1 / (1 + exp(-(as.matrix(x.test.gpu.logit) %*% beta_conv_iter)))
  error_test <- yhat_test - y_gpu_test
  cost_test  <- (1/(2*length(y_gpu_test))) * sum(y_gpu_test*log(yhat_test) + (1-y_gpu_test)*log(1-yhat_test))
  test_val <- list(yhat_test, error_test, cost_test)
  
  return(test_val)
}


##
## Let's define the main function for initializing the initial values of beta-i (slope) and beta-0 (y intercept)
##

main_function <- function(alpha, m, beta_value, thold){
  cost_return_train  <- list()
  beta_return_train  <- list()
  yhat_return_train  <- list()
  final_return_train <- list()
  
  cost_return_test   <- list()
  yhat_return_test   <- list()
  error_return_test  <- list()
  
  final              <- list()
  final_test         <- list()
  
  
  final <- gradient_descent(x.train.gpu.logit, y.train.gpu.logit, alpha, m, beta, thold)
  
  cost_return_train  <- final[[1]]
  beta_return_train  <- final[[2]]
  yhat_return_train  <- final[[3]]
  error_return_train <- final[[4]]
  
  
  conv_iter <- length(cost_return_train)
  conv_iter
  
  beta_conv_iter <- beta_return_train[conv_iter,1:15]
  beta_conv_iter
  
  cost_return_train[conv_iter]
  
  final_test <- linear_test_predict(beta_conv_iter, x_gpu_test, y_gpu_test)
  
  cost_return_test <- final_test[[3]]
  yhat_return_test <- final_test[[1]]
  error_return_test <- final_test[[2]]
  
  
  cost_return_test
  
  cost_result <- list(cost_return_train, cost_return_test, conv_iter, beta_conv_iter, yhat_return_test)
  return(cost_result)
  
}


##
## Invoking the main function to apply the Gradient Descent algorithm.
##

thold = 0.0000000001
alpha <- 0.00001
m <- nrow(gpu.train.df)
beta_value <<- rep(0,15)
cost_return <- main_function(alpha, m, beta_value, thold)
cost_return_train <- cost_return[[1]]
cost_return_test  <- cost_return[[2]]
conv_iter <- cost_return[[3]]
yhat_test <- cost_return[[4]]
cost_train_0.0001_al <- cost_return_train
cost_train_min_0.0001_al <- cost_return_train[conv_iter]
cost_test_0.0001_al <- cost_return_test



##
## Plotting various performance validation curves
##

plot(1:length(cost_train_0.0001_al), cost_train_0.0001_al, main = 'Cost function convergence at alpha 0.0001.', xlab = 'No. of Iterations', ylab = 'Cost Function value', col='red', type='l', xlim=c(0,10000), ylim=c(0.68,0.7),sub='Convergence Threshold value - 0.000001')
legend("topright",c("alpha=0.0001"),cex=0.7, bty='n', fill=c("red"))









