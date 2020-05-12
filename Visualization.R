##
### Loading required packages
##
pacman::p_load(data.table, forecast, leaps, tidyverse, caret, corrplot, glmnet, mlbench, ggplot2, gplots, pivottabler,MASS, e1071, fpp2, gains, pROC, knitr, gplots, FNN, 
               RColorBrewer, viridis, cowplot, ggpubr, gridExtra, rlist, d3heatmap)
##
## Data Loading and formating 
##
getwd()
gpu.df <- read.csv("sgemm_product.csv")

str(gpu.df)
head(gpu.df)

names(gpu.df)[15] = "Run1"
names(gpu.df)[16] = "Run2"
names(gpu.df)[17] = "Run3"
names(gpu.df)[18] = "Run4"

head(gpu.df)

gpu.df$Average <- (gpu.df$Run1 + gpu.df$Run2 + gpu.df$Run3 + gpu.df$Run4) / 4

head(gpu.df)
summary(gpu.df)
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

corrplot(cor(gpu_norm_df[]), method = "color", type = "lower", order = "hclust", tl.srt = 45)

##
##
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
## Checking ordinal and categorical variables impact on Average GPU Run
##
cd <- gpu.df %>%
  group_by(VWM) %>%
  summarise(MWG = mean(MWG), NWG = mean(NWG), KWG = mean(KWG), MDIMC = mean(MDIMC), NDIMC = mean(NDIMC), MDIMA = mean(MDIMA), NDIMB = mean(NDIMB), AvgRunTime = mean(Average))
cd

ce <- gpu.df %>%
  group_by(VWN) %>%
  summarise(MWG = mean(MWG), NWG = mean(NWG), KWG = mean(KWG), MDIMC = mean(MDIMC), NDIMC = mean(NDIMC), MDIMA = mean(MDIMA), NDIMB = mean(NDIMB), AvgRunTime = mean(Average))
ce

cf <- gpu.df %>%
  group_by(STRM) %>%
  summarise(AvgRunTime = mean(Average))
cf

cg <- gpu.df %>%
  group_by(STRN) %>%
  summarise(AvgRunTime = mean(Average))
cg

ch <- gpu.df %>%
  group_by(SA) %>%
  summarise(AvgRunTime = mean(Average))
ch

ci <- gpu.df %>%
  group_by(SB) %>%
  summarise(Avg = mean(Average))
ci

cd <- as.data.frame(cd)
ce <- as.data.frame(ce)
cf <- as.data.frame(cf)
cg <- as.data.frame(cg)
ch <- as.data.frame(ch)
ci <- as.data.frame(ci)

##
## Visualization
##
corrplot(cor(gpu.df[c(-15:-18)]), method = "color", type = "lower", order = "hclust", tl.srt = 45)

colmain <- col<- colorRampPalette(c("red","skyblue","blue"))(822)
heatmap.2(cor(gpu.df[c(-15:-18)]), col=colmain, cellnote = round(cor(gpu.df[c(-15:-18)]),2), dendrogram = "none",
          key = FALSE, trace = "none", margins = c(10,10), notecol = "black", main='Heat Map')

hist(gpu.df$Average, col='darkgreen', border='black', main='Distribution of Average Run Time.', xlab = 'Number of occurances.', ylab = 'Avg. Run Time Value.')
hist(log(gpu.df$Average), col='darkgreen', border='black', main='Distribution of Average Run Time.', xlab = 'Number of occurances.', ylab = 'Avg. Run Time Value.')


colors = c("skyblue2","blue4")
colors2 = c("skyblue1","skyblue3","blue3","black")

barplot(as.matrix(cd),beside=TRUE, cex.lab=1.0, cex.main=1.4, col=colors2, xlab='Feature names - factored based on VWM', ylab='Mean Value of Features', main='Effect of VWM on various Features.')
legend("topleft",c("VWM=1","VWM=2","VWM=4","VWM=8"),cex=1.0, bty='y', fill=colors2 )

barplot(as.matrix(ce),beside=TRUE, cex.lab=1.0, cex.main=1.4, col=colors2, xlab='Feature names - factored based on VWN', ylab='Mean Value of Features', main='Effect of VWN on various Features.')
legend("topleft",c("VWN=1","VWN=2","VWN=4","VWN=8"),cex=1.0, bty='y', fill=colors2 )

par(mfrow=c(2,2))
barplot(as.matrix(cf),beside=TRUE, cex.lab=1.0, cex.main=1.0, col=colors, xlab='Feature names - factored based on STRM', ylab='Mean Value of Features', main='Effect of STRM on various Features.')
legend("topleft",c("STRM=0","STRM=1"),cex=0.8, bty='n', fill=colors )

barplot(as.matrix(cg),beside=TRUE, cex.lab=1.0, cex.main=1.0, col=colors, xlab='Feature names - factored based on STRN', ylab='Mean Value of Features', main='Effect of STRN on various Features.')
legend("topleft",c("STRN=0","STRN=1"),cex=0.8, bty='n', fill=colors )

barplot(as.matrix(ch),beside=TRUE, cex.lab=1.0, cex.main=1.0, col=colors, xlab='Feature names - factored based on SA', ylab='Mean Value of Features', main='Effect of SA on various Features.')
legend("topleft",c("SA=0","SA=1"),cex=0.8, bty='n', fill=colors )

barplot(as.matrix(ci),beside=TRUE, cex.lab=1.0, cex.main=1.0, col=colors, xlab='Feature names - factored based on SB', ylab='Mean Value of Features', main='Effect of SB on various Features.')
legend("topleft",c("SB=0","SB=1"),cex=0.8, bty='n', fill=colors )

par(mfrow=c(2,2))
plot(gpu.df$MWG, gpu.df$Average, main = 'Average Run Time ~ MWG', xlab = 'MWG', ylab = 'Average Run Time', col=factor(gpu.df$MWG), pch=18)
plot(gpu.df$NWG, gpu.df$Average, main = 'Average Run Time ~ NWG', xlab = 'NWG', ylab = 'Average Run Time', col=factor(gpu.df$NWG), pch=18)
plot(gpu.df$KWG, gpu.df$Average, main = 'Average Run Time ~ KWG', xlab = 'KWG', ylab = 'Average Run Time', col=factor(gpu.df$KWG), pch=18)
plot(gpu.df$KWI, gpu.df$Average, main = 'Average Run Time ~ KWI', xlab = 'KWI', ylab = 'Average Run Time', col=factor(gpu.df$KWI), pch=18)

par(mfrow=c(2,2))
plot(gpu.df$MDIMC, gpu.df$Average, main = 'Average Run Time ~ MDIMC', xlab = 'MDIMC', ylab = 'Average Run Time', col=factor(gpu.df$MDIMC), pch=18)
plot(gpu.df$NDIMC, gpu.df$Average, main = 'Average Run Time ~ NDIMC', xlab = 'NDIMC', ylab = 'Average Run Time', col=factor(gpu.df$NDIMC), pch=18)
plot(gpu.df$MDIMA, gpu.df$Average, main = 'Average Run Time ~ MDIMA', xlab = 'MDIMA', ylab = 'Average Run Time', col=factor(gpu.df$MDIMA), pch=18)
plot(gpu.df$NDIMB, gpu.df$Average, main = 'Average Run Time ~ NDIMB', xlab = 'NDIMB', ylab = 'Average Run Time', col=factor(gpu.df$NDIMB), pch=18)






















