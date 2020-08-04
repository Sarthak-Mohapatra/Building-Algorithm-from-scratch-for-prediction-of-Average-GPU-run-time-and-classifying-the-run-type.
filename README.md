# Building Algorithms from scratch for prediction of Average GPU run time and classifying a run type as high or low time consuming.
As part of this project, I have developed algorithms from scratch using Gradient Descent method to predict average GPU Run Time and 
classify a run process in GPU as high or low time consuming.

The main focus of this project is to successfully create algorithms from scratch using Gradient Desent method for prediction and binary classfication. 

In this project, we have performed various experimentations by varying the learning rate(α) and the threshold value for convergence. We 
have observed the changes in the performance of the algorithms both in prediction and classification model.
Similarly, we have performed feature selection both by random selection and based on their importance. The prediction and classification performance were further evaluated and the best model with most significant features was finalized. 

## Table of Contents

* General Info
* Variable Description
* Feature Importance with Visualization
* Experimentations
* Technologies and Methods
* Project Report
* Status
* Contact

## General Info

For this project, we have used the SGEMM GPU kernel performance Data Set available for download at [UCI ML Website](https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance). The dataset measures 
the running time of various matrix multiplication processes where each matrix is of shape 2048 × 2048. The total number of observations 
in the dataset is 241600. For each test, four runs were performed, and their results were presented in the file. For our project, we 
have taken the average of four runs and have considered it as our target/dependent variable. 


## Variable Description

The GPU data set containing App details has 13 variables. The variable names and the description is mentioned below:

  - MWG - per-matrix 2D tiling at workgroup level	int	16, 32, 64, 128
  - NWG - per-matrix 2D tiling at workgroup level	int	16, 32, 64, 128
  - KWG - inner dimension of 2D tiling at workgroup level	int	16, 32
  - MDIMC - local workgroup size	Int	8, 16, 32
  - NDIMC - local workgroup size	Int	8, 16, 32
  - MDIMA - local memory shape	Int	8, 16, 32
  - NDIMB - local memory shape	Int	8, 16, 32
  - KWI - kernel loop unrolling factor	Int	2, 8
  - VWM - per-matrix vector widths for loading and storing	Int	1, 2, 4, 8
  - VWN - per-matrix vector widths for loading and storing	Int	1, 2, 4, 8
  - STRM - enable stride - access off-chip memory in 1 thread	Cat	0, 1
  - STRN - enable stride - access off-chip memory in 1 thread	Cat	0, 1
  - SA - per-matrix manual caching of the 2D workgroup tile	Cat	0, 1

  
 ## Feature Importance with Visualization
 
 The below screenshot shows the correlation of features:
 
 ![Heat-Map showing correlation](https://github.com/Sarthak-Mohapatra/Building-Algorithm-from-scratch-for-prediction-of-Average-GPU-run-time-and-classifying-the-run-type./blob/master/Heat-Map)
 
 It explains the correlation among variables in the dataset. The key takeaways from the above correlation matrix/ heat-map is mentioned below:
  - The dependent variable Average has the positive correlation with MWG and NWG. But the correlation coefficient is not high.
  - The dependent variable Average has negative correlation with NDIMC and MDIMC. But the correlation coefficient is not high.
  - The dependent variables have highest positive correlation of 0.35 with MWG and highest negative correlation of -0.22 MDIMC.
  - MWG has a small positive correlation with VWM. Also, NWG has a small positive correlation with VWN.
  - MDIMC is having a small negative correlation with NDIMC. 

Let's visualize the effect of VWM on other features:
![Effect of VWM](https://github.com/Sarthak-Mohapatra/Building-Algorithm-from-scratch-for-prediction-of-Average-GPU-run-time-and-classifying-the-run-type./blob/master/VWM%20effect)

From the graph we can observe that with increase in the order of VWM, the Average Run Time has increased significantly. Similarly, with increase in order of VWM, the order of MWG also seems to increase signifying the records having with higher order of VWM has higher order value of MWG. For all other variables/ features, there is not much significant impact of VWM as we can see that the changes are very minimal or unchanged.

Let's visualize the effect of STRM, STRN, SA and SB on Average Run Time:
![Effect of Categorical Variables](https://github.com/Sarthak-Mohapatra/Building-Algorithm-from-scratch-for-prediction-of-Average-GPU-run-time-and-classifying-the-run-type./blob/master/Categorical%20plots)

From the above graph, we can see that for STRM and STRN, there is not much variation in the Average Run Time. As compared to a run where the stride is not enabled to access off-chip memory in single thread (STRM = 0), the Average Run Time slightly decreases when stride is enabled to access off-chip memory in single thread (STRM = 1). Unlink STRM/ STRN, as compared to a run where per-matrix manual caching of the 2D workgroup tile is not there(SA=0/ SA=1), the Average Run Time increases when there is per-matrix manual caching of the 2D workgroup tile(SA=1/ SB=1) in a run process. 
Also, we can see that STRN has no impact on Average Run Time which is constant and has same mean value.

 ## Experimentations
 
 The Learning Rate (alpha), Convergence Threshold and Number of Iterations are few of the parameters based on which experimentations was 
 performed. The below table highlights the cost function value after an initial run of the Linear Regression using Gradient Descent with an alpha of 0.0001, convergence threshold of 0.00001 and number of iterations as 10000, 
 :
 

| __Data Set Type__ | __Alpha (Learning rate)__ | __Threshold__ |	__Num of Records(m)__ | __Min Cost Function Value__ |
| ----------------- | ------------------------- | ------------- | --------------------- | --------------------------- |
| __Training Data__ |	0.0001|	0.00001 |	169120 |	0.3651264 |
| __Validation Data__ |	0.0001 |	0.00001 |	72480 |	0.354251 |


The Learning Rate (alpha), Convergence Threshold and Number of Iterations are few of the parameters based on which experimentations was 
performed. The below table highlights the cost function value after an initial run of the Logistic Regression using Gradient Descent with an alpha of 0.0001, convergence threshold of 0.00001 and number of iterations as 10000, 
 :

| __Data Set Type__ | __Alpha (Learning rate)__ | __Threshold__ |	__Num of Records(m)__ | __Min Cost Function Value__ |
| ----------------- | ------------------------- | ------------- | --------------------- | --------------------------- |
| __Training Data__ |	0.0001|	0.00001 |	169120 |	0.1084185 |
| __Validation Data__ |	0.0001 |	0.00001 |	72480 |	0.5808993 |


## Technologies and Methods

* R-Studio
* Microsoft Excel

## Project Report

The project report is uploaded to the Git-Hub site and can be referenced here: [Project Report](https://github.com/Sarthak-Mohapatra/Building-Algorithm-from-scratch-for-prediction-of-Average-GPU-run-time-and-classifying-the-run-type./blob/master/Project%20Report.docx)

## Status

Project is: *finished*

## Contact

If you loved what you read here and feel like we can collaborate to produce some exciting stuff, or if you just want to shoot a question,
please feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/sarthakmohapatra1990/) or email me sarthakmohapatra1990@gmail.com.
