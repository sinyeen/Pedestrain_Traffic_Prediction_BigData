# Data Processing for Big Data: Pedestrain Traffic Prediction :walking_man: :vertical_traffic_light:
![](https://img.shields.io/badge/Author-SinYee-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a) ![](https://img.shields.io/badge/Tool-PySpark-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a) ![](https://img.shields.io/badge/Tool-MLlib-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a) ![](https://img.shields.io/badge/Tool-Pandas-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a)
## Project Overview
Develop machine learning models to predict the pedestrian traffic in the City of Melbourne. The machine learning models would be further integrated into the streaming platform using `Apache Kafka` and `Apache Spark` Streaming to perform prediction. The aim of this project is to build models for predicting whether the potential count would go above the threshold of 2000 and also predicting the possible count.

| Use Case | Description | Model |
| --- | ----------- | --- |
| 1 | Predict whether count would go above 2000 for the hours **between 9:00am and midnight** | Binary classification|
| 2 | Predict the possible count for the hours **between 9:00am and midnight** | Regression |

## Architecture
![](https://github.com/sinyeen/Pedestrain_Traffic_Prediction_BigData/blob/main/Images/Arch.JPG)

## Exploratory Data Analysis EDA
* Summary Statistic
Show basic statistics for each numeric column except <span style="color: green;">above_threshold</span> and <span style="color: green;">Date_Time</span>. Basic statistic to be shown:

![](https://github.com/sinyeen/Pedestrain_Traffic_Prediction_BigData/blob/main/Images/sum%20stats.JPG)

* Count threshold values
Count of above-threshold and below-threshold to see whether there is class imbalance.

![](https://github.com/sinyeen/Pedestrain_Traffic_Prediction_BigData/blob/main/Images/count%20no.JPG)

* Histogram
Display a histogram to show the distribution of the hourly counts. 

![](https://github.com/sinyeen/Pedestrain_Traffic_Prediction_BigData/blob/main/Images/histogram.JPG)

**Observation and Discussion**

The distribution is skewed to the right. This shows that the data is imbalanced as most of the hourly counts fall below 2000. In general, the frequency decreases as the hourly count increases.

The minority class is harder to predict because there are few examples of this class, by definition. This means it is more challenging for a model to learn the characteristics of examples from this class, and to differentiate examples from this class from the majority class (or classes). 

When working with an imbalanced classification problem, the minority class is typically of the most interest. This means that a model’s skill in correctly predicting the class label or probability for the minority class is more important than the majority class or classes. the learning process of most classification algorithms is often biased toward the majority class examples, so that minority ones are not well modeled into the final system.

The abundance of examples from the majority class (or classes) can swamp the minority class. Most machine learning algorithms for classification predictive models are designed and demonstrated on problems that assume an equal distribution of classes. This means that a naive application of a model may focus on learning the characteristics of the abundant observations only, neglecting the examples from the minority class that is, in fact, of more interest and whose predictions are more valuable. Also, this may cause the recall of the prediction model to be very low even with a high accuracy. 

* Line-plot
Plot line-plot to show the trend of the average daily count change by month. 

![](https://github.com/sinyeen/Pedestrain_Traffic_Prediction_BigData/blob/main/Images/line%20plot.JPG)

* Time in a day vs pedestrain counts
Correlation plots for the relationship between time in a day and the hourly pedestrain counts for different days in a week.

![](https://github.com/sinyeen/Pedestrain_Traffic_Prediction_BigData/blob/main/Images/trend.JPG)

* Time in a day vs pedestrain counts
Correlation plots for the relationship between time in a day and the hourly pedestrain counts for different days in a week.

![](https://github.com/sinyeen/Pedestrain_Traffic_Prediction_BigData/blob/main/Images/pes%20count.JPG)

**Observation and Discussion**

The average daily pedestrain counts for February, March and December are the highest among the months. The are a sharp drop from March to May. The pedestrain counts in May to August are relatively low compared to other months, and June has the lowest counts. The pedestrain count increases starting from August to December.

The average pedestrian traffic volume of the weekdays is higher than the weekends. There are three peak periods in a day for the weekdays i.e. around 7:30a.m., 12p.m. and 1p.m., and 5p.m. These hours are assumed to be the “go to work" hour, lunch hour, and “back from work" hour respectively. The volume of pedestrian is at the least from midnight until dawn i.e. 1a.m. to 5a.m. Unlike the weekdays, the trend lines of the weekends are much smoother as the weekends have only one peak in a day, that is from 1p.m. to 3p.m. Moreover, the pedestrian volume during the time between midnight and dawn is higher than the weekdays.

The average pedestrian traffic volume of Sensor 38 - Flinders St-Swanston St (West) is the highest. Sensor 4, 22, 41, 60 are also considered in the top 5 sensors with the most average pedestrain count in the Melbourne CBD. From the plot, we can see that most sensors have recorded more pedestrains in the weekdays than the weekends. Specifically, Sensor 15, 19, 26, 28, 32, 35, and 66 have higher pedestrain counts in the weekends. In addition, the pedestrain count trends for the weekdays and weekends are similar, with the exception of Sensor 9, 13, 16, 18, 24, 57, and 58. These sensors have big different in the pedestrain counts between the weekdays and weekends. From this information, we know that these places are crowded only in the weekdays, thus these places possible near to work places or school.

## Feature Engineering
Perform one-step time-series prediction
* Statistical Method: Hypothesis Testing 
* Visualisation: Look at **Seasonality** and **Correlation**

Prepare Pipelines for Spark ML Transformers/Estimators for features and labels
* `StringIndexer`
* `One Hot Encoder (OHE)`
* `VectorAssembler`

## Models Evaluation
**Use case 1: Predict whether count would go above 2000 for the hours between 9:00am and midnight**

* Decision Tree Model:
  * Accuracy = 0.905993
  * Precision = 0.642891
  * Recall = 0.304307
  * f1 = 0.413084 

* Gradient Boosted Tree Model:
  * Accuracy = 0.938833
  * Precision = 0.837779
  * Recall = 0.542379
  * f1 = 0.658466 

**Use case 2: Predict the possible count for the hours between 9:00am and midnight**

* Decision Tree Model: RMSE = 660.7

* Gradient Boosted Tree Model: RMSE = 625.3

**Discussion:**

*Use Case 1* 

Accuracy is normally suitable to be used in measuring the model performance of binary multiclass classification. However, The data set is imbalanced, and all the data points are classified as the majority class data points (i.e., hourly counts lower than 2000), causing high accuracy of the model. Therefore, although the models are relatively accurate, it will be not valuable as accuracy is not so reliable in measuring the model performance for this data set. F-score is most suitable to measure the model performance because it is a score that maintains a balance between the precision and recall of the model, where precision will summarise the fraction of the true positive class to the positive class and recall measure how accurate the positive class was predicted. For this case, it is important to have for the model that is good at precision and recall. For example, it is vital to be sure that the pedestrian counts at the specific location is actually high i.e., >2000 (precision) for the performers, and to record as many locations with high pedestrian counts in the CBD of Melbourne as possible (recall).

According to the AUC results, the F1 score is higher in the Gradient Boosted Tree (GBT) Model, thus a better model. Gradient-Boosted Trees (GBTs) are ensembles of decision trees. GBTs iteratively train decision trees in order to minimize a loss function. Like decision trees, GBTs handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.

*Use Case 2*

The GBT model is better as it has lower RMSE and R-square that is closer to 1. The GBT model with lower RMSE have predicted values that are closer to the observed data. Its R-square that is closer to 1 also shows that there are more observed data fitted into the regression line.


