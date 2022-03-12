# Out-of-Distribution input prediction for Task-Oriented Dialogue Systems
Explore anomaly detection techniques to enhance chat-bot responses.

The main objective of this project is to predict the out-of-distribution input data and pass only in-scope data to the chat-bots. The identified out-of- distribution data could be further used for re-training the model. In this way, the risk of giving incorrect output could be minimized.

This project has been submitted as Capstone project for Udacity AWS Machine Learning Engineer Nanodegree Program. 
(https://confirm.udacity.com/C4VUQJ7Q)

## Project Set Up and Installation
Use jupyter-lab to run .ipynb files
    Approach 1 (attempt) - variational_autoencoder.ipynb
    Approach 2 (attempt) - project_isolation.ipynb
    Approach 3 (successful) - sngp_with_bert_aws.ipynb
install the depenencies mentioned in the notebooks.
For SNGP with BERT model use AWS sageMaker studio with GPU instance for quicker training times (about 20 minutes with ml.g4dn.xlarge instance)

## Dataset
Dataset has been downloaded from (https://www.kaggle.com/stefanlarson/outofscope-intent-classification-dataset) is present in project folder.
The input dataset downloaded from kaggle. This dataset contains 15000 user's spoken queries collected over 150 intent classes, it also contains 1000 out-of-domain (OOD) sentences that are not covered by any of the known classes.



## Software and libraries
Below standard libraries were usd and are included in the import section of each notebook
Python 3.8
Keras
Tensorflow
SKlLearn
matplotlib
GloVe embeddings are downloaded in glove folder

##Hardware dependencies
Best to use a GPU to train SNGP with BERT model.


### Results
Metrics used
1. Precision
Precision is a measure of how well a model avoids classifying negative examples as positive. It is calculated as (True Positives/ (True Positives + False Positives))
In this project, precision is the measure of in-distribution examples not being classified as Out-of-distribution. A higher number of false positives would mean lower precision.
2. Recall
Recall is a measure of how well a model avoids classifying positive examples as negative. It is calculated as (True Positives / (True Positives + False Negatives)).
In this project, recall is the measure of out-of-distribution examples not being classified as in-distribution. A higher number of false negatives would mean lower recall.
3. AUPRC (Area Under Precision and Recall Curve)
AUPRC metric is used with imbalanced data where the focus is optimizing for true positives. Maximum AUPRC means perfect precision and perfect recall. In real-world it is always a trade-off between precision and recall.
In this project, objective is to find out-of-distribution data efficiently.  Thus, recall is to be optimized.
Below graph shows AUPRC curve for 3 runs.

SNGP Model 1
Batch size = 32
Epochs = 3   
 
SNGP Model 2
Batch size = 16
Epochs = 2  
  
SNGP Model 3
Batch size = 16
Epochs = 4

Training Accuracy    
SNGP Model 1 98.53%    
SNGP Model 2 99.39%    
SNGP Model 3 99.75%

Validation Accuracy    
SNGP Model 1 95.98%    
SNGP Model 2 95.98%    
SNGP Model 3 96.27%

AUPRC    
SNGP Model 1 90.26%    
SNGP Model 2 89.26%    
SNGP Model 3 89.15%

Justification
Benchmark Model - BERT – Full     
Validation Accuracy = 96.2 %
Recall = 52.3%

SNGP Model 3
Validation Accuracy = 96.29%
Recall = 100%
