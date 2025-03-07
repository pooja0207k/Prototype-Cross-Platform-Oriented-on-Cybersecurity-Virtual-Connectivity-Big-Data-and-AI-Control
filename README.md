# Intrusion Detection System (IDS) Using Machine Learning

This project focuses on building an **Intrusion Detection System (IDS)** using machine learning algorithms to classify network traffic as either **normal** or **anomalous** (which could indicate a security threat, such as an intrusion or denial-of-service attack). The model uses various classification algorithms to predict the nature of network traffic, based on the features of the network flow.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Dataset Description](#dataset-description)
  - [Features/Columns](#featurescolumns)
- [Machine Learning Models](#machine-learning-models)
  - [Data Preprocessing](#data-preprocessing)
  - [Models Used](#models-used)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The objective of this project is to apply machine learning algorithms to classify network traffic, focusing on identifying anomalous traffic that could indicate potential intrusions. The model can be used as part of an Intrusion Detection System (IDS) to improve cybersecurity by detecting potential threats in real-time.

The dataset used in this project is from the **KDD Cup 1999** competition, which is a well-known dataset used for intrusion detection research.

## Dataset

### Dataset Description

The dataset used in this project is the **KDD Cup 1999 dataset**. This dataset consists of **network traffic records**, with each record representing a single network connection (or flow) observed by a system. The goal is to classify each flow as either **normal** or **anomalous** (which could represent an attack, such as Denial-of-Service (DoS), Probe, Remote to Local (R2L), or User to Root (U2R) attacks).

The dataset contains **41 features** that describe various properties of the network connection and a **class label** that indicates whether the connection is normal or anomalous.

### Features/Columns

The dataset consists of the following columns (features) that represent network connection attributes:

1. **duration**: The duration (in seconds) of the network connection.
2. **protocol_type**: The protocol used for communication (e.g., TCP, UDP, ICMP).
3. **service**: The service/application running on the network (e.g., HTTP, FTP).
4. **flag**: The status or flag associated with the connection (e.g., normal, error).
5. **src_bytes**: The number of bytes sent from the source.
6. **dst_bytes**: The number of bytes sent to the destination.
7. **land**: Whether the source and destination IP addresses are the same (self-packet).
8. **wrong_fragment**: The number of incorrect or fragmented packets in the connection.
9. **urgent**: The number of urgent packets in the connection.
10. **real**: Real-time data about the connection.
11. **hot**: Whether the connection is considered "hot" (frequently observed).
12. **num_failed_logins**: The number of failed login attempts.
13. **logged_in**: Whether the user is logged in.
14. **num_compromised**: The number of compromised accounts.
15. **root_shell**: Whether the root shell (privileged access) was used in the connection.
16. **su_attempted**: The number of attempts to switch to a superuser.
17. **num_root**: The number of root-level operations performed.
18. **num_file_creations**: The number of files created during the connection.
19. **num_shells**: The number of shell accesses.
20. **num_access_files**: The number of files accessed during the connection.
21. **num_outbound_cmds**: The number of outbound commands.
22. **is_host_login**: Whether the host is logged in.
23. **is_guest_login**: Whether a guest login attempt was made.
24. **count**: The number of connections made to the same host.
25. **srv_count**: The number of connections made to the same service.
26. **serror_rate**: The error rate for the connection.
27. **srv_serror_rate**: The error rate for the same service.
28. **rerror_rate**: The rate of connection refusals.
29. **srv_rerror_rate**: The rate of connection refusals for the same service.
30. **same_srv_rate**: The percentage of connections to the same service.
31. **diff_srv_rate**: The percentage of connections to a different service.
32. **srv_diff_host_rate**: The percentage of connections to different hosts with the same service.
33. **dst_host_count**: The number of connections made to the destination host.
34. **dst_host_srv_count**: The number of connections made to the same service on the destination host.
35. **dst_host_same_srv_rate**: The percentage of connections to the same service on the destination host.
36. **dst_host_diff_srv_rate**: The percentage of connections to different services on the destination host.
37. **dst_host_same_src_port_rate**: The percentage of connections to the destination from the same source port.
38. **dst_host_srv_diff_host_rate**: The percentage of connections to different hosts for the same service on the destination.
39. **dst_host_serror_rate**: The error rate on the destination host.
40. **dst_host_srv_serror_rate**: The error rate on the same service for the destination host.
41. **dst_host_rerror_rate**: The error rate for refusals on the destination host.

### **Class Label (Target Variable)**
- **0**: **Normal** traffic
- **1**: **Anomalous** traffic (can include various types of attacks like DoS, Probe, R2L, U2R)

## Machine Learning Models

### Data Preprocessing

1. **Label Encoding**: Since the dataset contains categorical features (such as protocol type, service, flag, and class), we use **LabelEncoder** from the `sklearn` library to convert them into numerical values. This is necessary because machine learning models require numeric input.
   
2. **Feature Selection**: To enhance model performance, **SelectKBest** with **Chi-Square (chi2)** scoring function is used to select the top 10 features based on their relevance to the target variable.

3. **Feature Importance**: Using **ExtraTreesClassifier**, the importance of each feature is assessed to determine which features contribute most to classification.

4. **Missing Value Handling**: The dataset might contain missing values, and these are handled appropriately to avoid data inconsistencies during model training.

### Models Used

The following machine learning models are implemented for the classification task:

1. **Support Vector Machine (SVM)**: A classifier that works well for binary classification tasks and is effective in high-dimensional spaces.
   
2. **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies a data point based on how its neighbors are classified.

3. **AdaBoost Classifier**: An ensemble technique that combines multiple weak classifiers to create a stronger classifier by assigning higher weights to misclassified instances.

4. **Decision Tree Classifier**: A tree-based classifier that splits data into subgroups based on feature values to make predictions.

5. **Recurrent Neural Network (RNN)**: A deep learning model used for sequence prediction, which is trained to classify network traffic patterns.

### Results

The following performance metrics were evaluated for each model:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

| Algorithm                  | Accuracy  |
| -------------------------- | --------- |
| AdaBoostClassifier          | 0.956975  |
| SVM                         | 0.594589  |
| Decision Tree Classifier    | 0.982923  |
| KNN                         | 0.966955  |
| RNN                         | 0.911289  |

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/pooja0207k/AI-Based-Intrusion-Detection.git
    cd AI-Based-Intrusion-Detection

    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. The results will be printed, showing the accuracy and classification report for each model.

## Acknowledgments

- **KDD Cup 1999 Dataset**: Used for intrusion detection research and competitions.
- **Scikit-learn**: Used for implementing machine learning models and feature selection.
- **Keras**: Used for building the Recurrent Neural Network (RNN) model.

