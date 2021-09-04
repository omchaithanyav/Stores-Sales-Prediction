# 🏪🏬 Stores-Sales-Prediction
![banner-4](https://user-images.githubusercontent.com/45726271/132082577-7acf21ca-444b-4921-9822-5e14e088bb13.png)
This repository represents: "Sales Stores Prediction"(An end to end ML Project)

# ⏳ Dataset
* Download the dataset directly from kaggle:
https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data

# 📝 Description
Problem Statement:
Nowadays, shopping malls and Big Marts keep track of individual item sales data in 
order to forecast future client demand and adjust inventory management. In a data 
warehouse, these data stores hold a significant amount of consumer information and 
particular item details. By mining the data store from the data warehouse, more 
anomalies and common patterns can be discovered.

# My Approach to this problem:
1. First i started by exploring the dataset (link to download dataset is given above☝️).
2. Then i did some data preprocessing(Like- imputing missing data properly, replacing deuplicate classes, etc..) which was a necessary step.
3. After data preprocessing i performed some data analysis using python libraries(matplotlib, klib and seaborn).
4. Then i split the data to train and validate the model and after splitting i performed scaling operation to normalize the features.
5. The data is now ready to feed to our model, I tried various models for prediction and i found that Gradient Boosting Regressor was giving the best results.
6. Then i saved my model.(necessary step)
7. Used Flask to code a web application and in this code i loaded my saved model and used it for predictions.
8. Finally when the app was working well in the local server, i deployed it in heroku platform.

# 🖥️ Installation
# 🛠️ Requirements
* Python 3.0+
* pandas
* numpy
* scikit-learn
* Flask
* matplotlib
* seaborn

#Application Link: https://stores-sales-prediction-app.herokuapp.com/ 
