
Titanic-Dataset-Prediction

Project Overview 

This project predicts whether a passenger would survive the Titanic disaster using Machine Learning models trained on the famous Titanic dataset. This project analyzes Titanic passenger data to predict survival outcomes using Logistic Regression, Decision Tree, and Random Forest models. It includes data preprocessing, feature engineering, model training, evaluation, and visualization to gain insights into survival factors. The best-performing model is Logistic Regression, achieving an accuracy of 81.56%. 

  

🗂️ Dataset Information 

  

The dataset used is the Titanic dataset from Kaggle, which includes features such as: 

Pclass (Passenger class: 1st, 2nd, 3rd) 

Sex (Male/Female) 

Age 

Fare (Ticket Fare) 

Embarked (Port of Embarkation) 

Survived (Target variable: 0 = No, 1 = Yes) 

 

 

Dataset 

The dataset is from the Titanic competition dataset. 

It contains passenger details, including age, gender, ticket fare, class, and whether they survived. 

Missing values are handled for Age (filled with median) and Embarked (filled with mode). 

 

Data Preprocessing 

 Removed unnecessary columns (Name, Ticket, Cabin, PassengerId). 

 Handled missing values (Age, Embarked). 

 Encoded categorical variables (Sex, Embarked). 

 Standardized numerical features (Age, Fare) 

 Split data into training (80%) and testing (20%) sets. 

 

Exploratory Data Analysis (EDA) 

Various visualizations were created to understand the data, including: 

Survival rate by gender – Females had a higher survival rate. 

 Survival rate by class – 1st class passengers had a higher chance of survival. 

 Age distribution – Younger passengers had a slightly higher survival rate. 

 Feature correlation heatmap – Showed relationships between features. 

 

 

Machine Learning Models 

Three classification models were trained and evaluated: 
1️⃣ Logistic Regression 
2️⃣ Random Forest Classifier 
3️⃣ Decision Tree Classifier 

Each model was tested for accuracy and performance. 

 

 

 

Model Performance Comparison 

Model 

Accuracy 

Logistic Regression 

81.56% 

Random Forest 

77.09% 

Decision Tree 

75.98% 

The Logistic Regression model performed the best and is used for predictions in the app. 

Installation & Setup 

1. Install Dependencies 

            pip install -r requirements.txt 

2. Run the program 

       python  titanic_predict.py 
