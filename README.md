# Credit_card_froud_detection
🚀 Credit Card Fraud Detection using Logistic Regression

This project builds a machine learning model to detect fraudulent credit card transactions using Logistic Regression. Follow these steps to understand and implement it yourself:

1️⃣ Import Dependencies

📦 Use NumPy, Pandas, and Scikit-learn to handle data and train the model.
python
Copy
Edit
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  

2️⃣ Load the Dataset

📂 Load your credit_data.csv into a Pandas DataFrame:
python
Copy
Edit
credit_card_data = pd.read_csv('/content/credit_data.csv')  

📊 Check the first few rows using:
python
Copy
Edit
credit_card_data.head()

3️⃣ Data Exploration

🔍 View data info and check for missing values:
python
Copy
Edit
credit_card_data.info()  
credit_card_data.isnull().sum()  

🧮 Check the distribution of legitimate (0) and fraudulent (1) transactions:
python
Copy
Edit
credit_card_data['Class'].value_counts()

4️⃣ Separate Legitimate & Fraudulent Transactions

Split data into two groups for comparison:
python
Copy
Edit
legit = credit_card_data[credit_card_data.Class == 0]  
fraud = credit_card_data[credit_card_data.Class == 1]  

5️⃣ Handle Class Imbalance with Under-Sampling

🏷️ Balance the dataset by sampling the same number of legitimate transactions as fraudulent:
python
Copy
Edit
legit_sample = legit.sample(n=492)  
new_dataset = pd.concat([legit_sample, fraud], axis=0)  

6️⃣ Split Data into Features & Labels

Separate features (X) from labels (Y):
python
Copy
Edit
X = new_dataset.drop(columns='Class', axis=1)  
Y = new_dataset['Class']  
Split into training (80%) and testing (20%) sets:
python
Copy
Edit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)  

7️⃣ Train the Model

🏋️ Train a Logistic Regression model on the balanced training data:
python
Copy
Edit
model = LogisticRegression()  
model.fit(X_train, Y_train)  

8️⃣ Evaluate the Model

✅ Calculate the accuracy on the training data:
python
Copy
Edit
X_train_prediction = model.predict(X_train)  
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)  
print('Accuracy on Training data : ', training_data_accuracy)  
📈 Test the accuracy on unseen test data:
python
Copy
Edit
X_test_prediction = model.predict(X_test)  
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)  
print('Accuracy score on Test Data : ', test_data_accuracy)  
🎯 Next Steps
Try using advanced algorithms (e.g., Random Forest or XGBoost)
Tune hyperparameters for better accuracy
Add more features for improved fraud detection




/* ✨ For more information, log in to my profile or connect through LinkedIn to stay updated! 😊 */
