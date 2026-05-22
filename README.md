# Telecom Customer Churn Prediction using Decision Trees

## 📌 Overview
This project implements a Machine Learning workflow to predict telecom customer churn using a **Decision Tree Classifier**. The goal is to identify customers who are likely to cancel their subscriptions based on their demographic details, account information, and service usage. The project covers the entire pipeline from Automated Exploratory Data Analysis (EDA) to Data Preprocessing, Model Training, and Hyperparameter Tuning.

## 📊 Dataset
The dataset used is the **Telecom Customer Churn** dataset. It contains customer-level data with 21 columns including:
* **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account Info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Services:** `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
* **Target Variable:** `Churn` (Yes/No)

## 🛠️ Tech Stack & Libraries
* **Python 3.x**
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Data Visualization)
* **Sweetviz** (Automated EDA)
* **Scikit-Learn** (Machine Learning & Preprocessing)
* **SciPy** (Statistical distributions for tuning)

## 🚀 Project Workflow

### 1. Exploratory Data Analysis (EDA)
Instead of manual univariate and bivariate analysis, this project utilizes the **Sweetviz** library to generate a comprehensive, highly-detailed HTML report (`sweetviz_report.html`). This automated report rapidly outlines feature distributions, correlations, and relationships with the target variable.

### 2. Data Transformation & Preprocessing
* **Indexing:** Set `customerID` as the dataframe index to exclude it from predictive modeling while keeping records identifiable.
* **Handling Data Types & Missing Values:** Converted the `TotalCharges` column to a numeric data type, forcefully coercing errors into `NaN`, and subsequently filling missing values with `0`.
* **Categorical Encoding:** Applied `LabelEncoder` from Scikit-Learn to convert 16 string/categorical variables into machine-readable numerical formats.

### 3. Base Model Implementation
* Split the dataset into training and testing sets.
* Trained a baseline **Decision Tree Classifier** using the `entropy` criterion (Information Gain). 

### 4. Hyperparameter Tuning
To combat the natural tendency of Decision Trees to overfit the training data, extensive hyperparameter tuning was applied:
* **GridSearchCV:** Conducted an exhaustive search over a specified parameter grid, adjusting `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, and `min_samples_split`.
* **RandomizedSearchCV:** Utilized to sample a given number of candidates from a parameter space with a specified distribution, optimizing for runtime while seeking high-quality splits.

## 📈 Results & Evaluation
The model's performance was evaluated using Accuracy Scores and detailed Classification Reports (Precision, Recall, F1-Score). Hyperparameter tuning successfully reduced overfitting and improved the model's generalization on unseen data.

**Base Decision Tree Performance:**
* Training Accuracy: ~82.87%
* Testing Accuracy: ~75.94%

**Tuned Decision Tree Performance (Best Params: `max_depth=7, max_leaf_nodes=20, min_samples_leaf=1, min_samples_split=2`):**
* Training Accuracy: ~79.85%
* Testing Accuracy: ~78.42%

*Note: The reduction in training accuracy alongside the increase in testing accuracy demonstrates a successfully generalized model that mitigates overfitting.*

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
