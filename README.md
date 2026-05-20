# Telecom-Customer-Churn-Prediction-Decision-Tree-Approach

##  Project Overview
Customer retention is a critical metric for telecommunication companies. This project implements a **Decision Tree Classifier** to predict customer churn based on historical data. By analyzing customer demographics, account information, and service usage, the model aims to identify customers at high risk of leaving, enabling business teams to design proactive retention strategies.

##  About the Developer
Hello! I am Vydhyam Vishnusai, an ML-based developer and third-year B.Tech student specializing in Computer Science (Artificial Intelligence and Machine Learning) at Mohan Babu University. As I prepare for Machine Learning Engineer roles, I built this repository to demonstrate my proficiency in Python-driven data science, end-to-end data pipelines, exploratory data analysis (EDA), and predictive modeling.

##  Tech Stack & Tools
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Automated EDA:** Sweetviz

##  Dataset & Preprocessing
The dataset used contains **7,043** customer records and **21** features, including `tenure`, `MonthlyCharges`, `Contract` type, and the target variable `Churn`. 
* **Data Cleaning:** Transformed `TotalCharges` to numeric data types, handling missing or anomalous values by filling them with `0`. Standardized the structure by setting `customerID` as the dataset index.
* **Exploratory Data Analysis:** Leveraged **Sweetviz** to generate a comprehensive HTML report, rapidly visualizing target characteristics and feature associations (e.g., the correlation between `tenure` and `Churn`).
* **Feature Encoding:** Applied `LabelEncoder` across 16 categorical features to transform categorical text data into a machine-readable format.

##  Model Architecture
* **Algorithm:** Decision Tree Classifier
* **Core Hyperparameters:** `criterion='entropy'`, `random_state=42`

##  Model Performance & Evaluation
The model was evaluated using both accuracy scores and a detailed classification report to measure its predictive power on unseen data:
* **Training Accuracy:** `93.59%`
* **Testing Accuracy:** `70.12%`
* **Analysis:** The model achieves a high training accuracy but experiences a drop during testing, which is a classic indicator of high variance (overfitting) native to deep Decision Trees. The testing data yielded a macro average F1-score of `0.60` and a precision of `0.79` for non-churning customers. Recognizing this variance is a crucial step in the ML lifecycle, pointing directly to the next phase of optimization.


