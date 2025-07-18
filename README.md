
# Employee Burnout Prediction

## Project Overview
This project aims to predict employee burnout rates using various HR-related and environmental factors. By leveraging machine learning models, we identify key drivers of burnout and develop a predictive tool to help organizations proactively address employee well-being.

## Dataset
The dataset used in this project includes features such as Employee ID, Date of Joining, Gender, Company Type, WFH Setup Availability, Designation, Resource Allocation, Mental Fatigue Score, and the target variable, Burn Rate.

## Methodology

### 1. Data Loading and Initial Inspection
Loaded and performed an initial inspection of the training and test datasets.

### 2. Exploratory Data Analysis (EDA)
Conducted detailed EDA, including:
- Generating descriptive statistics.
- Visualizing distributions and relationships between features (e.g., correlation matrix).
- Identifying missing values.

### 3. Data Preprocessing
Prepared the data for modeling through several steps:
- **Handling Missing Values:** Imputed missing values in `Resource Allocation`, `Mental Fatigue Score`, and `Burn Rate` using the median strategy.
- **Encoding Categorical Variables:** Applied One-Hot Encoding to `Gender`, `Company Type`, and `WFH Setup Available` to convert them into numerical formats.
- **Feature Engineering:** Created a `Tenure` feature (employee's length of service) from `Date of Joining` and dropped the original `Date of Joining` and `Employee ID` columns.
- **Feature Scaling:** Applied `StandardScaler` to all numerical features to normalize their ranges, ensuring optimal model performance.
- **Data Splitting:** Divided the training data into training and validation sets (80/20 split) for robust model evaluation.

### 4. Model Training and Evaluation
Trained and evaluated several regression models to predict `Burn Rate`:
- **Linear Regression (Baseline):** Established a foundational performance benchmark.
- **Random Forest Regressor:** Explored an ensemble tree-based model for improved accuracy.
- **Gradient Boosting Regressor:** Utilized another powerful boosting ensemble technique.
- **XGBoost Regressor:** Implemented an optimized gradient boosting framework.

### 5. Hyperparameter Tuning
Performed hyperparameter tuning on the **Gradient Boosting Regressor** (which showed promising initial results) using `GridSearchCV` (or `RandomizedSearchCV` if computational resources were a concern) to find the optimal set of parameters.

## Key Findings and Model Performance

The **Tuned Gradient Boosting Regressor** emerged as the best-performing model, achieving:
- **R-squared (R2):** 0.8597
- **Mean Absolute Error (MAE):** 0.0523
- **Root Mean Squared Error (RMSE):** 0.0711

This indicates that the model explains almost 86% of the variance in employee burnout rates and provides highly accurate predictions.

## Feature Importances

Analysis of the best model's feature importances revealed:
- **Mental Fatigue Score** is overwhelmingly the most significant predictor (approx. 88% importance).
- **Resource Allocation** is the second most important factor (approx. 10% importance).
- Other features like `Designation`, `Tenure`, `WFH Setup Available`, `Gender`, and `Company Type` contribute to a much lesser extent.

This highlights that interventions focused on managing mental fatigue and optimizing resource allocation would be most impactful in mitigating employee burnout.

## Deployment (Streamlit)

This project is set up for deployment using Streamlit. The application will allow users to input employee features and receive a predicted burnout rate.

**Files for Deployment:**
- `app.py`: The Streamlit application script.
- `requirements.txt`: Lists all necessary Python libraries.
- `best_gb_model.pkl`: The saved trained Gradient Boosting model.
- `scaler.pkl`: The saved `StandardScaler` object used for feature scaling.

### How to Deploy on Streamlit Cloud (using your GitHub Repository)

1.  **Save/Download Files:** Ensure you have `app.py`, `requirements.txt`, `best_gb_model.pkl`, and `scaler.pkl` downloaded from Google Colab.
2.  **Create GitHub Repository:** Create a new public GitHub repository.
3.  **Upload Files:** Upload all the mentioned files to the root directory of your GitHub repository.
4.  **Login to Streamlit Cloud:** Go to [Streamlit Cloud](https://share.streamlit.io/) and log in with your GitHub account.
5.  **Deploy App:**
    * Click "New app" or "Deploy an app."
    * Select your GitHub repository and the branch where your files are located.
    * Set "Main file path" to `app.py`.
    * Click "Deploy!"

Your Streamlit app should then build and deploy, providing you with a public URL.

## Limitations and Future Work
- The model's generalizability might be limited to similar organizational contexts.
- Only available features were used; incorporating additional HR data (e.g., salary, team size, manager feedback) could further enhance prediction accuracy.
- Exploring causal relationships beyond correlation could provide deeper insights.
- The dataset's age might limit its relevance to current work trends.

## Contact
[EmeditWeb](https://github.com/EmeditWeb)
