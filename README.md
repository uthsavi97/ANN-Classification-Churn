# ANN-Classification-Churn Modelling

## Project Overview
This project demonstrates an end-to-end implementation of **Artificial Neural Networks (ANN)** for **Churn Prediction** using the **Churn Modelling dataset**. The process includes **data preprocessing, feature engineering, model building, training, evaluation, and deployment** using Streamlit.

## Dataset Overview
The dataset consists of customer data from a bank, with the aim of predicting whether a customer will **churn** (exit) or not.

### Features:
- **RowNumber, CustomerId, Surname**: Identification columns (Dropped as they do not contribute to prediction)
- **CreditScore**: Customer credit score
- **Geography**: Categorical feature (Converted to numerical)
- **Gender**: Categorical feature (Converted to numerical)
- **Age**: Age of the customer
- **Tenure**: Number of years with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products used
- **HasCrCard**: Whether the customer has a credit card (0/1)
- **IsActiveMember**: Whether the customer is active (0/1)
- **EstimatedSalary**: Salary estimate
- **Exited**: Target variable (1 = Churn, 0 = Retain)

## Steps Involved

### 1. Data Preprocessing & Feature Engineering
- Convert categorical variables (Geography, Gender) into numerical format using encoding.
- Standardize numerical features for better ANN training.
- Split the dataset into training and testing sets.

### 2. Building the ANN Model
- Input layer: Standardized numerical features.
- Hidden layers: Experimenting with different numbers of layers and neurons.
- Dropout layers: Prevent overfitting.
- Output layer: Predicting churn probability (binary classification).
- Loss function: Binary cross-entropy.
- Optimizer: Adam.

### 3. Training the ANN Model
- Step-by-step training with TensorFlow/Keras.
- Evaluation of model performance.
- Hyperparameter tuning for optimal architecture.

### 4. Model Optimization
- Determining the optimal number of hidden layers and neurons:
  - Start with a simple architecture and gradually increase complexity.
  - Use Grid Search or Random Search.
  - Perform cross-validation to evaluate different architectures.
  - Apply heuristics, such as ensuring the number of neurons in hidden layers is between the input and output layer sizes.

### 5. Saving & Loading the Model
- The trained ANN model is saved using **pickle** for later use.

### 6. Deploying ANN Model with Streamlit
- Integrating the trained ANN model into a **Streamlit** web application.
- Building an interactive UI for user input and model prediction.
- Deploying the web app on **Streamlit Cloud**.

## Technologies Used
- **tensorflow==2.18.0**
- **pandas**
- **numpy**
- **scikit-learn**
- **tensorboard**
- **matplotlib**
- **streamlit**
- **scikeras**

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Conclusion
This project provides a hands-on approach to developing, training, optimizing, and deploying an ANN model for churn prediction. The step-by-step implementation ensures a smooth transition from data preprocessing to model deployment in a real-world scenario.

