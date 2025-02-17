# ANN Classification - Churn Prediction

## Overview
This project demonstrates an end-to-end implementation of an Artificial Neural Network (ANN) model for churn prediction using the Churn dataset. It covers feature engineering, ANN model training, optimization, and deployment using Streamlit.

## Dataset
The dataset consists of the following features:
- **RowNumber, CustomerId, Surname** (Ignored in modeling)
- **CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary** (Numerical features)
- **Geography, Gender** (Categorical features converted to numerical values)
- **Exited** (Target variable)

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
2. Train the ANN model:
   ```python
   python train_model.py
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Conclusion
This project provides a hands-on approach to developing, training, optimizing, and deploying an ANN model for churn prediction. The step-by-step implementation ensures a smooth transition from data preprocessing to model deployment in a real-world scenario.

