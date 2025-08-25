import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("🧠 Student Mental Health - Depression Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    st.write("Null Values in Dataset:")
    st.write(df.isnull().sum())

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Sidebar for Prediction Mode
    mode = st.sidebar.radio("Choose Prediction Type:", 
                             ["Depression Score (0-5)", "Depression Risk (0 or 1)"])

    if mode == "Depression Score (0-5)":
        st.subheader("Prediction: Depression Score (0-5)")
        X = df.drop('Depression_Score', axis=1)
        y = df['Depression_Score']

    else:
        st.subheader("Prediction: Depression Risk (0 = Low, 1 = High)")
        df['Depression_Risk'] = df['Depression_Score'].apply(lambda x: 1 if x >= 2 else 0)
        X = df.drop(columns=['Depression_Score', 'Anxiety_Score', 'Stress_Level', 'Depression_Risk'])
        y = df['Depression_Risk']

    # Encode categorical variables
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Scale features
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with GridSearch
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    st.write("Training the model... please wait.")
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid, cv=5, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # Plot Confusion Matrix Heatmap
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.colorbar(ax.imshow(cm, interpolation='nearest', cmap='Blues'))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("Actual vs Predicted")
    fig2, ax2 = plt.subplots()
    jitter = 0.1
    y_test_jitter = y_test + np.random.uniform(-jitter, jitter, size=y_test.shape)
    y_pred_jitter = y_pred + np.random.uniform(-jitter, jitter, size=y_pred.shape)
    ax2.scatter(y_test_jitter, y_pred_jitter, alpha=0.7, color='blue')
    ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    st.pyplot(fig2)

else:
    st.warning("Please upload a CSV file to proceed.")
