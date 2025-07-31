import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

X['Sex'] = X['Sex'].map({'male': 1, 'female': 0})
X['Age'].fillna(X['Age'].mean(), inplace=True)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

st.title("Titanic Survival Prediction")
st.write("Enter the passenger details to predict survival")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Gender = st.selectbox("Gender",["male","female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
Fare = st.slider("Fare", 0.0, 2500.0,10.0)

if st.button("Predict"):
        row = pd.DataFrame({
            'Pclass': [Pclass],
            'Sex': [1 if Gender == "male" else 0],
            'Age': [Age],
            'SibSp': [SibSp],
            'Parch': [Parch],
            'Fare': [Fare]
        })

        prediction = model.predict(row)
        if prediction[0] == 1:
            st.success("🎉 The passenger **would have survived**!")
        else:
            st.error("💀 The passenger **would not have survived.**")
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")



