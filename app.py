import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.sidebar.title("Binary Classification Web App")
    st.sidebar.markdown("Are your mushrooms edible or Poissiones? 🍄")
    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col]= label.fit_transform(data[col])
        return data
    

    @st.cache(persist = True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test
    df = load_data()
    x_train,x_test,y_train,y_test = split(df)

    if st.sidebar.checkbox("Show raw Data", False):
        st.subheader("mushroom data set (classification")
        st.write(df)
    




if __name__ == '__main__':
    main()


