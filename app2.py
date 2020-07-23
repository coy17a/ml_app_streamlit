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
    st.title("Binary Classification Web App")
    st.sidebar.title('Model Parameters')
    st.sidebar.markdown('Are you musroom edible or posisson? ')

    @st.cache(persist=True)
    def load_Data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data
    @st.cache(persist=True)
    def split(df):
        y=df.type
        x=df.drop(columns='type')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        return x_train,x_test,y_train,y_test
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names) 
            st.pyplot()
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test) 
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model,x_test,y_test) 
            st.pyplot()
    
    df = load_Data()
    x_train,x_test,y_train,y_test = split(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Select the Classifier")
    classifier = st.sidebar.selectbox('Type',("SVM",'Logistic Regresion','Random Forest'))
    
    if classifier == 'SVM':
        st.sidebar.subheader('Model Paramaters')
        c=st.sidebar.number_input("C(Regularizaiton Parameter)",0.01,10.00,step=0.1, key='C')
        kernel=st.sidebar.selectbox('Kernel',('rbf','linear'),key='kernel')
        gamma = st.sidebar.radio('Gamma',('scale','auto'),key='gamma')
        metrics= st.sidebar.multiselect("What metric you want to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("classify", key = "classify"):
            model = SVC(C=c,kernel = kernel, gamma = gamma)
            model.fit(x_train,y_train)
            y_pred =model.predict(x_test)
            plot_metrics(metrics)
    #Logist Regression
    if classifier == 'Logistic Regresion':
        st.sidebar.subheader('Model Paramaters')
        max_iter=st.sidebar.slider('Max Iterations',min_value=100,max_value=500,value=250,key='max_iter')
        metrics= st.sidebar.multiselect("What metric you want to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        #if st.sidebar.button("classify", key = "classify"):
        c=st.sidebar.number_input("C(Regularizaiton Parameter)",0.01,10.00,step=0.1, key='C')
        model = LogisticRegression(C=c,max_iter = max_iter)
        model.fit(x_train,y_train)
        y_pred =model.predict(x_test)
        plot_metrics(metrics)
    #Random Froest
    if classifier == 'Random Forest':
        st.sidebar.subheader('Model Paramaters')
        c=st.sidebar.number_input("C(Regularizaiton Parameter)",0.01,10.00,step=0.1, key='C')
        kernel=st.sidebar.selectbox('Kernel',('rbf','linear'),key='kernel')
        gamma = st.sidebar.radio('Gamma',('scale','auto'),key='gamma')
        metrics= st.sidebar.multiselect("What metric you want to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("classify", key = "classify"):
            model = SVC(C=c,kernel = kernel, gamma = gamma)
            model.fit(x_train,y_train)
            y_pred =model.predict(x_test)
            plot_metrics(metrics)   

        



    # if st.sidebar.checkbox("Show raw data",False):
    #    st.subheader('Mushroom Data Set')
    #    st.write(df)
        
        




if __name__ == '__main__':
    main()