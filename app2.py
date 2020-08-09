import streamlit as st
import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from PIL import Image
#load loggo
image = Image.open('logo.png')
st.sidebar.image(image,use_column_width=True)
def main():
    st.title("Binary Classification Web App")
    nav = st.sidebar.radio('Navigation',['Home','Classify'])
    if nav == 'Classify':
        st.markdown(''' - In the sidebar the model for binary classification
- Choose the value for the hyperparameter for the model selected
- Choose metric for plottingS
- Click Classify buttom''')
        st.sidebar.title('Model Parameters')
        st.sidebar.markdown('Are you mushroom edible or posisson? ')

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
            if st.sidebar.button("classify", key = "classify"):
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
    else:
        st.markdown('''## Interactive Machine Learning Training with Streamlit

This applicaiton is my first attemp to use streamlite to deploy an interactive machine learnign app. The main purpose is to work with a simple dataset and study the performance of the different ML algorithms 

This app is design to:
- Create an interact interface for hyperparamater tunning
- Compare model performance using most common performance metrics.

## How To Use ?

- In the sidebar the model for binary classification
- Choose the value for the hyperparameter for the model selected
- Choose metric for plotting
- Click Classify buttom''')

        
    st.sidebar.info(
        """This an interactive app created to explore streamlit framework
- If you any question or comments please send me an [email](mailto:alejocoy17a@gmail.com)
- Alejandro Coy &copy 2020
- [Github Repo](https://github.com/coy17a/ml_app_streamlit)"""  )
  

    # if st.sidebar.checkbox("Show raw data",False):
    #    st.subheader('Mushroom Data Set')
    #    st.write(df)
        
        




if __name__ == '__main__':
    main()