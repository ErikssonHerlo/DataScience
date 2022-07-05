import string
from tokenize import Number
from sklearn.neural_network import MLPClassifier
import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree

def getOperator(valor: float) -> str:
    operador = "+ " if valor>=0 else "" 
    return operador

st.title("Arboles de Decisión")
st.write("""
La Regresión Polinomial es un caso especial de la Regresión Lineal, extiende el modelo lineal al agregar predictores adicionales, obtenidos al elevar cada uno de los predictores originales a una potencia. Por ejemplo, una regresión cúbica utiliza tres variables, como predictores. Este enfoque proporciona una forma sencilla de proporcionar un ajuste no lineal a los datos.
""")
st.write("""
El método estándar para extender la Regresión Lineal a una relación no lineal entre las variables dependientes e independientes, ha sido reemplazar el modelo lineal con una función polinomial.
""")
st.write("""
Por su parte, la ecuación general correspondiente a un modelo de regresión polinomial es:
""")
st.latex("Y=β0+β1Xi+βnXi^n+ϵi")

st.write("""
Como se puede observar para la Regresión Polinomial se crean algunas características adicionales que no se encuentran en la Regresión Lineal.

Un término polinomial, bien sea cuadrático o cúbico, convierte un modelo de regresión lineal en una curva, pero como los datos de “X” son cuadráticos o cúbicos pero el coeficiente “b” no lo es, todavía se califican como un modelo lineal.

Esto hace que sea una forma agradable y directa de modelar curvas sin tener que modelar modelos complicados no lineales.
""")

st.subheader("Carga del Archivo")
st.write("""
Para realizar un análisis de regresión líneal, es necesario cargar un archivo de datos, con un formato específico. Estos pueden ser archivos con extensiones: csv, xls,xlsx o json.
""")

uploadFile = st.file_uploader("Elija un archivo", type=['csv', 'xls', 'xlsx', 'json'])
if(uploadFile is not None):

    splitName = os.path.splitext(uploadFile.name)
    
    fileName = splitName[0]
    fileExtension = splitName[1]

    # Verificamos la extension del Archivo, para su lectura correspondiente
    if(fileExtension == ".csv"):
        df = pd.read_csv(uploadFile)
    elif(fileExtension == ".xls" or fileExtension == ".xlsx"):
        df = pd.read_excel(uploadFile)
    elif(fileExtension == ".json"):
        df = pd.read_json(uploadFile)

    # Imprimimos el contenido de la tabla
    st.markdown("#### Contenido del Archivo")
    st.dataframe(df)

    st.subheader("Parametrización")
    st.write("""
        Elija las variables que se utilizarán para el análisis del clasificador gaussiano """)
    st.markdown("#### Variable Objetivo")
    var_Object = st.selectbox("Por favor elija una opción", df.keys(), key="variableObjetivo")

    
    size = len(df.keys()) -1

    
    st.markdown("#### Valor de la Predicción")
    predValues = st.text_input(f"Ingrese los {size} valores de la predicción seguidos de una coma")
   
    #Transformar Data a Array
    field_y = df[var_Object]
    df = df.drop([var_Object], axis = 1)
    col_match = [s for s in df.head() if "NO" in s]
    if len(col_match) == 1: df = df.drop(['NO'], axis = 1)

    # Division de columnas
    fields_x = []
    le = preprocessing.LabelEncoder()
    headers = df.head()
    columns = headers.columns

    # Construccion de las tuplas
    for col in columns:
        col_list = df[col].tolist()
        print(col_list)
        col_trans = le.fit_transform(col_list)
        print(col_trans)
        fields_x.append(col_trans)
        print("---------- ----------")
    print(fields_x)


    # Agregamos el arreglo de tuplas a la lista
    features = list(zip(*fields_x))

    #Entrenamos el modelo
    mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500, alpha=0.0001,
    solver="adam", random_state = 21, tol = 0.000000001)
    mlp.fit(features,field_y)
    
    

    
    #predict = model.predict(predValues)
    predict = mlp.predict([[2, 1, 0, 0]])

    #Obtenemos la imagen para mostrarla
    
    if st.button('Calcular'):
        
        #st.subheader("Predicción")
        #indicadorPrediccion = "+ Positiva" if predict>=0 else "- Negativa"
        #st.metric(f"El valor de la predicción para los valores ingresados es de: ",str(predict))
        st.subheader("Predicción")
        #indicadorPrediccion = "+ Positiva" if predict>=0 else "- Negativa"
        st.metric(f"El valor de la predicción para los valores ingresados es de: ",str(predict))




        
    


else:
    st.warning("Debe Cargar un Archivo Previamente")
   


st.sidebar.title("Indice")
st.sidebar.markdown("### [Carga del Archivo](#carga-del-archivo)")
st.sidebar.markdown("- [Contenido del Archivo](#contenido-del-archivo)")
st.sidebar.markdown("### [Parametrización](#parametrizaci-n)")
st.sidebar.markdown("- [Variable Indepentiente (X)](#variable-independiente-x)")
st.sidebar.markdown("- [Variable Depentiente (Y)](#variable-dependiente-y)")
st.sidebar.markdown("- [Grado de la Función](#grado-de-la-funci-n)")
st.sidebar.markdown("- [Valor de la Predicción](#valor-de-la-predicci-n)")
st.sidebar.markdown("- [Colores de la Gráfica](#colores-de-la-gr-fica)")
st.sidebar.markdown("### [Graficación](#graficaci-n)")
st.sidebar.markdown("- [Datos de la Gráfica](#datos-de-la-gr-fica)")
st.sidebar.markdown("### [Función de la Tendencia](#funci-n-de-la-tendencia)")
st.sidebar.markdown("### [Predicción](#predicci-n)")


