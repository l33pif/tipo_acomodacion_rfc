import streamlit as st
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#Escribe un encabezado
st.write("""
            # Estimador de acomodacion

            **Objetivo**: Descubrir patrones en el viajero y en la estadía que permitan predecir el tipo de acomodación para futuras reservas.
           
            --- 
            """)
#Encabezado de sidebar
st.sidebar.header('User Input Parameters')

#Se usa cuando el archivo era .txt para convertitloa a.csv 
with open('train_data.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('train_d.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('c'))
        writer.writerows(lines)

#Limpiando datos del archivo de entrenamiento
train_data = pd.read_csv('train_d.csv')
train_data = train_data['c'].str.replace(r'\[REG\]', '')

train = pd.DataFrame(train_data.values.reshape((-1,8)))
train.columns=['registro','id','duracion_estadia','genero',	'edad','niños','codigo_destino','tipo_acomodacion']
train.drop(columns='registro', inplace=True)

genero = {' M': 0, ' F': 1}
train.genero = [genero[item] for item in train.genero]
acomo = {' AirBnB': 0, ' Hotel': 1}
train.tipo_acomodacion = [acomo[item] for item in train.tipo_acomodacion]
dest = {' <NA>': 0, ' AR': 1, ' COL': 2, ' ES': 3, ' IT': 4, ' NL': 5, ' PE': 6, ' UK': 7, ' US': 8}
train.codigo_destino = [dest[item] for item in train.codigo_destino]

#Transformando datos del archivo de entrenamiento
train['duracion_estadia'] = train['duracion_estadia'].astype(int)
train['edad'] = train['edad'].str.replace(' <NA>', '-1')
train['edad'] = train['edad'].astype(int)
train['edad'] = train['edad'].replace(-1, np.nan)
train['edad'].fillna(train['edad'].mean(), inplace=True)
train['edad'] = train['edad'].astype(int)
train['niños'] = train['niños'].str.replace(' <NA>', '0')
train['niños'] = train['niños'].astype(int)

#Relacion de datos del archivo de entrenamiento
df0 = train[train.tipo_acomodacion==0]
df1 = train[train.tipo_acomodacion==1]

st.write("""

            ## Proceso de entrenamiento

            En la siguiente grafica se muestra la relacion entre las personas que se hospedan en: 
            **Airbnb** con color *verde* y en **Hotel** de color *rojo*

            Dado el data set: [train.csv](https://www.mediafire.com/file/uln8erwqs17jptb/train_d.csv/file)
            """)

#Graficar la relacion de datos del archivo de entrenamiento
fig = plt.figure(figsize=(12,9))
ax = Axes3D(fig)
ax.set_xlabel('Duracion de estancia')
ax.set_ylabel('Edades')
ax.set_zlabel('Destino')
ax.scatter(df0['duracion_estadia'], df0['edad'], df0['codigo_destino'], zdir='z', c='g', marker='.')
ax.scatter(df1['duracion_estadia'], df1['edad'], df1['codigo_destino'], zdir='z', c='r', marker='d')
st.pyplot(fig)

#Modelar
X = train.drop(['tipo_acomodacion', 'id'], axis='columns')
y = train['tipo_acomodacion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, train_size=0.85)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred_forest = clf.predict(X_test)
acc_forest = metrics.accuracy_score(y_test, y_pred_forest)

# st.subheader('Tipo de acomodacion 0-Airbnb')
# st.subheader('Tipo de acomodacion 1-Airbnb')
# st.write(train.tipo_acomodacion)
st.subheader('Precision del modelo Random Forest Classifier:')
st.write(acc_forest)


#Probando con datos externos
st.write("""
        ---

        ## Implementando el modelo de entrenamiento en el dataset adicional: [DataAcomodacion.csv](https://www.mediafire.com/file/xdeguhvw2sfa16a/DataAcomodacion.csv/file)

        """)

#Usando dataset adicional 
acomodacion = pd.read_csv('DataAcomodacion.csv', encoding='latin')

#Limpiando y transformando los datos
genero = {'M': 0, 'F': 1}
acomodacion.genero = [genero[item] for item in acomodacion.genero]
dest = {np.nan: 0, 'AR': 1, 'COL': 2, 'ES': 3, 'IT': 4, 'NL': 5, 'PE': 6, 'UK': 7, 'US': 8}
acomodacion.codigo_destino = [dest[item] for item in acomodacion.codigo_destino]

X_ext = acomodacion.drop(['tipo_acomodacion', 'id'], axis='columns')

X_ext['edad'].fillna(X_ext['edad'].mean(), inplace=True)
X_ext['edad'] = X_ext['edad'].astype(int)

X_ext['niños'].fillna(0, inplace=True)
X_ext['niños'] = X_ext['niños'].astype(int)

tipo_aco = clf.predict(X_ext)

acomodacion['tipo_acomodacion'] = tipo_aco


#Relacion
dfaco0 = acomodacion[acomodacion.tipo_acomodacion==0]
dfaco1 = acomodacion[acomodacion.tipo_acomodacion==1]

st.write("""

            En la siguiente grafica se muestra la relacion entre las personas que se hospedan en: 
            **Airbnb** con color *verde* y en **Hotel** de color *rojo* del dataset adicional.
            """)

#Graficar
fig_aco = plt.figure(figsize=(12,9))
axac = Axes3D(fig_aco)
axac.set_xlabel('Duracion de estancia')
axac.set_ylabel('Edades')
axac.set_zlabel('Destino')
axac.scatter(dfaco0['duracion_estadia'], dfaco0['edad'], dfaco0['codigo_destino'], zdir='z', c='g', marker='.')
axac.scatter(dfaco1['duracion_estadia'], dfaco1['edad'], dfaco1['codigo_destino'], zdir='z', c='r', marker='d')
st.pyplot(fig_aco)

st.write("""
            En la siguiente dataframe se muestra el resultado del analisis en el dataset adicional.
            """)

acomod = {0:'AirBnB', 1:'Hotel'}
acomodacion.tipo_acomodacion = [acomod[item] for item in acomodacion.tipo_acomodacion]
acomodacion.to_csv('acomodacion_final.csv')

st.write(acomodacion)

st.write("""---""")
st.write("""---""")
st.write("""---""")

#Contenido de sidebar
def user_input_parameters():
    duracion_estadia = st.sidebar.number_input('Duracion', min_value=1, max_value=12)
    genero = st.sidebar.selectbox('Genero', ['M', 'F'])
    edad = st.sidebar.slider('Edad', min_value=25, max_value=60)
    niños = st.sidebar.selectbox('Niños', ['Si', 'No'])
    codigo_destino = st.sidebar.selectbox('Destino', ['AR', 'COL', 'ES', 'IT', 'NL', 'PE', 'UK', 'US'])
    data = {'duracion_estadia': duracion_estadia,
            'genero': genero,
            'edad': edad,
            'niños': niños,
            'codigo_destino': codigo_destino,
            }
    features = pd.DataFrame(data, index=[0])
    return features

#Body de la app
df = user_input_parameters()
st.subheader('User input parameters')
st.write(df)
st.subheader('Prediccion')
#prediccion conforme al input
input_df = pd.DataFrame(df)

genero = {'M': 0, 'F': 1}
input_df.genero = [genero[item] for item in input_df.genero]
dest = {np.nan: 0, 'AR': 1, 'COL': 2, 'ES': 3, 'IT': 4, 'NL': 5, 'PE': 6, 'UK': 7, 'US': 8}
input_df.codigo_destino = [dest[item] for item in input_df.codigo_destino]
nin = {'Si': 1, 'No': 0}
input_df.niños = [nin[item] for item in input_df.niños]

#X_input = input_df.drop(['tipo_acomodacion'], axis='columns')
input_df['edad'] = input_df['edad'].astype(int)


tipo_acom = clf.predict(input_df)

input_df['tipo_acomodacion'] = tipo_acom

acomoda = {0:'AirBnB', 1:'Hotel'}
input_df.tipo_acomodacion = [acomoda[item] for item in input_df.tipo_acomodacion]
st.write(input_df['tipo_acomodacion'])
st.write("""---""")
