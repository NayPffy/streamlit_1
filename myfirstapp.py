import streamlit as st
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Cargamos la información a utilizar
df = pd.read_csv('./in/income.csv')
st.write(df.columns)

st.image('./pig.jpg')

siteHeader = st.beta_container()
with siteHeader:
    # Título del dashboard
    st.title('Modelo de evaluación de ingresos')
    st.markdown("""El objetivo de este proyecto es proveer una herramienta que nos **permita predecir sí una persona ganará más o menos de $50K anuales**.\n""")

    
dataExploration = st.beta_container()
with dataExploration:
    # Subtítulo dentro del dashboard
    st.subheader('Dataset: Ingresos')
    st.text('Para el desarrollo del modelo utilizaremos una transformación del siguiente set de datos')
    st.dataframe(df.head())


dataViz = st.beta_container()
with dataViz:
    st.subheader('Exploración de la data:')
    st.text('Distribución de los datos con respecto al sexo.')
    st.area_chart(df.sex.value_counts())
    st.text('Distribución de los datos con respecto a la edad.')
    st.bar_chart(df.age.value_counts())

#histogram
# fig, ax = plt.subplots()
#bins = np.linspace(10, 90, 90)
#data = pd.DataFrame(df, columns = ['age'])
#ax.hist(data,bins)
#plt.title('Distribución de registros por Edad')
#st.pyplot(fig)

# st.checkbox(list(df.columns))    
# 'race','sex','workclass','education',    
    
newFeatures = st.beta_container()
with newFeatures:
    st.subheader('Selección de Variables: ')
    st.markdown('De manera inicial, el modelo trabaja con las variables: **race, sex, workclass y education**')
    st.text('¿Quieres considerar alguna otra variable? ¡Selecciona las que quieras!')

# st.multiselect('sex',[1,2,3,4])

optional_cols = ['education-num','marital-status','occupation','relationship']
options = st.multiselect('Variables que se añadirán al modelo:',
     optional_cols)

principal_columns = ['race','sex','workclass','education']
drop_columns = ['income','fnlwgt','capital-gain','capital-loss','native-country','income_bi']

if len(options) != 0:
    principal_columns = principal_columns + options
    drop_columns =drop_columns +[i for i in optional_cols if i not in options]
else:
    drop_columns = drop_columns + optional_cols

    
modelTraining = st.beta_container()
with modelTraining:
    st.subheader('Entrenamiento del modelo')
    st.text('En esta sección puedes hacer una selección de los hiperparámetros del modelo.')

# Definimos nuestras variables:   
Y = df['income_bi']
df = df.drop(drop_columns, axis=1)
X = pd.get_dummies(df, columns = principal_columns)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 15)

# Slider
max_depth = st.slider ('¿Cuál debería ser el valor de max_depth para el modelo?', min_value=1, max_value=10, value=2, step=1)

# Modelo
t = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=max_depth)
model = t.fit(x_train, y_train)

# Performance del modelo
score_train = model.score(x_train, y_train)
score_test = model.score(x_test, y_test)

Performance = st.beta_container()
with Performance:
    st.subheader('Performance del Modelo')
    col1, col2 = st.beta_columns(2)
    with col1:
        st.text('Train Score:') 
        st.text(round(score_train*100,2))
    with col2:
        st.text('Test Score:') 
        st.text(round(score_test*100,2))

st.markdown( """ <style>
 .main {
 background-color: #AF9EC;
}
</style>""", unsafe_allow_html=True )