import streamlit as st

siteHeader = st.beta_container()
with siteHeader:
    st.title('Modelo de evaluación de ingresos')
    st.markdown(""" En este proyecto se busca diseñar un modelo que nos permita identificar cuáles son las características principales que nos ayuden a predecir sí  una persona ganará más o menos de $50K anuales.""")