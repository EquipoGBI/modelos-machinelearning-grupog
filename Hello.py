import streamlit as st

st.set_page_config(
    page_title="Tarea Semana 12",
    page_icon="👋",
)

st.write("# Despliegue web de modelos del Grupo G 🤖")

st.sidebar.success("Seleccione un modelo del menú")

st.markdown(
    """
    # Grupo G - Integrantes:
    | Nombre | Participación|
    |--|--|
    | Oscar Stalyn, Yanfer Laura | Regresión líneal (RL) |
    | Diego Tharlez Montalvo Ortega | Support Vector Regression (SVR) |
    | Jorge Luis Quispe Alarcon | Twitter |
    | Wilker Edison,Atalaya Ramirez | Máquinas de vectores de soporte (SVM) |
    | Anthony Elias,Ricse Perez | Red Neuronal Recurrente(RNN) |
    | Carlos Daniel Tarmeño Noriega | K-Vecinos Cercanos(KNN) |
    | Nathaly Nicole Pichilingue Pimentel | Máquinas de vectores de soporte(SVC) y Random Forest(RF) |
    | Jorge Luis, Marin Evangelista | Redes Neuronales Bayesianas (RNB) |

    ### Especificaciones:
    **Donde muestra las predicciones/los resultados:**
    - Gráficamente. 
    - Númericamente los valores de las predicciones (print de dataframe con la predicción o clasificación).
    
    **Donde se muestra el EDA:**
    - Ploteo de los precios reales.
    (Ploteo de media móvil los precios reales.)

    **Donde el usuario pueda indicar:**
    - El modelo ejecutar.
    - La acción o instrumento financiero que quiera analizar.
"""
)
