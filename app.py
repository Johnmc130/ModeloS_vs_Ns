import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def convert_numpy_types(obj):
    """Convierte tipos numpy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Configuración de la página
st.set_page_config(
    page_title="Aplicación de Machine Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 Aplicación de Modelos de Machine Learning")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🔧 Panel de Control")
mode = st.sidebar.radio(
    "Selecciona el modo:",
    ["🏠 Inicio", "📊 Modo Supervisado", "🔍 Modo No Supervisado", "📁 Zona de Exportación"]
)

# Función para cargar datasets
@st.cache_data
def load_dataset(dataset_name):
    """Carga diferentes datasets según la selección"""
    
    # Diccionarios de traducción para características
    iris_features_es = {
        'sepal length (cm)': 'Longitud Sépalo (cm)',
        'sepal width (cm)': 'Anchura Sépalo (cm)',
        'petal length (cm)': 'Longitud Pétalo (cm)',
        'petal width (cm)': 'Anchura Pétalo (cm)'
    }
    
    wine_features_es = {
        'alcohol': 'Alcohol',
        'malic_acid': 'Ácido Málico',
        'ash': 'Cenizas',
        'alcalinity_of_ash': 'Alcalinidad de las Cenizas',
        'magnesium': 'Magnesio',
        'total_phenols': 'Fenoles Totales',
        'flavanoids': 'Flavonoides',
        'nonflavanoid_phenols': 'Fenoles No Flavonoides',
        'proanthocyanins': 'Proantocianinas',
        'color_intensity': 'Intensidad del Color',
        'hue': 'Tono',
        'od280/od315_of_diluted_wines': 'OD280/OD315 de Vinos Diluidos',
        'proline': 'Prolina'
    }
    
    breast_cancer_features_es = {
        'mean radius': 'Radio Promedio',
        'mean texture': 'Textura Promedio',
        'mean perimeter': 'Perímetro Promedio',
        'mean area': 'Área Promedio',
        'mean smoothness': 'Suavidad Promedio',
        'mean compactness': 'Compacidad Promedio',
        'mean concavity': 'Concavidad Promedio',
        'mean concave points': 'Puntos Cóncavos Promedio',
        'mean symmetry': 'Simetría Promedio',
        'mean fractal dimension': 'Dimensión Fractal Promedio',
        'radius error': 'Error del Radio',
        'texture error': 'Error de Textura',
        'perimeter error': 'Error del Perímetro',
        'area error': 'Error del Área',
        'smoothness error': 'Error de Suavidad',
        'compactness error': 'Error de Compacidad',
        'concavity error': 'Error de Concavidad',
        'concave points error': 'Error de Puntos Cóncavos',
        'symmetry error': 'Error de Simetría',
        'fractal dimension error': 'Error de Dimensión Fractal',
        'worst radius': 'Radio Peor',
        'worst texture': 'Textura Peor',
        'worst perimeter': 'Perímetro Peor',
        'worst area': 'Área Peor',
        'worst smoothness': 'Suavidad Peor',
        'worst compactness': 'Compacidad Peor',
        'worst concavity': 'Concavidad Peor',
        'worst concave points': 'Puntos Cóncavos Peor',
        'worst symmetry': 'Simetría Peor',
        'worst fractal dimension': 'Dimensión Fractal Peor'
    }
    
    if dataset_name == "Flores Iris":
        data = load_iris()
        feature_names_es = [iris_features_es.get(name, name) for name in data.feature_names]
        df = pd.DataFrame(data.data, columns=feature_names_es)
        df['target'] = data.target
        target_names_es = ['Setosa', 'Versicolor', 'Virginica']
        df['target_names'] = df['target'].map({i: name for i, name in enumerate(target_names_es)})
        return df, feature_names_es, target_names_es
    
    elif dataset_name == "Vinos":
        data = load_wine()
        feature_names_es = [wine_features_es.get(name, name) for name in data.feature_names]
        df = pd.DataFrame(data.data, columns=feature_names_es)
        df['target'] = data.target
        target_names_es = ['Clase 0', 'Clase 1', 'Clase 2']
        df['target_names'] = df['target'].map({i: name for i, name in enumerate(target_names_es)})
        return df, feature_names_es, target_names_es
    
    elif dataset_name == "Cáncer de Mama":
        data = load_breast_cancer()
        feature_names_es = [breast_cancer_features_es.get(name, name) for name in data.feature_names]
        df = pd.DataFrame(data.data, columns=feature_names_es)
        df['target'] = data.target
        target_names_es = ['Maligno', 'Benigno']
        df['target_names'] = df['target'].map({i: name for i, name in enumerate(target_names_es)})
        return df, feature_names_es, target_names_es

# Variables de sesión para mantener estado
if 'supervised_model' not in st.session_state:
    st.session_state.supervised_model = None
if 'unsupervised_model' not in st.session_state:
    st.session_state.unsupervised_model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'supervised_metrics' not in st.session_state:
    st.session_state.supervised_metrics = {}
if 'unsupervised_metrics' not in st.session_state:
    st.session_state.unsupervised_metrics = {}

# PÁGINA DE INICIO
if mode == "🏠 Inicio":
    st.markdown("## 📋 Descripción de la Aplicación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Modelos Implementados
        
        **Modelo Supervisado:**
        - **Gradient Boosting Classifier**
        - Algoritmo de ensemble que combina múltiples árboles de decisión
        - Excelente para clasificación con alta precisión
        
        **Modelo No Supervisado:**
        - **Isolation Forest**
        - Algoritmo para detección de anomalías
        - Identifica puntos atípicos en los datos
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Datasets Disponibles
        
        - **Flores Iris**: Clasificación de especies de flores (4 características)
        - **Vinos**: Clasificación de tipos de vino (13 características)
        - **Cáncer de Mama**: Diagnóstico médico (30 características)
        
        ### 📁 Funcionalidades
        
        - Entrenamiento interactivo de modelos
        - Evaluación con métricas estándar
        - Visualizaciones interactivas
        - Exportación a JSON y PKL
        """)
    
    # Información detallada de los datasets
    st.markdown("---")
    st.markdown("### 📋 Información Detallada de los Datasets")
    
    dataset_tab1, dataset_tab2, dataset_tab3 = st.tabs(["🌸 Flores Iris", "🍷 Vinos", "🏥 Cáncer de Mama"])
    
    with dataset_tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **📊 Características del Dataset:**
            - **Muestras:** 150 flores
            - **Características:** 4 medidas físicas
            - **Clases:** 3 especies de iris
            - **Balanceado:** Sí (50 ejemplos por clase)
            
            **🌸 Especies:**
            - Setosa
            - Versicolor  
            - Virginica
            """)
        with col_b:
            st.markdown("""
            **📏 Características Medidas:**
            - Longitud del Sépalo (cm)
            - Anchura del Sépalo (cm)
            - Longitud del Pétalo (cm)
            - Anchura del Pétalo (cm)
            
            **🎯 Ideal Para:**
            - Aprendizaje de clasificación
            - Visualización de datos
            - Comparación de algoritmos
            """)
    
    with dataset_tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **📊 Características del Dataset:**
            - **Muestras:** 178 vinos
            - **Características:** 13 análisis químicos
            - **Clases:** 3 tipos de vino
            - **Origen:** Región de Italia
            
            **🍷 Tipos:**
            - Clase 0, Clase 1, Clase 2
            - Diferentes cultivares
            """)
        with col_b:
            st.markdown("""
            **🧪 Análisis Químicos:**
            - Alcohol, Ácido Málico, Cenizas
            - Alcalinidad, Magnesio, Fenoles
            - Flavonoides, Proantocianinas
            - Intensidad del Color, Tono
            - Y más componentes químicos
            
            **🎯 Ideal Para:**
            - Clasificación multiclase
            - Análisis de componentes
            """)
    
    with dataset_tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **📊 Características del Dataset:**
            - **Muestras:** 569 diagnósticos
            - **Características:** 30 medidas morfológicas
            - **Clases:** 2 (Maligno/Benigno)
            - **Aplicación:** Diagnóstico médico
            
            **🏥 Diagnósticos:**
            - Maligno (cáncer)
            - Benigno (no cáncer)
            """)
        with col_b:
            st.markdown("""
            **🔬 Medidas Morfológicas:**
            - Radio, Textura, Perímetro
            - Área, Suavidad, Compacidad
            - Concavidad, Puntos Cóncavos
            - Simetría, Dimensión Fractal
            - Para media, error y peor caso
            
            **🎯 Ideal Para:**
            - Diagnóstico binario
            - Detección de anomalías
            - Aplicaciones médicas
            """)
    
    st.markdown("---")
    st.markdown("### 🚀 ¡Comienza seleccionando un modo en el panel lateral!")

# MODO SUPERVISADO
elif mode == "📊 Modo Supervisado":
    st.markdown("## 📊 Modelo Supervisado - Gradient Boosting")
    
    # Descripción del modelo
    st.markdown("### 🎯 ¿Qué es Gradient Boosting?")
    
    with st.expander("📚 Descripción del Algoritmo", expanded=False):
        st.markdown("""
        **Gradient Boosting** es un algoritmo de aprendizaje automático de tipo ensemble que:
        
        **Funcionamiento:**
        - Combina múltiples árboles de decisión débiles
        - Cada árbol nuevo corrige los errores del anterior
        - Utiliza gradiente descendente para minimizar la función de pérdida
        - Construye el modelo de forma secuencial
        
        **Ventajas:**
        - Alta precisión en clasificación
        - Maneja bien datos mixtos (numéricos y categóricos)
        - Robusto ante outliers
        - No requiere normalización de datos
        
        **Aplicaciones:**
        - Clasificación de imágenes
        - Diagnóstico médico
        - Detección de fraudes
        - Sistemas de recomendación
        """)
    
    # Configuración del modelo con explicaciones
    st.sidebar.markdown("### ⚙️ Configuración del Modelo")
    
    # Selección de dataset
    st.sidebar.markdown("### 📊 Selección de Dataset")
    dataset_name = st.sidebar.radio(
        "Elige un dataset:",
        ["Flores Iris", "Vinos", "Cáncer de Mama"]
    )
    
    # Cargar dataset
    if st.sidebar.button("🔄 Cargar Dataset"):
        # Limpiar modelos anteriores al cambiar dataset
        if st.session_state.dataset_name != dataset_name:
            st.session_state.supervised_model = None
            st.session_state.supervised_metrics = {}
        
        st.session_state.dataset = load_dataset(dataset_name)
        st.session_state.dataset_name = dataset_name
        st.success(f"Dataset {dataset_name} cargado exitosamente!")
    
    if st.session_state.dataset is not None:
        df, feature_names, target_names = st.session_state.dataset
        
        # Mostrar información del dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📏 Filas", df.shape[0])
        with col2:
            st.metric("📊 Columnas", df.shape[1] - 2)  # -2 por target y target_names
        with col3:
            st.metric("🎯 Clases", len(target_names))
        
        # Mostrar preview del dataset
        st.markdown("### 👀 Vista previa del dataset")
        
        # Control para número de filas a mostrar
        col1, col2 = st.columns([3, 1])
        with col2:
            num_rows = st.selectbox(
                "Filas a mostrar:",
                [5, 10, 20, 50, "Todas"],
                index=1,  # Default: 10 filas
                key="preview_rows"
            )
        
        # Mostrar dataset según selección
        if num_rows == "Todas":
            st.dataframe(df, use_container_width=True)
            st.info(f"📊 Mostrando todas las {len(df)} filas del dataset")
        else:
            st.dataframe(df.head(num_rows), use_container_width=True)
            st.info(f"📊 Mostrando las primeras {num_rows} filas de {len(df)} totales")
        
        # Preparar datos
        X = df[feature_names]
        y = df['target']
        
        # Configuración del modelo
        st.sidebar.markdown("### ⚙️ Configuración del Modelo")
        n_estimators = st.sidebar.slider("Número de estimadores", 50, 500, 100)
        max_depth = st.sidebar.slider("Profundidad máxima", 3, 10, 6)
        learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.01, 0.3, 0.1, 0.01)
        
        with st.sidebar.expander("📊 Parámetros Explicados"):
            st.markdown("""
            **Número de Estimadores:**
            - Cantidad de árboles en el ensemble
            - Más árboles = mayor precisión pero más lento
            - Rango recomendado: 100-300
            
            **Profundidad Máxima:**
            - Qué tan profundo puede crecer cada árbol
            - Mayor profundidad = más complejo
            - Evita overfitting con valores bajos
            
            **Tasa de Aprendizaje:**
            - Qué tanto contribuye cada árbol
            - Valores bajos = aprendizaje más conservador
            - Balance entre velocidad y estabilidad
            """)
        
        # Entrenar modelo
        if st.sidebar.button("🚀 Entrenar Modelo"):
            with st.spinner("Entrenando modelo..."):
                # División de datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Crear y entrenar modelo
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Predicciones
                y_pred = model.predict(X_test)
                
                # Calcular métricas
                metricas = {
                    'exactitud': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'sensibilidad': recall_score(y_test, y_pred, average='weighted'),
                    'puntuacion_f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Guardar en sesión (incluyendo los datos de entrenamiento)
                st.session_state.supervised_model = model
                st.session_state.supervised_metrics = metricas
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.current_X = X  # Guardar las características actuales
                st.session_state.current_feature_names = feature_names  # Guardar nombres de características
                st.session_state.current_target_names = target_names  # Guardar nombres de clases
                
                st.success("¡Modelo entrenado exitosamente!")
        
        # Mostrar métricas si el modelo está entrenado
        if st.session_state.supervised_model is not None:
            # Verificar si el dataset actual coincide con el modelo entrenado
            if (hasattr(st.session_state, 'current_feature_names') and 
                hasattr(st.session_state, 'dataset_name') and 
                st.session_state.dataset_name == dataset_name):
                
                st.markdown("### 📈 Métricas del Modelo")
                
                with st.expander("📊 ¿Qué significan estas métricas?", expanded=False):
                    st.markdown("""
                    **Exactitud (Accuracy):**
                    - Porcentaje de predicciones correctas sobre el total
                    - Ideal: cerca de 1.0 (100%)
                    - Útil cuando las clases están balanceadas
                    
                    **Precisión (Precision):**
                    - De las predicciones positivas, cuántas fueron correctas
                    - Evita falsos positivos
                    - Importante en diagnósticos médicos
                    
                    **Sensibilidad (Recall):**
                    - De los casos reales positivos, cuántos detectó
                    - Evita falsos negativos
                    - Crítico en detección de enfermedades
                    
                    **Puntuación F1:**
                    - Media armónica entre precisión y sensibilidad
                    - Balance entre ambas métricas
                    - útil cuando hay desbalance de clases
                    """)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Exactitud", f"{st.session_state.supervised_metrics['exactitud']:.3f}")
                with col2:
                    st.metric("🔍 Precisión", f"{st.session_state.supervised_metrics['precision']:.3f}")
                with col3:
                    st.metric("📊 Sensibilidad", f"{st.session_state.supervised_metrics['sensibilidad']:.3f}")
                with col4:
                    st.metric("⚖️ Puntuación F1", f"{st.session_state.supervised_metrics['puntuacion_f1']:.3f}")
                
                # Matriz de confusión
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
                
                # Usar los nombres de clases del modelo entrenado
                target_labels = (st.session_state.current_target_names 
                               if hasattr(st.session_state, 'current_target_names') 
                               else target_names)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=target_labels, yticklabels=target_labels)
                ax.set_title('Matriz de Confusión')
                ax.set_xlabel('Predicción')
                ax.set_ylabel('Valor Real')
                st.pyplot(fig)
                
            else:
                st.warning("⚠️ Has cambiado de dataset. Por favor, entrena nuevamente el modelo para ver métricas y hacer predicciones.")
            
            # Prueba interactiva
            st.markdown("### 🎮 Prueba Interactiva")
            st.markdown("Ajusta los valores para hacer una predicción:")
            
            # Verificar que tenemos los datos del modelo entrenado
            if (hasattr(st.session_state, 'current_feature_names') and 
                hasattr(st.session_state, 'current_X') and 
                hasattr(st.session_state, 'current_target_names')):
                
                current_features = st.session_state.current_feature_names
                current_X = st.session_state.current_X
                current_targets = st.session_state.current_target_names
                
                # Crear sliders para cada feature del modelo entrenado
                input_values = []
                
                # Organizar en columnas para mejor visualización
                n_features = len(current_features)
                n_cols = min(3, n_features)  # Máximo 3 columnas
                cols = st.columns(n_cols)
                
                for i, feature in enumerate(current_features):
                    with cols[i % n_cols]:
                        min_val = float(current_X[feature].min())
                        max_val = float(current_X[feature].max())
                        mean_val = float(current_X[feature].mean())
                        
                        value = st.slider(
                            f"{feature[:20]}...", 
                            min_val, max_val, mean_val,
                            key=f"slider_{i}_{st.session_state.dataset_name}"
                        )
                        input_values.append(value)
                
                # Hacer predicción
                if st.button("🔮 Predecir"):
                    prediction = st.session_state.supervised_model.predict([input_values])
                    prediction_proba = st.session_state.supervised_model.predict_proba([input_values])
                    
                    predicted_class = prediction[0]
                    predicted_label = current_targets[predicted_class]
                    confidence = np.max(prediction_proba) * 100
                    
                    st.success(f"**Predicción:** {predicted_label}")
                    st.info(f"**Confianza:** {confidence:.1f}%")
                    
                    # Guardar última predicción
                    st.session_state.last_prediction = {
                        'input': input_values,
                        'output_class': int(predicted_class),
                        'output_label': predicted_label,
                        'confidence': float(confidence)
                    }
            else:
                st.warning("⚠️ Por favor, entrena primero el modelo para habilitar las predicciones.")

# MODO NO SUPERVISADO
elif mode == "🔍 Modo No Supervisado":
    st.markdown("## 🔍 Modelo No Supervisado - Isolation Forest")
    
    # Descripción del modelo
    st.markdown("### 🕵️ ¿Qué es Isolation Forest?")
    
    with st.expander("📚 Descripción del Algoritmo", expanded=False):
        st.markdown("""
        **Isolation Forest** es un algoritmo de detección de anomalías que:
        
        **Funcionamiento:**
        - Construye árboles de aislamiento aleatorios
        - Separa puntos mediante divisiones aleatorias
        - Las anomalías se aíslan más rápidamente
        - No requiere etiquetas (aprendizaje no supervisado)
        
        **Principio:**
        - Puntos normales necesitan más divisiones para aislarse
        - Anomalías se separan con pocas divisiones
        - Calcula un "score de anomalía" para cada punto
        
        **Aplicaciones:**
        - Detección de fraudes financieros
        - Monitorización de sistemas
        - Control de calidad industrial
        - Seguridad en redes
        """)
    
    st.markdown("**Nota:** Selecciona un dataset para analizar anomalías.")
    
    # Selección de dataset independiente para modo no supervisado
    st.sidebar.markdown("### 📊 Selección de Dataset")
    dataset_name_unsupervised = st.sidebar.radio(
        "Elige un dataset para análisis de anomalías:",
        ["Flores Iris", "Vinos", "Cáncer de Mama"],
        key="unsupervised_dataset"
    )
    
    # Cargar dataset seleccionado
    df_unsupervised, feature_names_unsupervised, target_names_unsupervised = load_dataset(dataset_name_unsupervised)
    
    # Mostrar información del dataset seleccionado
    st.info(f"📊 Dataset seleccionado: **{dataset_name_unsupervised}**")
    
    # Mostrar información del dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📏 Filas", df_unsupervised.shape[0])
    with col2:
        st.metric("📊 Columnas", df_unsupervised.shape[1] - 2)  # -2 por target y target_names
    with col3:
        st.metric("🎯 Clases", len(target_names_unsupervised))
    
    # Preparar datos (solo características, sin etiquetas)
    X_unsupervised = df_unsupervised[feature_names_unsupervised]
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled_unsupervised = scaler.fit_transform(X_unsupervised)
    
    # Configuración del modelo
    st.sidebar.markdown("### ⚙️ Configuración Isolation Forest")
    
    contamination = st.sidebar.slider(
        "Contaminación (% de anomalías)", 
        0.01, 0.5, 0.1, 0.01
    )
    n_estimators = st.sidebar.slider("Número de árboles", 50, 200, 100)
    
    with st.sidebar.expander("🔍 Parámetros Explicados"):
        st.markdown("""
        **Contaminación:**
        - Porcentaje esperado de anomalías
        - 0.1 = 10% de datos son atípicos
        - Ajustar según conocimiento del dominio
        
        **Número de Árboles:**
        - Cantidad de árboles de aislamiento
        - Más árboles = estimación más estable
        - Valor típico: 100-200
        
        **Cómo Interpretar:**
        - Score negativo = más probable anomalía
        - Score positivo = comportamiento normal
        - Umbral automático basado en contaminación
        """)
    
    # Entrenar modelo
    if st.sidebar.button("🚀 Entrenar Isolation Forest"):
        with st.spinner("Entrenando modelo de detección de anomalías..."):
            # Crear y entrenar modelo
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42
            )
            
            # Entrenar y predecir
            anomaly_labels = iso_forest.fit_predict(X_scaled_unsupervised)
            anomaly_scores = iso_forest.decision_function(X_scaled_unsupervised)
            
            # Convertir etiquetas (-1 = anomalía, 1 = normal)
            cluster_labels = np.where(anomaly_labels == -1, 1, 0)  # 1 = anomalía, 0 = normal
            
            # Calcular métricas
            try:
                silhouette = silhouette_score(X_scaled_unsupervised, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_scaled_unsupervised, cluster_labels)
            except:
                silhouette = np.nan
                davies_bouldin = np.nan
            
            metricas = {
                'puntuacion_silueta': silhouette,
                'davies_bouldin': davies_bouldin,
                'anomalias_detectadas': np.sum(anomaly_labels == -1),
                'puntos_normales': np.sum(anomaly_labels == 1)
            }
            
            # Guardar en sesión
            st.session_state.unsupervised_model = iso_forest
            st.session_state.unsupervised_metrics = metricas
            st.session_state.anomaly_labels = anomaly_labels
            st.session_state.cluster_labels = cluster_labels
            st.session_state.anomaly_scores = anomaly_scores
            st.session_state.X_scaled_unsupervised = X_scaled_unsupervised
            st.session_state.scaler_unsupervised = scaler
            st.session_state.unsupervised_dataset_name = dataset_name_unsupervised
            st.session_state.unsupervised_df = df_unsupervised
            st.session_state.unsupervised_feature_names = feature_names_unsupervised
            
            st.success("¡Modelo de detección de anomalías entrenado!")
    
    # Mostrar resultados si el modelo está entrenado (FUERA del botón)
    if st.session_state.unsupervised_model is not None:
        # Verificar que el dataset actual coincide con el modelo entrenado
        if (hasattr(st.session_state, 'unsupervised_dataset_name') and 
            st.session_state.unsupervised_dataset_name == dataset_name_unsupervised):
                
            
            st.markdown("### 📊 Resultados de Detección de Anomalías")
            
            with st.expander("🕵️ ¿Cómo interpretar los resultados?", expanded=False):
                st.markdown("""
                **Puntuación Silueta:**
                - Mide qué tan bien separados están los clusters
                - Rango: -1 a 1
                - Valores altos (>0.5) = buena separación
                - Cerca de 0 = clusters superpuestos
                
                **Índice Davies-Bouldin:**
                - Mide la compacidad dentro de clusters
                - Valores más bajos = mejor clustering
                - Ideal: cercano a 0
                - Compara distancias intra vs inter-cluster
                
                **Anomalías vs Normales:**
                - Número de puntos clasificados como atípicos
                - Basado en el parámetro de contaminación
                - Revisar si el ratio es razonable para tu dominio
                """)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if not np.isnan(st.session_state.unsupervised_metrics['puntuacion_silueta']):
                    st.metric("📏 Puntuación Silueta", 
                            f"{st.session_state.unsupervised_metrics['puntuacion_silueta']:.3f}")
                else:
                    st.metric("📏 Puntuación Silueta", "N/A")
            
            with col2:
                if not np.isnan(st.session_state.unsupervised_metrics['davies_bouldin']):
                    st.metric("📊 Índice Davies-Bouldin", 
                            f"{st.session_state.unsupervised_metrics['davies_bouldin']:.3f}")
                else:
                    st.metric("📊 Índice Davies-Bouldin", "N/A")
            
            with col3:
                st.metric("⚠️ Anomalías", 
                        st.session_state.unsupervised_metrics['anomalias_detectadas'])
            
            with col4:
                st.metric("✅ Normales", 
                        st.session_state.unsupervised_metrics['puntos_normales'])
            
            # Visualización
            st.markdown("### 📈 Visualización de Anomalías")
            
            # Usar datos del modelo no supervisado entrenado
            if (hasattr(st.session_state, 'unsupervised_df') and 
                hasattr(st.session_state, 'unsupervised_feature_names') and
                len(st.session_state.unsupervised_feature_names) >= 2):
                
                unsupervised_df = st.session_state.unsupervised_df
                unsupervised_features = st.session_state.unsupervised_feature_names
                
                # Usar las dos primeras características para visualización
                fig = px.scatter(
                    x=unsupervised_df[unsupervised_features[0]], 
                    y=unsupervised_df[unsupervised_features[1]],
                    color=st.session_state.anomaly_labels,
                    color_discrete_map={1: 'blue', -1: 'red'},
                    labels={
                        'x': unsupervised_features[0],
                        'y': unsupervised_features[1],
                        'color': 'Tipo'
                    },
                    title="Detección de Anomalías (Rojo = Anomalía, Azul = Normal)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de scores de anomalía
                fig2 = px.histogram(
                    x=st.session_state.anomaly_scores,
                    nbins=30,
                    title="Distribución de Scores de Anomalía",
                    labels={'x': 'Score de Anomalía', 'y': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("⚠️ Has cambiado de dataset. Por favor, entrena nuevamente el modelo no supervisado para ver los resultados.")# ZONA DE EXPORTACIÓN
elif mode == "📁 Zona de Exportación":
    st.markdown("## 📁 Zona de Exportación (Dev Tools)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Exportación Modelo Supervisado")
        
        if st.session_state.supervised_model is not None:
            # JSON para modelo supervisado
            supervised_json = {
                "tipo_modelo": "Supervisado",
                "nombre_modelo": "Clasificador Gradient Boosting",
                "dataset_utilizado": st.session_state.dataset_name if st.session_state.dataset_name else "Desconocido",
                "fecha_hora": datetime.now().isoformat(),
                "metricas": st.session_state.supervised_metrics,
                "parametros": {
                    "num_estimadores": st.session_state.supervised_model.n_estimators,
                    "profundidad_maxima": st.session_state.supervised_model.max_depth,
                    "tasa_aprendizaje": st.session_state.supervised_model.learning_rate
                }
            }
            
            # Agregar última predicción si existe
            if hasattr(st.session_state, 'last_prediction'):
                supervised_json["prediccion_actual"] = st.session_state.last_prediction
            
            # Convertir tipos numpy
            supervised_json = convert_numpy_types(supervised_json)
            
            # Botón de descarga JSON
            json_str = json.dumps(supervised_json, indent=2)
            st.download_button(
                label="📥 Descargar JSON Supervisado",
                data=json_str,
                file_name="resultados_modelo_supervisado.json",
                mime="application/json"
            )
            
            # Botón de descarga PKL
            model_pkl = pickle.dumps(st.session_state.supervised_model)
            st.download_button(
                label="📥 Descargar Modelo PKL",
                data=model_pkl,
                file_name="modelo_gradient_boosting.pkl",
                mime="application/octet-stream"
            )
            
            # Mostrar preview del JSON
            st.markdown("#### 👀 Preview JSON:")
            st.json(supervised_json)
            
        else:
            st.warning("⚠️ No hay modelo supervisado entrenado.")
    
    with col2:
        st.markdown("### 🔍 Exportación Modelo No Supervisado")
        
        if st.session_state.unsupervised_model is not None:
            # JSON para modelo no supervisado
            unsupervised_json = {
                "tipo_modelo": "No Supervisado",
                "algoritmo": "Bosque de Aislamiento",
                "dataset_utilizado": st.session_state.dataset_name if st.session_state.dataset_name else "Desconocido",
                "fecha_hora": datetime.now().isoformat(),
                "parametros": {
                    "contaminacion": st.session_state.unsupervised_model.contamination,
                    "num_estimadores": st.session_state.unsupervised_model.n_estimators,
                    "max_muestras": st.session_state.unsupervised_model.max_samples
                },
                "metricas": st.session_state.unsupervised_metrics,
                "etiquetas_cluster": st.session_state.cluster_labels.tolist() if hasattr(st.session_state, 'cluster_labels') else []
            }
            
            # Convertir tipos numpy
            unsupervised_json = convert_numpy_types(unsupervised_json)
            
            # Botón de descarga JSON
            json_str = json.dumps(unsupervised_json, indent=2)
            st.download_button(
                label="📥 Descargar JSON No Supervisado",
                data=json_str,
                file_name="resultados_modelo_no_supervisado.json",
                mime="application/json"
            )
            
            # Botón de descarga PKL
            model_pkl = pickle.dumps(st.session_state.unsupervised_model)
            st.download_button(
                label="📥 Descargar Isolation Forest PKL",
                data=model_pkl,
                file_name="modelo_bosque_aislamiento.pkl",
                mime="application/octet-stream"
            )
            
            # Mostrar preview del JSON
            st.markdown("#### 👀 Preview JSON:")
            st.json(unsupervised_json)
            
        else:
            st.warning("⚠️ No hay modelo no supervisado entrenado.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Aplicación de Machine Learning | Desarrollado con Streamlit | 2025
</div>
""", unsafe_allow_html=True)