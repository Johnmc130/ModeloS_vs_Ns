# 🤖 Aplicación de Machine Learning

Aplicación web interactiva desarrollada con Streamlit que implementa modelos de aprendizaje automático tanto supervisados como no supervisados.

## 🎯 Características Principales

### 📊 Modelos Implementados

**Modelo Supervisado: Gradient Boosting**
- Clasificación usando ensemble de árboles de decisión
- División automática entrenamiento/prueba (80/20)
- Métricas de evaluación: Exactitud, Precisión, Sensibilidad, Puntuación F1
- Predicción interactiva en tiempo real

**Modelo No Supervisado: Isolation Forest**
- Detección automática de anomalías
- Análisis de outliers en los datos
- Métricas de clustering: Puntuación Silueta, Índice Davies-Bouldin
- Visualizaciones interactivas

### 📈 Datasets Disponibles

| Dataset | Descripción | Características | Clases |
|---------|-------------|-----------------|---------|
| **Flores Iris** | Clasificación de especies de flores | 4 | 3 |
| **Vinos** | Clasificación de tipos de vino | 13 | 3 |
| **Cáncer de Mama** | Diagnóstico médico | 30 | 2 |

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
pip install -r requirements.txt
```

### Ejecutar la Aplicación
```bash
streamlit run app.py
```

La aplicación estará disponible en: `http://localhost:8501`

## 🔧 Funcionalidades

### 📊 Modo Supervisado
1. Seleccionar dataset
2. Cargar y visualizar datos
3. Configurar parámetros del modelo
4. Entrenar Gradient Boosting
5. Evaluar rendimiento con métricas
6. Realizar predicciones interactivas

### 🔍 Modo No Supervisado
1. Utilizar dataset previamente cargado
2. Configurar parámetros de Isolation Forest
3. Entrenar modelo de detección de anomalías
4. Visualizar resultados y métricas
5. Analizar patrones de anomalías

### 📁 Exportación
- **JSON**: Resultados y configuraciones del modelo
- **PKL**: Modelos entrenados para reutilización
- Archivos listos para integración con otras aplicaciones

## 🎮 Interfaz de Usuario

### Navegación Principal
- **🏠 Inicio**: Información general y descripción
- **📊 Modo Supervisado**: Entrenamiento y evaluación supervisada
- **🔍 Modo No Supervisado**: Detección de anomalías
- **📁 Zona de Exportación**: Descarga de resultados

### Características Interactivas
- Sliders dinámicos para predicción en tiempo real
- Gráficos interactivos con Plotly
- Matrices de confusión visuales
- Métricas en tiempo real

## 📦 Estructura del Proyecto

```
├── app.py              # Aplicación principal de Streamlit
├── requirements.txt    # Dependencias del proyecto
├── run_app.bat        # Script de ejecución para Windows
├── test_models.py     # Scripts de prueba
└── README.md          # Documentación del proyecto
```

## 🛠️ Tecnologías Utilizadas

- **Framework**: Streamlit 1.28.0
- **ML**: scikit-learn 1.3.0
- **Visualización**: matplotlib 3.7.2, seaborn 0.12.2, plotly 5.15.0
- **Datos**: pandas 2.0.3, numpy 1.24.3
- **Exportación**: JSON, pickle

## 🎨 Ejemplos de Uso

### Predicción con Flores Iris
```
Longitud Sépalo: 5.1 cm
Anchura Sépalo: 3.5 cm
Longitud Pétalo: 1.4 cm
Anchura Pétalo: 0.2 cm
→ Resultado: Setosa (98% confianza)
```

### Detección de Anomalías
```
Dataset: Vinos (13 características)
Contaminación: 10%
→ Resultado: 15 anomalías detectadas de 178 muestras
```

## 📊 Métricas de Rendimiento

La aplicación proporciona métricas completas para evaluar el rendimiento:

**Supervisado**: Exactitud, Precisión, Sensibilidad, Puntuación F1
**No Supervisado**: Puntuación Silueta, Índice Davies-Bouldin

## 🤝 Contribuciones

Este proyecto está diseñado para fines educativos y de investigación en Machine Learning.

## 📄 Licencia

Proyecto académico - Universidad, 6to Ciclo, Inteligencia Artificial

---

**¡Explora el poder del Machine Learning de forma interactiva!** 🚀

- **Iris**: Clasificación de especies de flores (150 muestras, 4 características, 3 clases)
- **Wine**: Clasificación de tipos de vino (178 muestras, 13 características, 3 clases)
- **Breast Cancer**: Diagnóstico de cáncer de mama (569 muestras, 30 características, 2 clases)

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
python 3.8+
pip
```

### Instalación
```bash
# Navegar a la carpeta del proyecto
cd "c:\Users\Usuario\OneDrive\Desktop\Universidad\6to Ciclo\Inteligenica Artificial\Modelos"

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar la aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🎮 Cómo usar la aplicación

### 1. 🏠 Página de Inicio
- Descripción general de la aplicación
- Información sobre los modelos y datasets

### 2. 📊 Modo Supervisado
1. Selecciona un dataset desde el panel lateral
2. Carga el dataset haciendo click en "🔄 Cargar Dataset"
3. Ajusta los parámetros del modelo (estimadores, profundidad, tasa de aprendizaje)
4. Entrena el modelo con "🚀 Entrenar Modelo"
5. Visualiza las métricas y matriz de confusión
6. Usa la interfaz interactiva para hacer predicciones

**⚠️ Importante**: Si cambias de dataset, debes entrenar nuevamente el modelo. Cada dataset tiene diferente número de características:
- **Iris**: 4 características
- **Wine**: 13 características  
- **Breast Cancer**: 30 características

### 3. 🔍 Modo No Supervisado
1. Asegúrate de haber cargado un dataset en el modo supervisado
2. Ajusta los parámetros del Isolation Forest
3. Entrena el modelo de detección de anomalías
4. Visualiza los resultados y gráficos de anomalías

### 4. 📁 Zona de Exportación
- Descarga resultados en formato JSON para integración con React
- Descarga modelos entrenados en formato .pkl
- Visualiza previews de los archivos JSON

## 📁 Archivos Exportados

### JSON para React
Los archivos JSON contienen:
- Tipo de modelo y algoritmo usado
- Métricas de evaluación
- Parámetros del modelo
- Timestamp de entrenamiento
- Resultados de predicciones (supervisado)
- Etiquetas de clusters (no supervisado)

### Modelos PKL
Los archivos .pkl contienen los modelos entrenados que pueden ser:
- Cargados posteriormente para hacer predicciones
- Integrados en otros sistemas
- Utilizados en aplicaciones de producción

## 📈 Métricas Implementadas

### Modelo Supervisado
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Precisión promedio ponderada por clase
- **Recall**: Sensibilidad promedio ponderada por clase
- **F1-Score**: Media armónica entre precisión y recall

### Modelo No Supervisado
- **Silhouette Score**: Calidad de la separación de clusters (-1 a 1)
- **Davies-Bouldin Score**: Ratio de dispersión intra-cluster vs inter-cluster (menor es mejor)
- **Conteo de anomalías**: Número de puntos clasificados como anómalos

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework web para aplicaciones de ML
- **Scikit-learn**: Biblioteca de machine learning
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Matplotlib/Seaborn**: Visualización estática
- **Plotly**: Visualización interactiva
- **Pickle**: Serialización de modelos

## 📝 Estructura del Proyecto

```
Modelos/
├── app.py              # Aplicación principal de Streamlit
├── requirements.txt    # Dependencias del proyecto
└── README.md          # Esta documentación
```

## 🔧 Personalización

Para usar tus propios datasets:
1. Modifica la función `load_dataset()` en `app.py`
2. Agrega tu dataset siguiendo el formato pandas DataFrame
3. Asegúrate de que tenga columnas de características y target
4. Actualiza la lista de datasets en el selectbox

## 📞 Soporte

Si encuentras algún problema:
1. Verifica que todas las dependencias estén instaladas
2. Asegúrate de usar Python 3.8 o superior
3. Revisa que no haya conflictos con otras versiones de bibliotecas

## 📄 Licencia

Este proyecto es para uso educativo como parte de la actividad de Inteligencia Artificial.