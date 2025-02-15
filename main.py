import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.visualizer import Visualizer
import io

# Page config
st.set_page_config(
    page_title="Cancer Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    dp = DataProcessor()
    mt = ModelTrainer()
    viz = Visualizer()
    return dp, mt, viz

dp, mt, viz = init_components()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Analysis", "Model Performance", "Prediction"])

# Function to load and process data
@st.cache_data
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    return None

# Data Analysis Page
if page == "Data Analysis":
    st.title("Cancer Data Analysis")

    # File upload section
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    use_example_file = st.checkbox('Use example file', value=True)

    if uploaded_file is not None:
        data = process_uploaded_file(uploaded_file)
        if data is not None:
            dp.load_data_from_df(data)
            st.success("File successfully uploaded and processed!")
    elif use_example_file:
        data = dp.load_data("attached_assets/Cancer_Data.csv")

    if dp.data is not None:
        # Dataset Info
        st.header("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Shape:", dp.data.shape)
        with col2:
            st.write("Features:", len(dp.data.columns) - 2)  # Excluding 'id' and 'diagnosis'

        # Basic statistics
        st.header("Dataset Statistics")
        st.write(dp.get_feature_stats())

        # Correlation matrix
        st.header("Feature Correlations")
        corr_matrix = dp.get_correlation_matrix()
        if corr_matrix is not None:
            st.plotly_chart(viz.plot_correlation_heatmap(corr_matrix))

        # Data distribution
        st.header("Feature Distributions")
        if dp.X is not None and len(dp.X.columns) > 0:
            col1, col2 = st.columns(2)
            with col1:
                selected_feature = st.selectbox("Select Feature", dp.X.columns)
            with col2:
                chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot", "Violin Plot"])

            if chart_type == "Histogram":
                fig = px.histogram(
                    dp.X,
                    x=selected_feature,
                    title=f'Distribution of {selected_feature}',
                    template='plotly_white'
                )
            elif chart_type == "Box Plot":
                fig = px.box(
                    dp.X,
                    y=selected_feature,
                    title=f'Box Plot of {selected_feature}',
                    template='plotly_white'
                )
            else:  # Violin Plot
                fig = px.violin(
                    dp.X,
                    y=selected_feature,
                    title=f'Violin Plot of {selected_feature}',
                    template='plotly_white'
                )
            st.plotly_chart(fig)

            # Train models if data is processed
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    X_train, X_test, y_train, y_test = dp.preprocess_data()
                    mt.train_models(X_train, X_test, y_train, y_test)
                st.success("Models trained successfully!")

# Model Performance Page
elif page == "Model Performance":
    st.title("Model Performance Comparison")

    if not mt.results:
        st.warning("Please train the models first in the Data Analysis page.")
    else:
        # Model metrics
        st.header("Model Metrics")
        for model_name, results in mt.results.items():
            st.subheader(model_name)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.2%}")
            with col2:
                st.text("Classification Report")
                st.text(results['classification_report'])

            # Confusion Matrix
            st.plotly_chart(viz.plot_confusion_matrix(results['confusion_matrix']))

        # Feature importance
        st.header("Feature Importance")
        if dp.X is not None:
            importance_dict = mt.get_feature_importance(dp.X.columns)
            if importance_dict:
                st.plotly_chart(viz.plot_feature_importance(importance_dict))

# Prediction Page
else:
    st.title("Cancer Prediction")

    if not mt.trained_models:
        st.warning("Please train the models first in the Data Analysis page.")
    else:
        st.write("Enter patient measurements to get prediction")

        # Create input fields for all features
        input_features = {}
        if dp.X is not None:
            cols = st.columns(3)
            for idx, feature in enumerate(dp.X.columns):
                with cols[idx % 3]:
                    input_features[feature] = st.number_input(
                        feature,
                        value=float(dp.X[feature].mean()),
                        format="%.6f"
                    )

            if st.button("Predict"):
                # Prepare input data
                input_df = pd.DataFrame([input_features])

                # Get predictions
                predictions = mt.predict(input_df)

                # Display results
                for model_name, pred_dict in predictions.items():
                    st.write(f"### {model_name} Prediction")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Diagnosis", pred_dict['prediction'])
                    with col2:
                        st.metric("Confidence", f"{pred_dict['probability']:.2%}")