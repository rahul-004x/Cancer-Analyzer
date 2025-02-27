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

    data_loaded = False

    if uploaded_file is not None:
        data = process_uploaded_file(uploaded_file)
        if data is not None:
            dp.load_data_from_df(data)
            st.success("File successfully uploaded and processed!")
            data_loaded = True
    elif use_example_file:
        data = dp.load_data("attached_assets/Cancer_Data.csv")
        data_loaded = True

    if data_loaded and dp.data is not None:
        # Dataset Info
        st.header("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Shape:", dp.data.shape)
        with col2:
            st.write("Features:", len(dp.data.columns) - 2)  # Excluding 'id' and 'diagnosis'

        # Basic statistics
        st.header("Dataset Statistics")
        stats = dp.get_feature_stats()
        if stats is not None:
            st.write(stats)

        # Process data for visualizations
        if st.button("Process Data for Visualization"):
            with st.spinner("Processing data..."):
                try:
                    X_train, X_test, y_train, y_test = dp.preprocess_data()
                    st.success("Data processed successfully!")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.stop()

        if dp.X is not None:
            # Correlation matrix
            st.header("Feature Correlations")
            corr_matrix = dp.get_correlation_matrix()
            if corr_matrix is not None:
                try:
                    st.plotly_chart(viz.plot_correlation_heatmap(corr_matrix))
                except Exception as e:
                    st.error(f"Error plotting correlation matrix: {str(e)}")

            # Data distribution
            st.header("Feature Distributions")
            feature_names = dp.get_feature_names()
            if feature_names:
                col1, col2 = st.columns(2)
                with col1:
                    selected_feature = st.selectbox("Select Feature", feature_names)
                with col2:
                    chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot", "Violin Plot"])

                try:
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
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")

            # Train models if data is processed
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    try:
                        X_train, X_test, y_train, y_test = dp.preprocess_data()
                        mt.train_models(X_train, X_test, y_train, y_test)
                        st.success("Models trained successfully!")
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")

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
        
        # Choose input method
        input_method = st.radio("Input Method", ["Manual Entry", "CSV Upload"])
        
        if input_method == "CSV Upload":
            patient_file = st.file_uploader("Upload Patient CSV", type=["csv"])
            if patient_file is not None:
                try:
                    patient_df = pd.read_csv(patient_file)
                    expected_features = dp.get_feature_names()
                    dp.validate_input_data(patient_df[expected_features])
                    input_df = patient_df[expected_features].iloc[[0]]
                    st.success("Patient measurements loaded from CSV.")
                except Exception as e:
                    st.error(f"Error processing patient CSV: {str(e)}")
                    input_df = None
            else:
                st.info("Please upload a CSV file with patient measurements.")
                input_df = None
        else:
            # Manual Entry without auto-filled values
            input_features = {}
            if dp.X is not None:
                cols = st.columns(3)
                for idx, feature in enumerate(dp.X.columns):
                    with cols[idx % 3]:
                        val_str = st.text_input(feature, placeholder=f"Enter {feature}", key=feature)
                        input_features[feature] = float(val_str) if val_str.strip() != "" else None
            # Validate that all fields have been filled:
            if any(value is None for value in input_features.values()):
                st.error("Please fill all the fields with valid numeric values.")
                st.stop()
            input_df = pd.DataFrame([input_features])
        
        if input_df is not None and st.button("Predict"):
            predictions = mt.predict(input_df)
            for model_name, pred_dict in predictions.items():
                st.write(f"### {model_name} Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Diagnosis", pred_dict['prediction'])
                with col2:
                    st.metric("Confidence", f"{pred_dict['probability']:.2%}")
            
            # --- Display Spider Chart if input data includes all required keys ---
            input_data = input_df.to_dict(orient='records')[0]
            required_keys = [
                'area_mean', 'perimeter_mean', 'radius_mean', 'texture_mean', 'smoothness_mean',
                'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'area_se', 'perimeter_se', 'radius_se', 'texture_se', 'smoothness_se',
                'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                'area_worst', 'perimeter_worst', 'radius_worst', 'texture_worst', 'smoothness_worst',
                'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
            ]
            if all(key in input_data for key in required_keys):
                spider_chart = viz.create_spider_chart(input_data)
                st.plotly_chart(spider_chart)
            else:
                st.info("Spider chart not available: input data does not include all required metrics for spider chart.")