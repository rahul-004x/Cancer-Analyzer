import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def plot_correlation_heatmap(corr_matrix):
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(
            title='Feature Correlation Heatmap',
            width=800,
            height=800
        )
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_dict):
        fig = px.bar(
            x=list(importance_dict.values()),
            y=list(importance_dict.keys()),
            orientation='h',
            title='Feature Importance'
        )
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Features',
            width=800,
            height=600
        )
        return fig
    
    @staticmethod
    def plot_confusion_matrix(conf_matrix):
        fig = px.imshow(
            conf_matrix,
            labels=dict(x='Predicted', y='Actual'),
            x=['Benign', 'Malignant'],
            y=['Benign', 'Malignant'],
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            title='Confusion Matrix',
            width=600,
            height=600
        )
        return fig
