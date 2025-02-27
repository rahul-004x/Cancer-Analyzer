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

    @staticmethod
    def create_spider_chart(input_data):
        """
        Create a radar (spider) chart to visualize patient characteristics.
        Expects input_data to be a dict with normalized (0-1) values for:
          - Mean: area_mean, perimeter_mean, radius_mean, texture_mean, smoothness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
          - Standard Error: area_se, perimeter_se, radius_se, texture_se, smoothness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se
          - Worst: area_worst, perimeter_worst, radius_worst, texture_worst, smoothness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
        """
        import plotly.graph_objects as go

        categories = ['Area', 'Perimeter', 'Radius', 'Texture', 'Smoothness', 
                      'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']
                      
        # Retrieve three layers of values in the same order
        mean_values = [
            input_data['area_mean'], input_data['perimeter_mean'], input_data['radius_mean'],
            input_data['texture_mean'], input_data['smoothness_mean'], input_data['concavity_mean'],
            input_data['concave_points_mean'], input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ]
        se_values = [
            input_data['area_se'], input_data['perimeter_se'], input_data['radius_se'],
            input_data['texture_se'], input_data['smoothness_se'], input_data['concavity_se'],
            input_data['concave_points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ]
        worst_values = [
            input_data['area_worst'], input_data['perimeter_worst'], input_data['radius_worst'],
            input_data['texture_worst'], input_data['smoothness_worst'], input_data['concavity_worst'],
            input_data['concave_points_worst'], input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ]
        
        # Close the loop for radar chart
        categories += [categories[0]]
        mean_values += [mean_values[0]]
        se_values += [se_values[0]]
        worst_values += [worst_values[0]]
        
        fig = go.Figure()
        
        # Mean values in blue
        fig.add_trace(go.Scatterpolar(
            r=mean_values,
            theta=categories,
            fill='toself',
            name='Mean Values',
            line=dict(color='blue'),
            marker=dict(size=4)
        ))
        
        # Standard Error values in light blue
        fig.add_trace(go.Scatterpolar(
            r=se_values,
            theta=categories,
            fill='toself',
            name='Standard Error',
            line=dict(color='lightblue'),
            marker=dict(size=4)
        ))
        
        # Worst values in red
        fig.add_trace(go.Scatterpolar(
            r=worst_values,
            theta=categories,
            fill='toself',
            name='Worst Values',
            line=dict(color='red'),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["0.2", "0.4", "0.6", "0.8", "1.0"]
                )
            ),
            showlegend=True,
            title='Patient Characteristics Radar Chart',
            template='plotly_white'
        )
        return fig
