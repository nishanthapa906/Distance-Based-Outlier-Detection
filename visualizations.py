import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_outlier_plot(data, outlier_indices, outlier_scores, x_column, y_column, algorithm_name):
    """
    Create an interactive scatter plot showing outliers
    
    Args:
        data: DataFrame with the data
        outlier_indices: Indices of outliers
        outlier_scores: Outlier scores
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        algorithm_name: Name of the algorithm
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    # Create masks for normal and outlier points
    normal_mask = np.ones(len(data), dtype=bool)
    normal_mask[outlier_indices] = False
    
    # Create the figure
    fig = go.Figure()
    
    # Add normal points
    fig.add_trace(go.Scatter(
        x=data[x_column][normal_mask],
        y=data[y_column][normal_mask],
        mode='markers',
        name='Normal Points',
        marker=dict(
            color='lightblue',
            size=6,
            opacity=0.7
        ),
        hovertemplate='<b>Normal Point</b><br>' +
                      f'{x_column}: %{{x:.3f}}<br>' +
                      f'{y_column}: %{{y:.3f}}<br>' +
                      '<extra></extra>'
    ))
    
    # Add outlier points
    if len(outlier_indices) > 0:
        fig.add_trace(go.Scatter(
            x=data[x_column].iloc[outlier_indices],
            y=data[y_column].iloc[outlier_indices],
            mode='markers',
            name='Outliers',
            marker=dict(
                color='red',
                size=10,
                symbol='x',
                opacity=0.8
            ),
            hovertemplate='<b>Outlier</b><br>' +
                          f'{x_column}: %{{x:.3f}}<br>' +
                          f'{y_column}: %{{y:.3f}}<br>' +
                          'Score: %{customdata:.3f}<br>' +
                          '<extra></extra>',
            customdata=outlier_scores[outlier_indices]
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{algorithm_name} - Outlier Detection Results',
        xaxis_title=x_column,
        yaxis_title=y_column,
        hovermode='closest',
        showlegend=True,
        width=800,
        height=600
    )
    
    return fig

def create_comparison_plot(data, results, x_column, y_column):
    """
    Create a comparison plot showing results from multiple algorithms
    
    Args:
        data: DataFrame with the data
        results: Dictionary of results from different algorithms
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        
    Returns:
        plotly.graph_objects.Figure: Comparison plot
    """
    # Create subplots
    n_algorithms = len(results)
    cols = min(2, n_algorithms)
    rows = (n_algorithms + 1) // 2
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(results.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = ['red', 'orange', 'purple', 'green', 'blue']
    
    for i, (algorithm, result) in enumerate(results.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        # Normal points
        normal_mask = np.ones(len(data), dtype=bool)
        normal_mask[result['outliers']] = False
        
        fig.add_trace(go.Scatter(
            x=data[x_column][normal_mask],
            y=data[y_column][normal_mask],
            mode='markers',
            name=f'{algorithm} - Normal',
            marker=dict(color='lightblue', size=4, opacity=0.6),
            showlegend=False
        ), row=row, col=col)
        
        # Outliers
        if len(result['outliers']) > 0:
            fig.add_trace(go.Scatter(
                x=data[x_column].iloc[result['outliers']],
                y=data[y_column].iloc[result['outliers']],
                mode='markers',
                name=f'{algorithm} - Outliers',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8,
                    symbol='x',
                    opacity=0.8
                ),
                showlegend=i == 0
            ), row=row, col=col)
    
    fig.update_layout(
        title='Algorithm Comparison',
        height=300 * rows,
        showlegend=True
    )
    
    return fig

def create_score_distribution_plot(results):
    """
    Create a plot showing the distribution of outlier scores
    
    Args:
        results: Dictionary of results from different algorithms
        
    Returns:
        plotly.graph_objects.Figure: Score distribution plot
    """
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (algorithm, result) in enumerate(results.items()):
        if 'scores' in result:
            fig.add_trace(go.Histogram(
                x=result['scores'],
                name=algorithm,
                opacity=0.7,
                nbinsx=50,
                marker_color=colors[i % len(colors)]
            ))
    
    fig.update_layout(
        title='Outlier Score Distributions',
        xaxis_title='Outlier Score',
        yaxis_title='Frequency',
        barmode='overlay',
        height=500
    )
    
    return fig

def create_3d_plot(data, outlier_indices, x_column, y_column, z_column, algorithm_name):
    """
    Create a 3D scatter plot for outlier visualization
    
    Args:
        data: DataFrame with the data
        outlier_indices: Indices of outliers
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        z_column: Column name for z-axis
        algorithm_name: Name of the algorithm
        
    Returns:
        plotly.graph_objects.Figure: 3D scatter plot
    """
    # Create masks for normal and outlier points
    normal_mask = np.ones(len(data), dtype=bool)
    normal_mask[outlier_indices] = False
    
    fig = go.Figure()
    
    # Add normal points
    fig.add_trace(go.Scatter3d(
        x=data[x_column][normal_mask],
        y=data[y_column][normal_mask],
        z=data[z_column][normal_mask],
        mode='markers',
        name='Normal Points',
        marker=dict(
            color='lightblue',
            size=4,
            opacity=0.6
        )
    ))
    
    # Add outlier points
    if len(outlier_indices) > 0:
        fig.add_trace(go.Scatter3d(
            x=data[x_column].iloc[outlier_indices],
            y=data[y_column].iloc[outlier_indices],
            z=data[z_column].iloc[outlier_indices],
            mode='markers',
            name='Outliers',
            marker=dict(
                color='red',
                size=8,
                symbol='x',
                opacity=0.8
            )
        ))
    
    fig.update_layout(
        title=f'{algorithm_name} - 3D Outlier Detection',
        scene=dict(
            xaxis_title=x_column,
            yaxis_title=y_column,
            zaxis_title=z_column
        ),
        height=600
    )
    
    return fig

def create_performance_comparison_chart(performance_metrics):
    """
    Create a bar chart comparing algorithm performance
    
    Args:
        performance_metrics: Dictionary of performance metrics
        
    Returns:
        plotly.graph_objects.Figure: Performance comparison chart
    """
    algorithms = list(performance_metrics.keys())
    
    # Extract metrics
    execution_times = [performance_metrics[alg].get('execution_time', 0) for alg in algorithms]
    n_outliers = [performance_metrics[alg].get('n_outliers', 0) for alg in algorithms]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Execution Time (seconds)', 'Number of Outliers Detected'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Execution time chart
    fig.add_trace(go.Bar(
        x=algorithms,
        y=execution_times,
        name='Execution Time',
        marker_color='lightblue'
    ), row=1, col=1)
    
    # Number of outliers chart
    fig.add_trace(go.Bar(
        x=algorithms,
        y=n_outliers,
        name='Outliers Detected',
        marker_color='lightcoral'
    ), row=1, col=2)
    
    fig.update_layout(
        title='Algorithm Performance Comparison',
        height=400,
        showlegend=False
    )
    
    return fig

def create_feature_importance_plot(feature_importance):
    """
    Create a plot showing feature importance
    
    Args:
        feature_importance: Dictionary of feature importance scores
        
    Returns:
        plotly.graph_objects.Figure: Feature importance plot
    """
    features = list(feature_importance.keys())
    importance_scores = list(feature_importance.values())
    
    fig = go.Figure(go.Bar(
        x=features,
        y=importance_scores,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Feature Importance (Based on Variance)',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        height=400
    )
    
    return fig

def create_confusion_matrix_heatmap(cm_results, algorithm_name):
    """
    Create a heatmap visualization of the confusion matrix
    
    Args:
        cm_results: Results from calculate_confusion_matrix
        algorithm_name: Name of the algorithm
        
    Returns:
        plotly.graph_objects.Figure: Confusion matrix heatmap
    """
    cm = cm_results['confusion_matrix']
    
    # Create annotations for the heatmap
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=str(cm[i][j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "black", size=16)
                )
            )
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Outlier'],
        y=['Actual Normal', 'Actual Outlier'],
        colorscale='Blues',
        showscale=True
    ))
    
    # Add annotations
    fig.update_layout(
        title=f'Confusion Matrix - {algorithm_name}',
        annotations=annotations,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
        width=500,
        height=400
    )
    
    return fig

def create_performance_metrics_radar_chart(performance_metrics):
    """
    Create a radar chart comparing algorithm performance metrics
    
    Args:
        performance_metrics: Dictionary of performance metrics for each algorithm
        
    Returns:
        plotly.graph_objects.Figure: Radar chart
    """
    fig = go.Figure()
    
    # Metrics to display
    metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'specificity']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (algorithm, cm_results) in enumerate(performance_metrics.items()):
        if 'precision' in cm_results:  # Check if confusion matrix results are available
            values = [cm_results.get(metric, 0) for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=algorithm,
                line_color=colors[i % len(colors)]
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Algorithm Performance Comparison (Radar Chart)",
        height=500
    )
    
    return fig

def create_roc_curve_placeholder(algorithm_results):
    """
    Create a placeholder ROC curve visualization
    Note: This is a simplified version - full ROC curve would require probability scores
    
    Args:
        algorithm_results: Dictionary of algorithm results
        
    Returns:
        plotly.graph_objects.Figure: ROC curve plot
    """
    fig = go.Figure()
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Classifier'
    ))
    
    # Add placeholder points for each algorithm
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (algorithm, metrics) in enumerate(algorithm_results.items()):
        if 'recall' in metrics and 'specificity' in metrics:
            tpr = metrics['recall']  # True Positive Rate
            fpr = 1 - metrics['specificity']  # False Positive Rate
            
            fig.add_trace(go.Scatter(
                x=[fpr],
                y=[tpr],
                mode='markers',
                marker=dict(size=10, color=colors[i % len(colors)]),
                name=f'{algorithm} (TPR={tpr:.3f}, FPR={fpr:.3f})'
            ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        height=500
    )
    
    return fig
