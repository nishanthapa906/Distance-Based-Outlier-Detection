import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, calinski_harabasz_score, confusion_matrix, classification_report
import time

def calculate_metrics(outlier_indices, outlier_scores):
    """
    Calculate various metrics for outlier detection results
    
    Args:
        outlier_indices: Indices of detected outliers
        outlier_scores: Outlier scores for all points
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['n_outliers'] = len(outlier_indices)
    metrics['outlier_percentage'] = len(outlier_indices) / len(outlier_scores) * 100
    
    # Score statistics
    metrics['mean_score'] = np.mean(outlier_scores)
    metrics['std_score'] = np.std(outlier_scores)
    metrics['min_score'] = np.min(outlier_scores)
    metrics['max_score'] = np.max(outlier_scores)
    
    # Outlier score statistics
    if len(outlier_indices) > 0:
        outlier_scores_subset = outlier_scores[outlier_indices]
        metrics['outlier_mean_score'] = np.mean(outlier_scores_subset)
        metrics['outlier_std_score'] = np.std(outlier_scores_subset)
        metrics['outlier_min_score'] = np.min(outlier_scores_subset)
        metrics['outlier_max_score'] = np.max(outlier_scores_subset)
    else:
        metrics['outlier_mean_score'] = 0
        metrics['outlier_std_score'] = 0
        metrics['outlier_min_score'] = 0
        metrics['outlier_max_score'] = 0
    
    # Score distribution metrics
    if len(outlier_scores) > 0:
        metrics['score_range'] = metrics['max_score'] - metrics['min_score']
        metrics['score_q25'] = np.percentile(outlier_scores, 25)
        metrics['score_q75'] = np.percentile(outlier_scores, 75)
        metrics['score_iqr'] = metrics['score_q75'] - metrics['score_q25']
    
    return metrics

def evaluate_clustering_quality(data, outlier_indices):
    """
    Evaluate clustering quality using silhouette score
    
    Args:
        data: Original dataset
        outlier_indices: Indices of detected outliers
        
    Returns:
        dict: Clustering quality metrics
    """
    if len(outlier_indices) == 0 or len(outlier_indices) == len(data):
        return {'silhouette_score': 0, 'calinski_harabasz_score': 0}
    
    # Create labels (0 for normal, 1 for outlier)
    labels = np.zeros(len(data))
    labels[outlier_indices] = 1
    
    try:
        # Calculate silhouette score
        silhouette = silhouette_score(data, labels)
        
        # Calculate Calinski-Harabasz score
        calinski_harabasz = calinski_harabasz_score(data, labels)
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz
        }
    except Exception as e:
        return {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'error': str(e)}

def calculate_algorithm_stability(data, algorithm_func, n_runs=5):
    """
    Calculate stability of an algorithm across multiple runs
    
    Args:
        data: Dataset to analyze
        algorithm_func: Function that returns outlier indices
        n_runs: Number of runs to perform
        
    Returns:
        dict: Stability metrics
    """
    results = []
    
    for _ in range(n_runs):
        outlier_indices, _ = algorithm_func(data)
        results.append(set(outlier_indices))
    
    # Calculate Jaccard similarity between runs
    similarities = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            intersection = len(results[i].intersection(results[j]))
            union = len(results[i].union(results[j]))
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
    
    return {
        'mean_jaccard_similarity': np.mean(similarities),
        'std_jaccard_similarity': np.std(similarities),
        'min_jaccard_similarity': np.min(similarities),
        'max_jaccard_similarity': np.max(similarities)
    }

def display_performance_metrics(performance_metrics):
    """
    Display performance metrics in Streamlit
    
    Args:
        performance_metrics: Dictionary of metrics for each algorithm
    """
    # Create a DataFrame for better display
    metrics_df = pd.DataFrame(performance_metrics).T
    
    # Display execution time comparison
    st.subheader("⏱️ Execution Time")
    col1, col2, col3 = st.columns(3)
    
    execution_times = {alg: metrics.get('execution_time', 0) 
                      for alg, metrics in performance_metrics.items()}
    
    fastest_alg = min(execution_times, key=execution_times.get)
    slowest_alg = max(execution_times, key=execution_times.get)
    
    with col1:
        st.metric("Fastest Algorithm", fastest_alg, f"{execution_times[fastest_alg]:.2f}s")
    with col2:
        st.metric("Slowest Algorithm", slowest_alg, f"{execution_times[slowest_alg]:.2f}s")
    with col3:
        avg_time = np.mean(list(execution_times.values()))
        st.metric("Average Time", f"{avg_time:.2f}s")
    
    # Display outlier detection statistics
    st.subheader("📊 Detection Statistics")
    
    # Create comparison table
    comparison_data = []
    for algorithm, metrics in performance_metrics.items():
        comparison_data.append({
            'Algorithm': algorithm,
            'Outliers Found': metrics.get('n_outliers', 0),
            'Outlier %': f"{metrics.get('outlier_percentage', 0):.2f}%",
            'Execution Time (s)': f"{metrics.get('execution_time', 0):.2f}",
            'Mean Score': f"{metrics.get('outlier_mean_score', 0):.3f}",
            'Score Range': f"{metrics.get('score_range', 0):.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Display detailed metrics in expandable sections
    for algorithm, metrics in performance_metrics.items():
        with st.expander(f"📈 Detailed Metrics - {algorithm}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Statistics**")
                st.metric("Total Outliers", metrics.get('n_outliers', 0))
                st.metric("Outlier Percentage", f"{metrics.get('outlier_percentage', 0):.2f}%")
                st.metric("Execution Time", f"{metrics.get('execution_time', 0):.2f}s")
            
            with col2:
                st.markdown("**Score Statistics**")
                st.metric("Mean Score", f"{metrics.get('mean_score', 0):.3f}")
                st.metric("Score Std Dev", f"{metrics.get('std_score', 0):.3f}")
                st.metric("Score Range", f"{metrics.get('score_range', 0):.3f}")

def generate_performance_report(performance_metrics, data_info):
    """
    Generate a comprehensive performance report
    
    Args:
        performance_metrics: Dictionary of metrics for each algorithm
        data_info: Information about the dataset
        
    Returns:
        dict: Comprehensive performance report
    """
    report = {
        'dataset_info': data_info,
        'algorithm_performance': performance_metrics,
        'summary': {}
    }
    
    # Calculate summary statistics
    execution_times = [metrics.get('execution_time', 0) 
                      for metrics in performance_metrics.values()]
    outlier_counts = [metrics.get('n_outliers', 0) 
                     for metrics in performance_metrics.values()]
    
    report['summary'] = {
        'total_algorithms_tested': len(performance_metrics),
        'average_execution_time': np.mean(execution_times),
        'total_unique_outliers': len(set().union(*[
            set(metrics.get('outlier_indices', [])) 
            for metrics in performance_metrics.values()
        ])),
        'average_outliers_per_algorithm': np.mean(outlier_counts),
        'execution_time_variance': np.var(execution_times)
    }
    
    # Recommendations
    recommendations = []
    
    # Speed recommendation
    fastest_alg = min(performance_metrics.keys(), 
                     key=lambda x: performance_metrics[x].get('execution_time', float('inf')))
    recommendations.append(f"For fastest execution: {fastest_alg}")
    
    # Sensitivity recommendation
    most_sensitive = max(performance_metrics.keys(),
                        key=lambda x: performance_metrics[x].get('n_outliers', 0))
    recommendations.append(f"For highest sensitivity: {most_sensitive}")
    
    report['recommendations'] = recommendations
    
    return report

def calculate_consensus_outliers(results):
    """
    Calculate consensus outliers from multiple algorithms
    
    Args:
        results: Dictionary of results from different algorithms
        
    Returns:
        dict: Consensus analysis
    """
    all_outliers = {}
    
    # Collect all outliers from each algorithm
    for algorithm, result in results.items():
        if 'outliers' in result:
            all_outliers[algorithm] = set(result['outliers'])
    
    if not all_outliers:
        return {'consensus_outliers': set(), 'agreement_matrix': {}}
    
    # Find consensus outliers (detected by multiple algorithms)
    all_indices = set().union(*all_outliers.values())
    consensus_outliers = set()
    
    agreement_counts = {}
    for idx in all_indices:
        count = sum(1 for outlier_set in all_outliers.values() if idx in outlier_set)
        agreement_counts[idx] = count
        
        # Consider consensus if detected by at least half of the algorithms
        if count >= len(all_outliers) / 2:
            consensus_outliers.add(idx)
    
    # Calculate agreement matrix
    agreement_matrix = {}
    algorithms = list(all_outliers.keys())
    
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i <= j:
                intersection = len(all_outliers[alg1].intersection(all_outliers[alg2]))
                union = len(all_outliers[alg1].union(all_outliers[alg2]))
                jaccard = intersection / union if union > 0 else 0
                agreement_matrix[f"{alg1}_{alg2}"] = jaccard
    
    return {
        'consensus_outliers': consensus_outliers,
        'agreement_counts': agreement_counts,
        'agreement_matrix': agreement_matrix,
        'total_consensus': len(consensus_outliers)
    }

def calculate_confusion_matrix(y_true, y_pred, labels=None):
    """
    Calculate confusion matrix for outlier detection results
    
    Args:
        y_true: True binary labels (0: normal, 1: outlier)
        y_pred: Predicted binary labels (0: normal, 1: outlier)
        labels: Label names for display
        
    Returns:
        dict: Confusion matrix and related metrics
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate performance metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'confusion_matrix': cm,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'accuracy': accuracy
    }

def evaluate_outlier_detection_performance(data, true_outliers, detected_outliers):
    """
    Evaluate outlier detection performance using various metrics
    
    Args:
        data: Original dataset
        true_outliers: True outlier indices (if available)
        detected_outliers: Detected outlier indices
        
    Returns:
        dict: Performance evaluation metrics
    """
    if true_outliers is None or len(true_outliers) == 0:
        return {'error': 'No ground truth available for evaluation'}
    
    # Create binary labels
    y_true = np.zeros(len(data))
    y_true[true_outliers] = 1
    
    y_pred = np.zeros(len(data))
    y_pred[detected_outliers] = 1
    
    # Calculate confusion matrix and metrics
    cm_results = calculate_confusion_matrix(y_true, y_pred)
    
    # Calculate additional metrics
    jaccard_similarity = len(set(true_outliers).intersection(set(detected_outliers))) / len(set(true_outliers).union(set(detected_outliers)))
    
    return {
        **cm_results,
        'jaccard_similarity': jaccard_similarity,
        'true_outlier_count': len(true_outliers),
        'detected_outlier_count': len(detected_outliers)
    }

def display_confusion_matrix(cm_results, algorithm_name):
    """
    Display confusion matrix in Streamlit
    
    Args:
        cm_results: Results from calculate_confusion_matrix
        algorithm_name: Name of the algorithm
    """
    st.subheader(f"🔍 Confusion Matrix - {algorithm_name}")
    
    # Display confusion matrix
    cm = cm_results['confusion_matrix']
    
    # Create confusion matrix DataFrame for better display
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Normal', 'Actual Outlier'],
        columns=['Predicted Normal', 'Predicted Outlier']
    )
    
    st.dataframe(cm_df, use_container_width=True)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", f"{cm_results['precision']:.3f}")
    with col2:
        st.metric("Recall", f"{cm_results['recall']:.3f}")
    with col3:
        st.metric("F1-Score", f"{cm_results['f1_score']:.3f}")
    with col4:
        st.metric("Accuracy", f"{cm_results['accuracy']:.3f}")
    
    # Detailed breakdown
    with st.expander("Detailed Breakdown"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**True Positives (TP)**")
            st.info(f"Correctly identified outliers: {cm_results['true_positives']}")
            
            st.markdown("**False Positives (FP)**")
            st.warning(f"Normal points misclassified as outliers: {cm_results['false_positives']}")
        
        with col2:
            st.markdown("**True Negatives (TN)**")
            st.success(f"Correctly identified normal points: {cm_results['true_negatives']}")
            
            st.markdown("**False Negatives (FN)**")
            st.error(f"Outliers missed by the algorithm: {cm_results['false_negatives']}")

def compare_algorithm_performance(results_dict, true_outliers=None):
    """
    Compare performance of multiple algorithms
    
    Args:
        results_dict: Dictionary containing results from multiple algorithms
        true_outliers: True outlier indices (if available)
        
    Returns:
        dict: Comparison results
    """
    if true_outliers is None:
        return {'error': 'No ground truth available for comparison'}
    
    comparison_results = {}
    
    for algorithm, result in results_dict.items():
        if 'outliers' in result:
            # Create binary labels for this algorithm
            y_true = np.zeros(len(result.get('scores', [])))
            y_true[true_outliers] = 1
            
            y_pred = np.zeros(len(result.get('scores', [])))
            y_pred[result['outliers']] = 1
            
            # Calculate confusion matrix
            cm_results = calculate_confusion_matrix(y_true, y_pred)
            comparison_results[algorithm] = cm_results
    
    return comparison_results
