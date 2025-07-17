import streamlit as st
import pandas as pd
import numpy as np
import time
import io
from utils.data_processor import DataProcessor
from utils.outlier_detectors import OutlierDetectorFactory
from utils.visualizations import create_outlier_plot, create_comparison_plot, create_confusion_matrix_heatmap, create_performance_metrics_radar_chart
from utils.metrics import calculate_metrics, display_performance_metrics, calculate_confusion_matrix, display_confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Outlier Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'outlier_results' not in st.session_state:
    st.session_state.outlier_results = {}
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}

def main():
    st.title("🔍 Outlier Detection System")
    st.markdown("**Detect outliers in large datasets using distance-based algorithms**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=['csv'],
            help="Upload a CSV file with numerical columns for outlier detection"
        )
        
        if uploaded_file is not None:
            # Process uploaded data
            with st.spinner("Processing dataset..."):
                processor = DataProcessor()
                try:
                    st.session_state.processed_data = processor.process_file(uploaded_file)
                    st.success(f"Dataset loaded: {st.session_state.processed_data['data'].shape[0]} rows, {st.session_state.processed_data['data'].shape[1]} columns")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    return
        
        # Algorithm selection and parameters
        if st.session_state.processed_data is not None:
            st.subheader("Algorithm Selection")
            
            algorithms = st.multiselect(
                "Select Algorithms",
                ["K-Nearest Neighbors", "Local Outlier Factor", "Isolation Forest"],
                default=["Local Outlier Factor"]
            )
            
            if algorithms:
                st.subheader("Algorithm Parameters")
                
                # Distance metric selection
                distance_metric = st.selectbox(
                    "Distance Metric",
                    ["euclidean", "manhattan", "cosine"],
                    help="Distance metric for KNN and LOF algorithms"
                )
                
                # Common parameters
                contamination = st.slider(
                    "Contamination Rate",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.1,
                    step=0.01,
                    help="Expected proportion of outliers in the dataset"
                )
                
                # Algorithm-specific parameters
                params = {"contamination": contamination, "metric": distance_metric}
                
                if "K-Nearest Neighbors" in algorithms:
                    st.markdown("**KNN Parameters**")
                    params["knn_neighbors"] = st.slider(
                        "Number of Neighbors (KNN)",
                        min_value=1,
                        max_value=50,
                        value=20,
                        help="Number of neighbors for KNN algorithm"
                    )
                
                if "Local Outlier Factor" in algorithms:
                    st.markdown("**LOF Parameters**")
                    params["lof_neighbors"] = st.slider(
                        "Number of Neighbors (LOF)",
                        min_value=1,
                        max_value=50,
                        value=20,
                        help="Number of neighbors for LOF algorithm"
                    )
                
                if "Isolation Forest" in algorithms:
                    st.markdown("**Isolation Forest Parameters**")
                    params["n_estimators"] = st.slider(
                        "Number of Trees",
                        min_value=10,
                        max_value=500,
                        value=100,
                        help="Number of isolation trees"
                    )
                
                # Run detection button
                if st.button("🔍 Run Outlier Detection", type="primary"):
                    run_outlier_detection(algorithms, params)
    
    # Main content area
    if st.session_state.processed_data is not None:
        display_main_content()
    else:
        st.info("👆 Please upload a CSV dataset to begin outlier detection")

def run_outlier_detection(algorithms, params):
    """Run outlier detection with selected algorithms and parameters"""
    data = st.session_state.processed_data['data']
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.session_state.outlier_results = {}
    st.session_state.performance_metrics = {}
    
    for i, algorithm in enumerate(algorithms):
        status_text.text(f"Running {algorithm}...")
        
        start_time = time.time()
        
        try:
            # Create and run detector
            detector = OutlierDetectorFactory.create_detector(algorithm, params)
            outliers, scores = detector.detect_outliers(data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Store results
            st.session_state.outlier_results[algorithm] = {
                'outliers': outliers,
                'scores': scores,
                'execution_time': execution_time
            }
            
            # Calculate metrics
            metrics = calculate_metrics(outliers, scores)
            st.session_state.performance_metrics[algorithm] = metrics
            
        except Exception as e:
            st.error(f"Error running {algorithm}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(algorithms))
    
    status_text.text("✅ Detection complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

def display_main_content():
    """Display main content with results and visualizations"""
    data = st.session_state.processed_data['data']
    numerical_columns = st.session_state.processed_data['numerical_columns']
    
    # Dataset overview
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Numerical Columns", len(numerical_columns))
    with col3:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        if st.session_state.outlier_results:
            total_outliers = sum(len(result['outliers']) for result in st.session_state.outlier_results.values())
            st.metric("Total Outliers Found", total_outliers)
    
    # Data preview
    with st.expander("📋 Data Preview"):
        st.dataframe(data.head(100), use_container_width=True)
    
    # Results section
    if st.session_state.outlier_results:
        st.header("🎯 Detection Results")
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        display_performance_metrics(st.session_state.performance_metrics)
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        # Column selection for visualization
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("X-axis", numerical_columns, key="x_axis")
        with col2:
            y_column = st.selectbox("Y-axis", numerical_columns, index=1 if len(numerical_columns) > 1 else 0, key="y_axis")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Individual Results", "Algorithm Comparison", "Outlier Statistics", "Confusion Matrix", "Performance Metrics"])
        
        with tabs[0]:
            # Individual algorithm results
            for algorithm, result in st.session_state.outlier_results.items():
                st.markdown(f"**{algorithm}**")
                
                # Create visualization
                fig = create_outlier_plot(
                    data, 
                    result['outliers'], 
                    result['scores'],
                    x_column, 
                    y_column, 
                    algorithm
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Outliers Found", len(result['outliers']))
                with col2:
                    st.metric("Outlier Percentage", f"{len(result['outliers'])/len(data)*100:.2f}%")
                with col3:
                    st.metric("Execution Time", f"{result['execution_time']:.2f}s")
        
        with tabs[1]:
            # Algorithm comparison
            if len(st.session_state.outlier_results) > 1:
                fig = create_comparison_plot(
                    data,
                    st.session_state.outlier_results,
                    x_column,
                    y_column
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select multiple algorithms to see comparison")
        
        with tabs[2]:
            # Detailed outlier statistics
            st.subheader("📊 Outlier Statistics by Algorithm")
            
            for algorithm, result in st.session_state.outlier_results.items():
                with st.expander(f"{algorithm} - Detailed Statistics"):
                    outlier_data = data.iloc[result['outliers']]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Outlier Summary Statistics**")
                        st.dataframe(outlier_data.describe())
                    
                    with col2:
                        st.markdown("**Score Distribution**")
                        score_stats = {
                            'Min Score': np.min(result['scores']),
                            'Max Score': np.max(result['scores']),
                            'Mean Score': np.mean(result['scores']),
                            'Std Score': np.std(result['scores'])
                        }
                        for stat, value in score_stats.items():
                            st.metric(stat, f"{value:.4f}")
        
        with tabs[3]:
            # Confusion Matrix Tab
            st.subheader("🔍 Confusion Matrix Analysis")
            
            # Option to simulate ground truth for demonstration
            st.info("Note: For demonstration purposes, we simulate ground truth by using the top 10% most extreme outlier scores from the Local Outlier Factor algorithm as 'true outliers'.")
            
            # Add option to use any algorithm as ground truth
            if st.session_state.outlier_results:
                available_algorithms = list(st.session_state.outlier_results.keys())
                ground_truth_algorithm = st.selectbox(
                    "Select algorithm to use as ground truth:",
                    available_algorithms,
                    index=available_algorithms.index("Local Outlier Factor") if "Local Outlier Factor" in available_algorithms else 0
                )
                
                # Use selected algorithm as ground truth
                gt_result = st.session_state.outlier_results[ground_truth_algorithm]
                gt_scores = gt_result['scores']
                
                # Create simulated ground truth: top 10% most extreme scores
                threshold = np.percentile(gt_scores, 90)
                simulated_true_outliers = np.where(gt_scores > threshold)[0]
                
                st.markdown(f"**Simulated Ground Truth:** {len(simulated_true_outliers)} outliers (top 10% of {ground_truth_algorithm} scores)")
                
                # Calculate confusion matrix for each algorithm
                for algorithm, result in st.session_state.outlier_results.items():
                    if algorithm != ground_truth_algorithm:  # Skip ground truth algorithm
                        # Create binary labels
                        y_true = np.zeros(len(data))
                        y_true[simulated_true_outliers] = 1
                        
                        y_pred = np.zeros(len(data))
                        y_pred[result['outliers']] = 1
                        
                        # Calculate confusion matrix
                        cm_results = calculate_confusion_matrix(y_true, y_pred)
                        
                        # Display confusion matrix
                        st.markdown(f"### {algorithm}")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            display_confusion_matrix(cm_results, algorithm)
                        
                        with col2:
                            # Create confusion matrix heatmap
                            fig_cm = create_confusion_matrix_heatmap(cm_results, algorithm)
                            st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.warning("Please run outlier detection algorithms first to see confusion matrix analysis.")
        
        with tabs[4]:
            # Performance Metrics Tab
            st.subheader("📈 Performance Metrics Comparison")
            
            if st.session_state.outlier_results:
                available_algorithms = list(st.session_state.outlier_results.keys())
                ground_truth_algorithm = st.selectbox(
                    "Select ground truth algorithm:",
                    available_algorithms,
                    index=available_algorithms.index("Local Outlier Factor") if "Local Outlier Factor" in available_algorithms else 0,
                    key="performance_gt"
                )
                
                # Use selected algorithm as ground truth
                gt_result = st.session_state.outlier_results[ground_truth_algorithm]
                gt_scores = gt_result['scores']
                
                # Create simulated ground truth: top 10% most extreme scores
                threshold = np.percentile(gt_scores, 90)
                simulated_true_outliers = np.where(gt_scores > threshold)[0]
                
                # Calculate performance metrics for all algorithms
                performance_results = {}
                for algorithm, result in st.session_state.outlier_results.items():
                    if algorithm != ground_truth_algorithm:  # Skip ground truth algorithm
                        # Create binary labels
                        y_true = np.zeros(len(data))
                        y_true[simulated_true_outliers] = 1
                        
                        y_pred = np.zeros(len(data))
                        y_pred[result['outliers']] = 1
                        
                        # Calculate confusion matrix
                        cm_results = calculate_confusion_matrix(y_true, y_pred)
                        performance_results[algorithm] = cm_results
                
                if performance_results:
                    # Create radar chart
                    fig_radar = create_performance_metrics_radar_chart(performance_results)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Performance comparison table
                    st.subheader("📊 Performance Summary Table")
                    comparison_data = []
                    for algorithm, metrics in performance_results.items():
                        comparison_data.append({
                            'Algorithm': algorithm,
                            'Precision': f"{metrics['precision']:.3f}",
                            'Recall': f"{metrics['recall']:.3f}",
                            'F1-Score': f"{metrics['f1_score']:.3f}",
                            'Accuracy': f"{metrics['accuracy']:.3f}",
                            'Specificity': f"{metrics['specificity']:.3f}"
                        })
                    
                    performance_df = pd.DataFrame(comparison_data)
                    st.dataframe(performance_df, use_container_width=True)
                else:
                    st.info("Run multiple algorithms to see performance comparison.")
            else:
                st.warning("Please run outlier detection algorithms first to see performance metrics.")
        
        # Export functionality
        st.header("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export outliers
            if st.button("📥 Export Outliers"):
                export_outliers()
        
        with col2:
            # Export full results
            if st.button("📊 Export Full Results"):
                export_full_results()
        
        with col3:
            # Download PDF documentation
            if st.button("📄 Download PDF Documentation"):
                generate_and_download_pdf()

def export_outliers():
    """Export identified outliers to CSV"""
    data = st.session_state.processed_data['data']
    
    # Combine all outliers
    all_outliers = set()
    for result in st.session_state.outlier_results.values():
        all_outliers.update(result['outliers'])
    
    outlier_data = data.iloc[list(all_outliers)]
    
    # Add algorithm information
    for algorithm, result in st.session_state.outlier_results.items():
        outlier_data[f'{algorithm}_outlier'] = outlier_data.index.isin(result['outliers'])
    
    # Convert to CSV
    csv = outlier_data.to_csv(index=False)
    st.download_button(
        label="Download Outliers CSV",
        data=csv,
        file_name="outliers.csv",
        mime="text/csv"
    )

def export_full_results():
    """Export full results including scores and classifications"""
    data = st.session_state.processed_data['data'].copy()
    
    # Add results for each algorithm
    for algorithm, result in st.session_state.outlier_results.items():
        # Add outlier classification
        data[f'{algorithm}_outlier'] = False
        data.loc[result['outliers'], f'{algorithm}_outlier'] = True
        
        # Add outlier scores
        scores = np.full(len(data), 0.0)
        scores[result['outliers']] = np.array(result['scores'])[result['outliers']]
        data[f'{algorithm}_score'] = scores
    
    # Convert to CSV
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Full Results CSV",
        data=csv,
        file_name="full_results.csv",
        mime="text/csv"
    )

def generate_and_download_pdf():
    """Generate and provide PDF documentation for download"""
    try:
        # Import the PDF generator
        import sys
        sys.path.append('.')
        from generate_pdf_documentation import PDFDocumentGenerator
        
        # Generate PDF
        pdf_generator = PDFDocumentGenerator("Outlier_Detection_Documentation.pdf")
        pdf_generator.generate_complete_document()
        
        # Read the generated PDF
        with open("Outlier_Detection_Documentation.pdf", "rb") as pdf_file:
            pdf_data = pdf_file.read()
        
        # Provide download button
        st.download_button(
            label="Download PDF Documentation",
            data=pdf_data,
            file_name="Outlier_Detection_Documentation.pdf",
            mime="application/pdf"
        )
        
        st.success("PDF documentation generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        st.info("PDF documentation is available in the project files.")

if __name__ == "__main__":
    main()
