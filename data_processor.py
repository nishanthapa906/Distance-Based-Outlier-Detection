import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataProcessor:
    """Handles data loading, preprocessing, and validation for outlier detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
    
    def process_file(self, uploaded_file):
        """
        Process uploaded CSV file and prepare it for outlier detection
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            dict: Processed data with metadata
        """
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate dataset
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Identify numerical columns
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_columns) == 0:
                raise ValueError("No numerical columns found in the dataset")
            
            # Handle missing values
            df_numeric = df[numerical_columns].copy()
            
            # Check for missing values
            missing_counts = df_numeric.isnull().sum()
            if missing_counts.any():
                st.warning(f"Missing values found in columns: {missing_counts[missing_counts > 0].to_dict()}")
                st.info("Missing values will be imputed with mean values")
                
                # Impute missing values
                df_numeric = pd.DataFrame(
                    self.imputer.fit_transform(df_numeric),
                    columns=numerical_columns,
                    index=df_numeric.index
                )
            
            # Check for infinite values
            inf_mask = np.isinf(df_numeric.values)
            if inf_mask.any():
                st.warning("Infinite values detected and will be replaced with NaN, then imputed")
                df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
                df_numeric = pd.DataFrame(
                    self.imputer.fit_transform(df_numeric),
                    columns=numerical_columns,
                    index=df_numeric.index
                )
            
            # Validate data size
            if len(df_numeric) < 10:
                raise ValueError("Dataset too small. Need at least 10 rows for meaningful outlier detection")
            
            # Scale the data for better algorithm performance
            df_scaled = pd.DataFrame(
                self.scaler.fit_transform(df_numeric),
                columns=numerical_columns,
                index=df_numeric.index
            )
            
            # Prepare metadata
            metadata = {
                'total_rows': len(df),
                'numerical_columns': len(numerical_columns),
                'missing_values': missing_counts.sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            return {
                'data': df_scaled,
                'original_data': df_numeric,
                'full_data': df,
                'numerical_columns': numerical_columns,
                'metadata': metadata
            }
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def get_data_summary(self, data):
        """
        Generate summary statistics for the dataset
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            dict: Summary statistics
        """
        return {
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'describe': data.describe(),
            'null_counts': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
    
    def validate_columns_for_detection(self, data, min_columns=2):
        """
        Validate that the dataset has enough columns for detection
        
        Args:
            data: DataFrame to validate
            min_columns: Minimum number of columns required
            
        Returns:
            bool: True if validation passes
        """
        if data.shape[1] < min_columns:
            raise ValueError(f"Need at least {min_columns} numerical columns for outlier detection")
        
        return True
    
    def prepare_for_algorithm(self, data, algorithm_type):
        """
        Prepare data specifically for different algorithm types
        
        Args:
            data: DataFrame to prepare
            algorithm_type: Type of algorithm ('knn', 'lof', 'isolation_forest')
            
        Returns:
            np.ndarray: Prepared data array
        """
        # Convert to numpy array
        data_array = data.values
        
        # Algorithm-specific preparations
        if algorithm_type in ['knn', 'lof']:
            # These algorithms work better with standardized data
            if not hasattr(self.scaler, 'mean_'):
                # If scaler not fitted, fit it
                self.scaler.fit(data_array)
            return data_array
        
        elif algorithm_type == 'isolation_forest':
            # Isolation Forest can work with original scale
            return data_array
        
        return data_array
    
    def get_feature_importance(self, data):
        """
        Calculate feature importance based on variance
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            dict: Feature importance scores
        """
        variances = data.var()
        normalized_variances = variances / variances.sum()
        
        return normalized_variances.to_dict()
