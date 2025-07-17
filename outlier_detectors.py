import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import distance_metrics
import time

class BaseOutlierDetector:
    """Base class for outlier detection algorithms"""
    
    def __init__(self, contamination=0.1, metric='euclidean'):
        self.contamination = contamination
        self.metric = metric
        self.model = None
        self.fitted = False
    
    def detect_outliers(self, data):
        """
        Detect outliers in the dataset
        
        Args:
            data: DataFrame or numpy array
            
        Returns:
            tuple: (outlier_indices, outlier_scores)
        """
        raise NotImplementedError("Subclasses must implement detect_outliers method")
    
    def fit(self, data):
        """Fit the outlier detection model"""
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, data):
        """Predict outliers on new data"""
        raise NotImplementedError("Subclasses must implement predict method")

class KNNOutlierDetector(BaseOutlierDetector):
    """K-Nearest Neighbors based outlier detector"""
    
    def __init__(self, contamination=0.1, metric='euclidean', n_neighbors=20):
        super().__init__(contamination, metric)
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    
    def detect_outliers(self, data):
        """
        Detect outliers using KNN distance approach
        
        Args:
            data: DataFrame or numpy array
            
        Returns:
            tuple: (outlier_indices, outlier_scores)
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Fit the model
        self.model.fit(data)
        
        # Calculate distances to k-nearest neighbors
        distances, indices = self.model.kneighbors(data)
        
        # Calculate outlier scores (mean distance to k nearest neighbors)
        outlier_scores = np.mean(distances, axis=1)
        
        # Determine outlier threshold
        threshold = np.percentile(outlier_scores, (1 - self.contamination) * 100)
        
        # Get outlier indices
        outlier_indices = np.where(outlier_scores > threshold)[0]
        
        return outlier_indices, outlier_scores
    
    def fit(self, data):
        """Fit the KNN model"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.model.fit(data)
        self.fitted = True
    
    def predict(self, data):
        """Predict outliers on new data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        distances, _ = self.model.kneighbors(data)
        outlier_scores = np.mean(distances, axis=1)
        
        return outlier_scores

class LOFOutlierDetector(BaseOutlierDetector):
    """Local Outlier Factor based outlier detector"""
    
    def __init__(self, contamination=0.1, metric='euclidean', n_neighbors=20):
        super().__init__(contamination, metric)
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric
        )
    
    def detect_outliers(self, data):
        """
        Detect outliers using Local Outlier Factor
        
        Args:
            data: DataFrame or numpy array
            
        Returns:
            tuple: (outlier_indices, outlier_scores)
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Fit and predict
        predictions = self.model.fit_predict(data)
        
        # Get outlier scores (negative values indicate outliers)
        outlier_scores = -self.model.negative_outlier_factor_
        
        # Get outlier indices (where prediction is -1)
        outlier_indices = np.where(predictions == -1)[0]
        
        return outlier_indices, outlier_scores
    
    def fit(self, data):
        """Fit the LOF model"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.model.fit(data)
        self.fitted = True
    
    def predict(self, data):
        """Predict outliers on new data"""
        # LOF doesn't support prediction on new data in the traditional sense
        # We need to create a new detector for new data
        raise NotImplementedError("LOF doesn't support prediction on new data")

class IsolationForestOutlierDetector(BaseOutlierDetector):
    """Isolation Forest based outlier detector"""
    
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        super().__init__(contamination)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    def detect_outliers(self, data):
        """
        Detect outliers using Isolation Forest
        
        Args:
            data: DataFrame or numpy array
            
        Returns:
            tuple: (outlier_indices, outlier_scores)
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Fit and predict
        predictions = self.model.fit_predict(data)
        
        # Get outlier scores (more negative = more outlier-like)
        outlier_scores = -self.model.decision_function(data)
        
        # Get outlier indices (where prediction is -1)
        outlier_indices = np.where(predictions == -1)[0]
        
        return outlier_indices, outlier_scores
    
    def fit(self, data):
        """Fit the Isolation Forest model"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.model.fit(data)
        self.fitted = True
    
    def predict(self, data):
        """Predict outliers on new data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        predictions = self.model.predict(data)
        outlier_scores = -self.model.decision_function(data)
        
        return predictions, outlier_scores

class OutlierDetectorFactory:
    """Factory class to create outlier detectors"""
    
    @staticmethod
    def create_detector(algorithm_name, params):
        """
        Create an outlier detector based on algorithm name and parameters
        
        Args:
            algorithm_name: Name of the algorithm
            params: Dictionary of parameters
            
        Returns:
            BaseOutlierDetector: Configured detector instance
        """
        contamination = params.get('contamination', 0.1)
        metric = params.get('metric', 'euclidean')
        
        if algorithm_name == "K-Nearest Neighbors":
            n_neighbors = params.get('knn_neighbors', 20)
            return KNNOutlierDetector(
                contamination=contamination,
                metric=metric,
                n_neighbors=n_neighbors
            )
        
        elif algorithm_name == "Local Outlier Factor":
            n_neighbors = params.get('lof_neighbors', 20)
            return LOFOutlierDetector(
                contamination=contamination,
                metric=metric,
                n_neighbors=n_neighbors
            )
        
        elif algorithm_name == "Isolation Forest":
            n_estimators = params.get('n_estimators', 100)
            return IsolationForestOutlierDetector(
                contamination=contamination,
                n_estimators=n_estimators
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    @staticmethod
    def get_available_algorithms():
        """Get list of available algorithms"""
        return ["K-Nearest Neighbors", "Local Outlier Factor", "Isolation Forest"]
    
    @staticmethod
    def get_algorithm_info(algorithm_name):
        """Get information about a specific algorithm"""
        info = {
            "K-Nearest Neighbors": {
                "description": "Distance-based outlier detection using k-nearest neighbors",
                "parameters": ["n_neighbors", "metric", "contamination"],
                "pros": ["Simple and intuitive", "Works well with local outliers"],
                "cons": ["Sensitive to curse of dimensionality", "Computationally expensive"]
            },
            "Local Outlier Factor": {
                "description": "Density-based outlier detection comparing local density",
                "parameters": ["n_neighbors", "metric", "contamination"],
                "pros": ["Good for varying density clusters", "Local outlier detection"],
                "cons": ["Sensitive to parameters", "Memory intensive"]
            },
            "Isolation Forest": {
                "description": "Tree-based outlier detection using isolation principle",
                "parameters": ["n_estimators", "contamination"],
                "pros": ["Fast and scalable", "Works well in high dimensions"],
                "cons": ["Less interpretable", "May miss local outliers"]
            }
        }
        
        return info.get(algorithm_name, {})

def compare_algorithms(data, algorithms, params):
    """
    Compare multiple outlier detection algorithms
    
    Args:
        data: Dataset to analyze
        algorithms: List of algorithm names
        params: Dictionary of parameters
        
    Returns:
        dict: Results for each algorithm
    """
    results = {}
    
    for algorithm in algorithms:
        try:
            start_time = time.time()
            
            # Create detector
            detector = OutlierDetectorFactory.create_detector(algorithm, params)
            
            # Detect outliers
            outlier_indices, outlier_scores = detector.detect_outliers(data)
            
            end_time = time.time()
            
            # Store results
            results[algorithm] = {
                'outlier_indices': outlier_indices,
                'outlier_scores': outlier_scores,
                'execution_time': end_time - start_time,
                'n_outliers': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(data) * 100
            }
            
        except Exception as e:
            results[algorithm] = {
                'error': str(e),
                'execution_time': 0,
                'n_outliers': 0,
                'outlier_percentage': 0
            }
    
    return results
