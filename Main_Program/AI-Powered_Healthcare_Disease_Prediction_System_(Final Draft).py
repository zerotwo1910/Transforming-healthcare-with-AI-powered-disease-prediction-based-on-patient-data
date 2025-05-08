import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import os
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HealthcareAI:
    """
    A comprehensive AI-powered healthcare disease prediction system
    that integrates data preprocessing, multiple ML models, evaluation
    metrics, and a user-friendly interface.
    """
    
    def __init__(self):
        self.dataset = None
        self.features = None
        self.target = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.model_accuracies = {}
        self.feature_importances = None
        
    def load_data(self, filepath):
        """Load and prepare the dataset"""
        try:
            self.dataset = pd.read_csv(filepath)
            logging.info(f"Dataset loaded successfully with {self.dataset.shape[0]} records and {self.dataset.shape[1]} features.")
            return True
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        """Return basic statistics and information about the dataset"""
        if self.dataset is None:
            return "No dataset loaded"
        
        info = {
            "shape": self.dataset.shape,
            "columns": self.dataset.columns.tolist(),
            "missing_values": self.dataset.isnull().sum().to_dict(),
            "summary_stats": self.dataset.describe().to_dict(),
        }
        return info
    
    def preprocess_data(self, target_column, features_to_use=None, test_size=0.2):
        """Preprocess the data and prepare for model training"""
        if self.dataset is None:
            logging.error("No dataset loaded")
            return False
            
        if target_column not in self.dataset.columns:
            logging.error(f"Target column '{target_column}' not found in dataset")
            return False
        
        if features_to_use and not all(f in self.dataset.columns for f in features_to_use):
            missing_features = set(features_to_use) - set(self.dataset.columns)
            logging.error(f"Some features not found in dataset: {missing_features}")
            return False
        
        # Check for significant missing data
        missing_ratio = self.dataset.isnull().sum() / len(self.dataset)
        if any(missing_ratio > 0.5):
            logging.warning("Some columns have more than 50% missing values")
        
        # Handle missing values
        for col in self.dataset.columns:
            if self.dataset[col].isnull().sum() > 0:
                if self.dataset[col].dtype in [np.float64, np.int64]:
                    self.dataset[col].fillna(self.dataset[col].median(), inplace=True)
                else:
                    self.dataset[col].fillna(self.dataset[col].mode()[0], inplace=True)
        
        # Set target and features
        self.target = self.dataset[target_column]
        
        if features_to_use:
            self.features = self.dataset[features_to_use]
        else:
            self.features = self.dataset.drop(target_column, axis=1)
        
        # Encode categorical features
        categorical_cols = self.features.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logging.info(f"Encoding categorical features: {categorical_cols}")
            self.features = pd.get_dummies(self.features, columns=categorical_cols)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42)
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logging.info("Data preprocessing completed successfully")
        return True
    
    def train_models(self):
        """Train multiple models and select the best one"""
        if not hasattr(self, 'X_train_scaled'):
            logging.error("Please preprocess the data first")
            return False
        
        logging.info("Training models...")
        
        # Initialize models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            logging.info(f"Training {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            self.models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            mean_cv_score = np.mean(cv_scores)
            
            # Test set accuracy
            y_pred = model.predict(self.X_test_scaled)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            self.model_accuracies[name] = {
                "cv_score": mean_cv_score,
                "test_accuracy": test_accuracy
            }
            
            logging.info(f"{name} - CV Score: {mean_cv_score:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Find the best model based on test accuracy
        self.best_model_name = max(self.model_accuracies, key=lambda x: self.model_accuracies[x]["test_accuracy"])
        self.best_model = self.models[self.best_model_name]
        
        logging.info(f"Best model: {self.best_model_name} with accuracy: {self.model_accuracies[self.best_model_name]['test_accuracy']:.4f}")
        
        # Get feature importances if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importances = dict(zip(self.features.columns, self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'coef_'):
            coef = self.best_model.coef_[0] if self.best_model.coef_.ndim == 2 else self.best_model.coef_
            self.feature_importances = dict(zip(self.features.columns, coef))
            
        return True
    
    def optimize_model(self):
        """Optimize the best model using grid search"""
        if self.best_model_name is None:
            logging.error("Please train models first")
            return False
            
        logging.info(f"Optimizing {self.best_model_name}...")
        
        param_grids = {
            "Logistic Regression": {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        grid = GridSearchCV(
            self.best_model, 
            param_grids[self.best_model_name], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid.fit(self.X_train_scaled, self.y_train)
        self.best_model = grid.best_estimator_
        self.models[self.best_model_name] = self.best_model
        
        # Update accuracy
        y_pred = self.best_model.predict(self.X_test_scaled)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        self.model_accuracies[self.best_model_name]["test_accuracy"] = test_accuracy
        
        logging.info(f"Optimized {self.best_model_name} - Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"Best parameters: {grid.best_params_}")
        
        return True
    
    def evaluate_model(self):
        """Evaluate the best model and return performance metrics"""
        if self.best_model is None:
            return "No model trained"
            
        y_pred = self.best_model.predict(self.X_test_scaled)
        y_prob = self.best_model.predict_proba(self.X_test_scaled)[:, 1] if len(self.best_model.classes_) == 2 else None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # ROC curve (only for binary classification)
        roc_data = None
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "roc_curve": roc_data
        }
    
    def predict(self, patient_data, threshold=0.5):
        """Predict disease probability for new patient data"""
        if self.best_model is None:
            logging.error("No trained model available")
            return None
            
        # Validate input features
        required_features = set(self.features.columns)
        provided_features = set(patient_data.keys())
        missing_features = required_features - provided_features
        if missing_features:
            logging.error(f"Missing required features: {missing_features}")
            return None
        
        # Convert to DataFrame to ensure feature order
        df = pd.DataFrame([patient_data])
        
        # Align with training features
        for col in self.features.columns:
            if col not in df.columns:
                df[col] = 0  # Default value
        
        df = df[self.features.columns]  # Reorder columns to match training data
        
        # Scale the input data
        scaled_data = self.scaler.transform(df)
        
        # Make prediction
        probability = self.best_model.predict_proba(scaled_data)[:, 1][0] if len(self.best_model.classes_) == 2 else None
        prediction = 1 if probability >= threshold else 0 if probability is not None else None
        
        return {
            "probability": probability,
            "prediction": prediction
        }
    
    def save_model(self, filepath):
        """Save the trained model and associated data"""
        if self.best_model is None:
            logging.error("No trained model to save")
            return False
            
        model_data = {
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "scaler": self.scaler,
            "feature_names": self.features.columns.tolist(),
            "target_name": self.target.name,
            "model_accuracies": self.model_accuracies,
            "feature_importances": self.feature_importances
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model successfully saved to {filepath}")
        return True
    
    def load_model(self, filepath):
        """Load a previously saved model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.best_model = model_data["best_model"]
            self.best_model_name = model_data["best_model_name"]
            self.scaler = model_data["scaler"]
            self.feature_importances = model_data["feature_importances"]
            self.model_accuracies = model_data["model_accuracies"]
            
            logging.info(f"Model successfully loaded from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importances(self):
        """Get feature importances of the best model"""
        return self.feature_importances
        
    def visualize_roc_curve(self, fig=None, ax=None):
        """Visualize ROC curve of the best model"""
        if self.best_model is None:
            logging.error("No trained model available")
            return None
            
        if len(self.best_model.classes_) != 2:
            logging.warning("ROC curve visualization is only supported for binary classification")
            return None
            
        y_prob = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        return fig
    
    def visualize_confusion_matrix(self, fig=None, ax=None):
        """Visualize confusion matrix of the best model"""
        if self.best_model is None:
            logging.error("No trained model available")
            return None
            
        y_pred = self.best_model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred)
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        
        return fig
    
    def visualize_feature_importance(self, fig=None, ax=None):
        """Visualize feature importances of the best model"""
        if self.feature_importances is None:
            logging.error("No feature importances available")
            return None
            
        # Sort features by importance
        sorted_features = dict(sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True))
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        bars = ax.barh(list(sorted_features.keys()), list(sorted_features.values()))
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        
        # Add values to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        return fig

    def visualize_correlation_matrix(self, fig=None, ax=None):
        """Visualize correlation matrix of the features"""
        if self.features is None:
            logging.error("No features available")
            return None
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        # Calculate correlation matrix
        corr_matrix = self.features.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    vmin=-1, vmax=1, ax=ax, fmt='.2f')
        ax.set_title('Feature Correlation Matrix')
        
        return fig

    def visualize_patient_distribution(self, fig=None, ax=None):
        """Visualize distribution of patients with/without heart disease before and after prediction"""
        if self.best_model is None or not hasattr(self, 'y_test'):
            logging.error("No trained model or test data available")
            return None
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Get actual and predicted distributions
        y_pred = self.best_model.predict(self.X_test_scaled)
        actual_counts = pd.Series(self.y_test).value_counts().sort_index()
        predicted_counts = pd.Series(y_pred).value_counts().sort_index()
        
        # Create bar plot
        labels = ['No Disease', 'Disease']
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, actual_counts, width, label='Actual', color='skyblue')
        ax.bar(x + width/2, predicted_counts, width, label='Predicted', color='salmon')
        
        ax.set_ylabel('Number of Patients')
        ax.set_title('Patient Distribution: Actual vs Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(actual_counts):
            ax.text(i - width/2, v + 0.5, str(v), ha='center')
        for i, v in enumerate(predicted_counts):
            ax.text(i + width/2, v + 0.5, str(v), ha='center')
        
        return fig

class HealthcareAI_GUI:
    """
    GUI for the HealthcareAI system
    """
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Healthcare Disease Prediction System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        self.ai_system = HealthcareAI()
        self.dataset_loaded = False
        self.model_trained = False
        
        # Setup the main frames
        self.setup_frames()
        self.setup_menu()
        self.setup_notebook()
        
        # Setup status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_frames(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
    def setup_menu(self):
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Dataset", command=self.load_dataset)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Model menu
        model_menu = tk.Menu(menu_bar, tearoff=0)
        model_menu.add_command(label="Preprocess Data", command=self.preprocess_data)
        model_menu.add_command(label="Train Models", command=self.train_models)
        model_menu.add_command(label="Optimize Model", command=self.optimize_model)
        model_menu.add_command(label="Evaluate Model", command=self.evaluate_model)
        menu_bar.add_cascade(label="Model", menu=model_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
        
    def setup_notebook(self):
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data Explorer")
        self.notebook.add(self.model_tab, text="Model Training")
        self.notebook.add(self.visualization_tab, text="Visualizations")
        self.notebook.add(self.prediction_tab, text="Disease Prediction")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_prediction_tab()
        self.setup_visualization_tab()
        
    def setup_data_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.data_tab, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Load dataset button
        load_btn = ttk.Button(left_frame, text="Load Dataset", command=self.load_dataset)
        load_btn.pack(fill=tk.X, pady=5)
        
        # Dataset info
        info_frame = ttk.LabelFrame(left_frame, text="Dataset Info", padding="10")
        info_frame.pack(fill=tk.X, pady=10)
        
        self.dataset_info_var = tk.StringVar()
        self.dataset_info_var.set("No dataset loaded")
        info_label = ttk.Label(info_frame, textvariable=self.dataset_info_var)
        info_label.pack(fill=tk.X)
        
        # Right frame for data preview
        right_frame = ttk.Frame(self.data_tab, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Data preview
        preview_frame = ttk.LabelFrame(right_frame, text="Data Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for data preview
        self.data_tree = ttk.Treeview(preview_frame)
        self.data_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars for treeview
        vsb = ttk.Scrollbar(self.data_tree, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(self.data_tree, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_model_tab(self):
        # Left frame for controls
        left_frame = ttk.Frame(self.model_tab, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Target selection
        target_frame = ttk.LabelFrame(left_frame, text="Target Selection", padding="10")
        target_frame.pack(fill=tk.X, pady=5)
        
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(target_frame, textvariable=self.target_var, state="readonly")
        self.target_combo.pack(fill=tk.X, pady=5)
        
        # Feature selection
        feature_frame = ttk.LabelFrame(left_frame, text="Feature Selection", padding="10")
        feature_frame.pack(fill=tk.X, pady=5)
        
        # Create a frame with scrollbar for feature checkboxes
        scroll_frame = ttk.Frame(feature_frame)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.feature_canvas = tk.Canvas(scroll_frame)
        self.feature_scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.feature_canvas.yview)
        self.feature_frame_inner = ttk.Frame(self.feature_canvas)
        
        self.feature_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.feature_canvas.create_window((0, 0), window=self.feature_frame_inner, anchor="nw")
        self.feature_canvas.configure(yscrollcommand=self.feature_scrollbar.set)
        self.feature_frame_inner.bind("<Configure>", lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all")))
        
        # Feature checkboxes will be added dynamically when dataset is loaded
        self.feature_vars = {}
        
        # Preprocessing button
        preprocess_btn = ttk.Button(left_frame, text="Preprocess Data", command=self.preprocess_data)
        preprocess_btn.pack(fill=tk.X, pady=10)
        
        # Training buttons
        train_frame = ttk.LabelFrame(left_frame, text="Model Training", padding="10")
        train_frame.pack(fill=tk.X, pady=5)
        
        train_btn = ttk.Button(train_frame, text="Train Models", command=self.train_models)
        train_btn.pack(fill=tk.X, pady=5)
        
        optimize_btn = ttk.Button(train_frame, text="Optimize Best Model", command=self.optimize_model)
        optimize_btn.pack(fill=tk.X, pady=5)
        
        evaluate_btn = ttk.Button(train_frame, text="Evaluate Model", command=self.evaluate_model)
        evaluate_btn.pack(fill=tk.X, pady=5)
        
        # Right frame for results
        right_frame = ttk.Frame(self.model_tab, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Training results
        results_frame = ttk.LabelFrame(right_frame, text="Training Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for results text
        results_sb = ttk.Scrollbar(self.results_text, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_sb.set)
        results_sb.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_prediction_tab(self):
        # Left frame for input
        left_frame = ttk.Frame(self.prediction_tab, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Input form
        input_frame = ttk.LabelFrame(left_frame, text="Patient Data Input", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas with scrollbar for input fields
        canvas_frame = ttk.Frame(input_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.input_canvas = tk.Canvas(canvas_frame)
        self.input_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.input_canvas.yview)
        self.input_frame_inner = ttk.Frame(self.input_canvas)
        
        self.input_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.input_canvas.create_window((0, 0), window=self.input_frame_inner, anchor="nw")
        self.input_canvas.configure(yscrollcommand=self.input_scrollbar.set)
        self.input_frame_inner.bind("<Configure>", lambda e: self.input_canvas.configure(scrollregion=self.input_canvas.bbox("all")))
        
        # Input fields will be added dynamically when dataset is loaded
        self.input_fields = {}
        
        # Right frame for prediction results
        right_frame = ttk.Frame(self.prediction_tab, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Prediction results
        prediction_frame = ttk.LabelFrame(right_frame, text="Prediction Results", padding="10")
        prediction_frame.pack(fill=tk.BOTH, expand=True)
        
        self.prediction_result_var = tk.StringVar()
        self.prediction_result_var.set("No prediction made yet")
        
        # Result display with large font
        result_font = ("Arial", 16, "bold")
        self.prediction_label = ttk.Label(prediction_frame, textvariable=self.prediction_result_var, font=result_font)
        self.prediction_label.pack(pady=20)
        
        # Probability gauge
        self.prob_canvas = tk.Canvas(prediction_frame, width=300, height=150, bg="white")
        self.prob_canvas.pack(pady=10)
        
        # Prediction button
        predict_btn = ttk.Button(prediction_frame, text="Make Prediction", command=self.make_prediction)
        predict_btn.pack(pady=10)
        
    def setup_visualization_tab(self):
        # Controls frame
        controls_frame = ttk.Frame(self.visualization_tab, padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Visualization type selection
        viz_label = ttk.Label(controls_frame, text="Select Visualization:")
        viz_label.pack(side=tk.LEFT, padx=5)
        
        self.viz_type_var = tk.StringVar()
        viz_options = ["ROC Curve", "Confusion Matrix", "Feature Importance", 
                      "Correlation Matrix", "Patient Distribution"]
        viz_combo = ttk.Combobox(controls_frame, textvariable=self.viz_type_var, 
                               values=viz_options, state="readonly")
        viz_combo.pack(side=tk.LEFT, padx=5)
        viz_combo.bind("<<ComboboxSelected>>", self.update_visualization)
        
        # Plot frame
        self.plot_frame = ttk.LabelFrame(self.visualization_tab, text="Visualization", padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Default message
        self.viz_message_var = tk.StringVar()
        self.viz_message_var.set("Train a model to see visualizations")
        self.viz_message_label = ttk.Label(self.plot_frame, textvariable=self.viz_message_var)
        self.viz_message_label.pack(expand=True)
        
    def load_dataset(self):
        filepath = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
            
        # Load the dataset
        if self.ai_system.load_data(filepath):
            self.status_var.set(f"Dataset loaded: {os.path.basename(filepath)}")
            self.dataset_loaded = True
            
            # Update dataset info
            info = self.ai_system.explore_data()
            self.dataset_info_var.set(f"Rows: {info['shape'][0]}, Columns: {info['shape'][1]}")
            
            # Display data preview
            self.update_data_preview()
            
            # Update target combobox
            self.update_target_combo()
            
            # Update feature checkboxes
            self.update_feature_checkboxes()
            
            # Create input fields for prediction
            self.create_input_fields()
            
            messagebox.showinfo("Success", "Dataset loaded successfully")
        else:
            messagebox.showerror("Error", "Failed to load dataset")
            
    def update_data_preview(self):
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
            
        # Configure columns
        self.data_tree["columns"] = list(self.ai_system.dataset.columns)
        self.data_tree["show"] = "headings"
        
        for col in self.ai_system.dataset.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
            
        # Add data rows (first 50 rows for preview)
        for i, row in self.ai_system.dataset.head(50).iterrows():
            self.data_tree.insert("", "end", values=list(row))
            
    def update_target_combo(self):
        # Update target selection dropdown
        self.target_combo["values"] = list(self.ai_system.dataset.columns)
        if "target" in self.ai_system.dataset.columns:
            self.target_var.set("target")
        else:
            self.target_var.set(self.ai_system.dataset.columns[-1])
            
    def update_feature_checkboxes(self):
        # Clear existing checkboxes
        for widget in self.feature_frame_inner.winfo_children():
            widget.destroy()
            
        # Create checkboxes for each column
        self.feature_vars = {}
        for col in self.ai_system.dataset.columns:
            if col != self.target_var.get():  # Skip target column
                self.feature_vars[col] = tk.BooleanVar(value=True)
                chk = ttk.Checkbutton(self.feature_frame_inner, text=col, variable=self.feature_vars[col])
                chk.pack(anchor=tk.W)
                
    def create_input_fields(self):
        # Clear existing fields
        for widget in self.input_frame_inner.winfo_children():
            widget.destroy()
            
        # Create entry fields for each column (except target)
        self.input_fields = {}
        row = 0
        for col in self.ai_system.dataset.columns:
            if col != self.target_var.get():  # Skip target column
                frame = ttk.Frame(self.input_frame_inner)
                frame.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
                
                # Label
                label = ttk.Label(frame, text=f"{col}:", width=15, anchor=tk.E)
                label.grid(row=0, column=0, padx=5, pady=2)
                
                # Entry
                entry = ttk.Entry(frame, width=20)
                entry.grid(row=0, column=1, padx=5, pady=2)
                
                self.input_fields[col] = entry
                row += 1
                
        # Add sample values button
        sample_btn = ttk.Button(self.input_frame_inner, text="Fill with Sample Values", command=self.fill_sample_values)
        sample_btn.grid(row=row, column=0, sticky="ew", padx=5, pady=10)
                
    def fill_sample_values(self):
        # Fill input fields with sample values from dataset
        if not self.dataset_loaded:
            return
            
        # Get a random sample from dataset
        sample = self.ai_system.dataset.sample(1).iloc[0]
        
        for col, entry in self.input_fields.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(sample[col]))
            
    def preprocess_data(self):
        if not self.dataset_loaded:
            self.root.after(0, lambda: messagebox.showinfo("Info", "Please load a dataset first"))
            return
            
        # Get selected target
        target = self.target_var.get()
        
        # Get selected features
        selected_features = [col for col, var in self.feature_vars.items() if var.get()]
        
        if not selected_features:
            self.root.after(0, lambda: messagebox.showinfo("Info", "Please select at least one feature"))
            return
        
        # Preprocess data
        def preprocess_task():
            try:
                self.root.after(0, lambda: self.results_text.insert(tk.END, "\nStarting preprocessing...\n"))
                self.root.after(0, lambda: self.results_text.see(tk.END))
                
                self.root.after(0, lambda: self.results_text.insert(tk.END, "Handling missing values...\n"))
                self.root.after(0, lambda: self.results_text.see(tk.END))
                
                success = self.ai_system.preprocess_data(target, selected_features)
                
                if success:
                    self.root.after(0, lambda: self.status_var.set("Data preprocessing completed"))
                    self.root.after(0, lambda: self.results_text.insert(tk.END, f"\n--- Data Preprocessing ---\n"))
                    self.root.after(0, lambda: self.results_text.insert(tk.END, f"Target: {target}\n"))
                    self.root.after(0, lambda: self.results_text.insert(tk.END, f"Features: {', '.join(selected_features)}\n"))
                    self.root.after(0, lambda: self.results_text.insert(tk.END, f"Training set: {len(self.ai_system.X_train)} samples\n"))
                    self.root.after(0, lambda: self.results_text.insert(tk.END, f"Test set: {len(self.ai_system.X_test)} samples\n"))
                    self.root.after(0, lambda: self.results_text.see(tk.END))
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Data preprocessing completed"))
                else:
                    self.root.after(0, lambda: self.status_var.set("Data preprocessing failed"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Data preprocessing failed"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"\nError during preprocessing: {str(e)}\n"))
                self.root.after(0, lambda: self.results_text.see(tk.END))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Data preprocessing failed: {str(e)}"))
        
        threading.Thread(target=preprocess_task).start()
        self.status_var.set("Preprocessing data...")
            
    def train_models(self):
        if not hasattr(self.ai_system, 'X_train_scaled'):
            self.root.after(0, lambda: messagebox.showinfo("Info", "Please preprocess the data first"))
            return
            
        # Train models in a separate thread
        def train_task():
            progress = ttk.Progressbar(self.model_tab, mode='indeterminate')
            self.root.after(0, lambda: progress.pack(fill=tk.X, pady=5))
            progress.start()
            
            success = self.ai_system.train_models()
            
            if success:
                self.root.after(0, lambda: self.status_var.set("Model training completed"))
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"\n--- Model Training Results ---\n"))
                
                for model_name, metrics in self.ai_system.model_accuracies.items():
                    self.root.after(0, lambda mn=model_name, m=metrics: self.results_text.insert(tk.END, f"{mn}:\n"))
                    self.root.after(0, lambda mn=model_name, m=metrics: self.results_text.insert(tk.END, f"  Cross-validation score: {m['cv_score']:.4f}\n"))
                    self.root.after(0, lambda mn=model_name, m=metrics: self.results_text.insert(tk.END, f"  Test accuracy: {m['test_accuracy']:.4f}\n"))
                
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"\nBest model: {self.ai_system.best_model_name} with accuracy: {self.ai_system.model_accuracies[self.ai_system.best_model_name]['test_accuracy']:.4f}\n"))
                self.root.after(0, lambda: self.results_text.see(tk.END))
                
                self.model_trained = True
                self.root.after(0, self.update_visualization)
                self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed"))
            else:
                self.root.after(0, lambda: self.status_var.set("Model training failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Model training failed"))
            
            self.root.after(0, lambda: progress.stop())
            self.root.after(0, lambda: progress.destroy())
        
        threading.Thread(target=train_task).start()
        self.status_var.set("Training models...")
            
    def optimize_model(self):
        if not self.model_trained:
            self.root.after(0, lambda: messagebox.showinfo("Info", "Please train models first"))
            return
            
        # Optimize model in a separate thread
        def optimize_task():
            progress = ttk.Progressbar(self.model_tab, mode='indeterminate')
            self.root.after(0, lambda: progress.pack(fill=tk.X, pady=5))
            progress.start()
            
            success = self.ai_system.optimize_model()
            
            if success:
                self.root.after(0, lambda: self.status_var.set("Model optimization completed"))
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"\n--- Model Optimization Results ---\n"))
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"Optimized {self.ai_system.best_model_name}:\n"))
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"  Test accuracy: {self.ai_system.model_accuracies[self.ai_system.best_model_name]['test_accuracy']:.4f}\n"))
                self.root.after(0, lambda: self.results_text.see(tk.END))
                
                self.root.after(0, self.update_visualization)
                self.root.after(0, lambda: messagebox.showinfo("Success", "Model optimization completed"))
            else:
                self.root.after(0, lambda: self.status_var.set("Model optimization failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Model optimization failed"))
            
            self.root.after(0, lambda: progress.stop())
            self.root.after(0, lambda: progress.destroy())
        
        threading.Thread(target=optimize_task).start()
        self.status_var.set("Optimizing model...")
            
    def evaluate_model(self):
        if not self.model_trained:
            self.root.after(0, lambda: messagebox.showinfo("Info", "Please train models first"))
            return
            
        # Evaluate model
        results = self.ai_system.evaluate_model()
        
        if results != "No model trained":
            self.root.after(0, lambda: self.status_var.set("Model evaluation completed"))
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"\n--- Model Evaluation ---\n"))
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"Accuracy: {results['accuracy']:.4f}\n\n"))
            
            # Classification report
            report = results['classification_report']
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Classification Report:\n"))
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    self.root.after(0, lambda l=label, m=metrics: self.results_text.insert(tk.END, f"  Class {l}:\n"))
                    self.root.after(0, lambda l=label, m=metrics: self.results_text.insert(tk.END, f"    Precision: {m['precision']:.4f}\n"))
                    self.root.after(0, lambda l=label, m=metrics: self.results_text.insert(tk.END, f"    Recall: {m['recall']:.4f}\n"))
                    self.root.after(0, lambda l=label, m=metrics: self.results_text.insert(tk.END, f"    F1-score: {m['f1-score']:.4f}\n"))
                    self.root.after(0, lambda l=label, m=metrics: self.results_text.insert(tk.END, f"    Support: {m['support']}\n"))
            
            if results['roc_curve']:
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"\nROC AUC: {results['roc_curve']['auc']:.4f}\n"))
            self.root.after(0, lambda: self.results_text.see(tk.END))
            
            # Update visualization
            self.root.after(0, self.update_visualization)
        else:
            self.root.after(0, lambda: messagebox.showinfo("Info", "No model trained yet"))
            
    def update_visualization(self, event=None):
        if not self.model_trained and self.viz_type_var.get() != "Correlation Matrix":
            self.root.after(0, lambda: self.viz_message_var.set("Train a model to see visualizations"))
            return
            
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Create figure and canvas
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        viz_type = self.viz_type_var.get() if self.viz_type_var.get() else "ROC Curve"
        
        if viz_type == "ROC Curve":
            fig = self.ai_system.visualize_roc_curve(fig, ax)
        elif viz_type == "Confusion Matrix":
            fig = self.ai_system.visualize_confusion_matrix(fig, ax)
        elif viz_type == "Feature Importance":
            fig = self.ai_system.visualize_feature_importance(fig, ax)
        elif viz_type == "Correlation Matrix":
            fig = self.ai_system.visualize_correlation_matrix(fig, ax)
        elif viz_type == "Patient Distribution":
            fig = self.ai_system.visualize_patient_distribution(fig, ax)
            
        if fig:
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.root.after(0, lambda: self.viz_message_var.set("Visualization not available"))
            
    def make_prediction(self):
        if not self.model_trained:
            self.root.after(0, lambda: messagebox.showinfo("Info", "Please train a model first"))
            return
            
        # Collect input data
        patient_data = {}
        for col, entry in self.input_fields.items():
            try:
                value = entry.get().strip()
                # Check expected type based on dataset
                expected_type = self.ai_system.dataset[col].dtype
                if np.issubdtype(expected_type, np.number):
                    try:
                        value = float(value)
                    except ValueError:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"Invalid numeric value for {col}"))
                        return
                patient_data[col] = value
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Invalid value for {col}: {str(e)}"))
                return
                
        # Make prediction
        result = self.ai_system.predict(patient_data)
        
        if result and result['probability'] is not None:
            probability = result["probability"]
            prediction = result["prediction"]
            
            # Update prediction text
            if prediction == 1:
                self.root.after(0, lambda: self.prediction_result_var.set(f"Disease Predicted\nProbability: {probability:.2%}"))
                self.root.after(0, lambda: self.prediction_label.configure(foreground="red"))
            else:
                self.root.after(0, lambda: self.prediction_result_var.set(f"No Disease Predicted\nProbability: {probability:.2%}"))
                self.root.after(0, lambda: self.prediction_label.configure(foreground="green"))
                
            # Update probability gauge
            self.root.after(0, lambda: self.update_probability_gauge(probability))
        else:
            self.root.after(0, lambda: messagebox.showerror("Error", "Prediction failed or not supported for multi-class"))
            
    def update_probability_gauge(self, probability):
        # Clear canvas
        self.prob_canvas.delete("all")
        
        # Draw gauge
        width = self.prob_canvas.winfo_width()
        height = self.prob_canvas.winfo_height()
        
        # Background
        self.prob_canvas.create_rectangle(10, height-30, width-10, height-10, fill="lightgray", outline="gray")
        
        # Value
        gauge_width = int((width-20) * probability)
        color = self.get_color_gradient(probability)
        self.prob_canvas.create_rectangle(10, height-30, 10+gauge_width, height-10, fill=color, outline="")
        
        # Percentage text
        self.prob_canvas.create_text(width/2, height-20, text=f"{probability:.1%}", font=("Arial", 12, "bold"))
        
        # Labels
        self.prob_canvas.create_text(10, height-40, text="0%", anchor=tk.W)
        self.prob_canvas.create_text(width-10, height-40, text="100%", anchor=tk.E)
        
    def get_color_gradient(self, value):
        # Generate color from green (0.0) to red (1.0)
        r = int(min(255, value * 2 * 255))
        g = int(min(255, (1 - value) * 2 * 255))
        b = 0
        
        # Convert to hex
        return f"#{r:02x}{g:02x}{b:02x}"
            
    def save_model(self):
        if not self.model_trained:
            self.root.after(0, lambda: messagebox.showinfo("Info", "Please train a model first"))
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
            
        if self.ai_system.save_model(filepath):
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Model saved to {filepath}"))
        else:
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to save model"))
            
    def load_model(self):
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
            
        if self.ai_system.load_model(filepath):
            self.model_trained = True
            self.root.after(0, lambda: self.status_var.set(f"Model loaded: {os.path.basename(filepath)}"))
            
            # Update results text
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"\n--- Model Loaded ---\n"))
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"Model: {self.ai_system.best_model_name}\n"))
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"Accuracy: {self.ai_system.model_accuracies[self.ai_system.best_model_name]['test_accuracy']:.4f}\n"))
            self.root.after(0, lambda: self.results_text.see(tk.END))
            
            # Update visualization
            self.root.after(0, self.update_visualization)
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "Model loaded successfully"))
        else:
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load model"))
            
    def show_about(self):
        about_text = """
        AI-Powered Healthcare Disease Prediction System
        
        This application demonstrates how AI can transform healthcare
        by predicting diseases based on patient data.
        
        Features:
        - Multiple machine learning models
        - Model evaluation and optimization
        - Interactive visualizations
        - User-friendly interface for predictions
        
        Created as an example of healthcare AI applications.
        """
        
        self.root.after(0, lambda: messagebox.showinfo("About", about_text))
        
    def show_instructions(self):
        instructions = """
        How to use this application:
        
        1. Load Dataset: Start by loading a healthcare dataset in CSV format
        
        2. Data Exploration: Explore the dataset in the Data Explorer tab
        
        3. Model Training:
           - Select target variable (disease indicator)
           - Select features to use for prediction
           - Preprocess the data
           - Train multiple models
           - Optimize the best model
           - Evaluate model performance
        
        4. Visualizations: View ROC curve, Confusion Matrix, Feature Importance, Correlation Matrix, and Patient Distribution
        
        5. Disease Prediction:
           - Enter patient data
           - Click "Make Prediction" to see the result
        
        You can save and load trained models for future use.
        """
        
        self.root.after(0, lambda: messagebox.showinfo("Instructions", instructions))

def main():
    root = tk.Tk()
    app = HealthcareAI_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()