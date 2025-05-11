import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from flask import Flask, request, render_template_string, flash, make_response, send_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'secure-secret-key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Ensure static/images directory exists
os.makedirs(os.path.join(app.root_path, 'static', 'images'), exist_ok=True)

# Categorical feature options (for validation)
CATEGORICAL_OPTIONS = {
    'Sex': ['0', '1'],
    'CP': ['0', '1', '2', '3'],
    'Fbs': ['0', '1'],
    'Restecg': ['0', '1', '2'],
    'Exang': ['0', '1'],
    'Slope': ['0', '1', '2'],
    'CA': ['0', '1', '2', '3', '4'],
    'Thal': ['0', '1', '2', '3']
}

# Base HTML template
BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Roboto', sans-serif;
            background: #121212;
            color: #E0E0E0;
            min-height: 100vh;
            position: relative;
        }
        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #1f1b4e, #3f51b5);
            z-index: -1;
        }
        .sidebar {
            width: 200px;
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.2);
        }
        .sidebar h3 {
            color: #BB86FC;
            margin-bottom: 20px;
        }
        .sidebar a {
            display: block;
            color: #E0E0E0;
            text-decoration: none;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 8px;
            transition: background 0.3s;
        }
        .sidebar a:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .main-content {
            margin-left: 220px;
            padding: 30px;
            max-width: 1000px;
        }
        .glass-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .flash-message {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
            color: #E0E0E0;
        }
        .flash-message.error {
            border-color: #ef5350;
            color: #ef5350;
        }
        .flash-message.success {
            border-color: #4caf50;
            color: #4caf50;
        }
    </style>
</head>
<body>
    <div id="particles-js"></div>
    <div class="sidebar">
        <h3>Healthcare AI</h3>
        <a href="/">Home</a>
        <a href="/visualise">Visualise Results</a>
    </div>
    <div class="main-content">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="flash-message {{ category }}">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <div class="glass-container">
            {{ content | safe }}
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <script>
        const particlesConfig = {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: ["#ffffff", "#b3e5fc", "#e1bee7"] },
                shape: { type: "circle" },
                opacity: { value: 0.5, random: true },
                size: { value: 4, random: true },
                line_linked: { enable: false },
                move: { enable: true, speed: 2, direction: "none", random: true, out_mode: "out" }
            },
            interactivity: {
                detect_on: "canvas",
                events: { onhover: { enable: true, mode: "repulse" }, onclick: { enable: true, mode: "push" }, resize: true },
                modes: { repulse: { distance: 100, duration: 0.4 }, push: { particles_nb: 4 } }
            },
            retina_detect: true
        };
        particlesJS("particles-js", particlesConfig);
    </script>
</body>
</html>
"""

# Predict page template (fully hardcoded)
PREDICT_HTML = """
<h1 style="margin-bottom: 20px;">Predict Heart Disease</h1>
<form method="POST" action="/" style="margin-bottom: 20px;">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
        <!-- Age -->
        <div style="display: flex; flex-direction: column;">
            <label for="Age" style="margin-bottom: 5px; font-weight: 500;">Age</label>
            <input type="number" id="Age" name="Age" step="any" value="63.0"
                   required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Sex -->
        <div style="display: flex; flex-direction: column;">
            <label for="Sex" style="margin-bottom: 5px; font-weight: 500;">Sex</label>
            <select id="Sex" name="Sex" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">Female</option>
                <option value="1" selected>Male</option>
            </select>
        </div>
        <!-- CP (Chest Pain Type) -->
        <div style="display: flex; flex-direction: column;">
            <label for="CP" style="margin-bottom: 5px; font-weight: 500;">Chest Pain Type</label>
            <select id="CP" name="CP" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">Asymptomatic</option>
                <option value="1">Typical Angina</option>
                <option value="2">Atypical Angina</option>
                <option value="3" selected>Non-Anginal Pain</option>
            </select>
        </div>
        <!-- Trestbps -->
        <div style="display: flex; flex-direction: column;">
            <label for="Trestbps" style="margin-bottom: 5px; font-weight: 500;">Resting BP</label>
            <input type="number" id="Trestbps" name="Trestbps" step="any" value="145.0"
                   required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Chol -->
        <div style="display: flex; flex-direction: column;">
            <label for="Chol" style="margin-bottom: 5px; font-weight: 500;">Cholesterol</label>
            <input type="number" id="Chol" name="Chol" step="any" value="233.0"
                   required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Fbs -->
        <div style="display: flex; flex-direction: column;">
            <label for="Fbs" style="margin-bottom: 5px; font-weight: 500;">Fasting Blood Sugar</label>
            <select id="Fbs" name="Fbs" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">≤ 120 mg/dl</option>
                <option value="1" selected>> 120 mg/dl</option>
            </select>
        </div>
        <!-- Restecg -->
        <div style="display: flex; flex-direction: column;">
            <label for="Restecg" style="margin-bottom: 5px; font-weight: 500;">Resting ECG</label>
            <select id="Restecg" name="Restecg" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>Normal</option>
                <option value="1">ST-T Wave Abnormality</option>
                <option value="2">Left Ventricular Hypertrophy</option>
            </select>
        </div>
        <!-- Thalach -->
        <div style="display: flex; flex-direction: column;">
            <label for="Thalach" style="margin-bottom: 5px; font-weight: 500;">Max Heart Rate</label>
            <input type="number" id="Thalach" name="Thalach" step="any" value="150.0"
                   required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Exang -->
        <div style="display: flex; flex-direction: column;">
            <label for="Exang" style="margin-bottom: 5px; font-weight: 500;">Exercise Angina</label>
            <select id="Exang" name="Exang" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>No</option>
                <option value="1">Yes</option>
            </select>
        </div>
        <!-- Oldpeak -->
        <div style="display: flex; flex-direction: column;">
            <label for="Oldpeak" style="margin-bottom: 5px; font-weight: 500;">ST Depression</label>
            <input type="number" id="Oldpeak" name="Oldpeak" step="any" value="2.3"
                   required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
        </div>
        <!-- Slope -->
        <div style="display: flex; flex-direction: column;">
            <label for="Slope" style="margin-bottom: 5px; font-weight: 500;">Slope</label>
            <select id="Slope" name="Slope" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>Unsloping</option>
                <option value="1">Flat</option>
                <option value="2">Downsloping</option>
            </select>
        </div>
        <!-- CA -->
        <div style="display: flex; flex-direction: column;">
            <label for="CA" style="margin-bottom: 5px; font-weight: 500;">Major Vessels</label>
            <select id="CA" name="CA" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0" selected>0 Vessels</option>
                <option value="1">1 Vessel</option>
                <option value="2">2 Vessels</option>
                <option value="3">3 Vessels</option>
                <option value="4">4 Vessels</option>
            </select>
        </div>
        <!-- Thal -->
        <div style="display: flex; flex-direction: column;">
            <label for="Thal" style="margin-bottom: 5px; font-weight: 500;">Thalassemia</label>
            <select id="Thal" name="Thal" required style="padding: 8px; background: rgba(255, 255, 255, 0.1); color: #E0E0E0; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 8px;">
                <option value="0">Unknown</option>
                <option value="1" selected>Normal</option>
                <option value="2">Fixed Defect</option>
                <option value="3">Reversible Defect</option>
            </select>
        </div>
    </div>
    <button type="submit" style="margin-top: 20px; padding: 10px 20px; background: #BB86FC; color: #121212; border: none; border-radius: 8px; cursor: pointer;">
        Predict
    </button>
</form>
<div id="prediction-result" style="margin-top: 20px;">
    <h3 style="margin-bottom: 20px;">Prediction Result</h3>
    <p>Submit the form to see the prediction result.</p>
</div>
"""

# Visualise page template
VISUALISE_HTML = """
<h1 style="margin-bottom: 20px;">Model Performance Visualizations</h1>
<div style="display: flex; flex-direction: column; gap: 20px;">
    {% if roc_curve %}
    <div>
        <h3>ROC Curve</h3>
        <img src="{{ roc_curve }}" alt="ROC Curve" style="max-width: 100%; border-radius: 8px;">
    </div>
    {% endif %}
    {% if confusion_matrix %}
    <div>
        <h3>Confusion Matrix</h3>
        <img src="{{ confusion_matrix }}" alt="Confusion Matrix" style="max-width: 100%; border-radius: 8px;">
    </div>
    {% endif %}
    {% if feature_importance %}
    <div>
        <h3>Feature Importance</h3>
        <img src="{{ feature_importance }}" alt="Feature Importance" style="max-width: 100%; border-radius: 8px;">
    </div>
    {% endif %}
    {% if correlation_matrix %}
    <div>
        <h3>Correlation Matrix</h3>
        <img src="{{ correlation_matrix }}" alt="Correlation Matrix" style="max-width: 100%; border-radius: 8px;">
    </div>
    {% endif %}
    {% if patient_distribution %}
    <div>
        <h3>Patient Distribution</h3>
        <img src="{{ patient_distribution }}" alt="Patient Distribution" style="max-width: 100%; border-radius: 8px;">
    </div>
    {% endif %}
</div>
"""

# Prediction result template (for POST response)
PREDICTION_RESULT_HTML = """
<h3 style="margin-bottom: 20px;">Prediction Result</h3>
<div style="background: {{ '#ef5350' if prediction_class == 'danger' else '#4caf50' }}; padding: 15px; border-radius: 8px;">
    <strong>{{ prediction_text }}</strong>
    <p>Probability: {{ probability }}%</p>
</div>
"""

def create_visualization(viz_type, data, figsize=(10, 8), save_path=None):
    """
    Comprehensive visualization function that creates various healthcare analytics visualizations

    Parameters:
    -----------
    viz_type : str
        Type of visualization to create. Options are:
        "roc_curve", "confusion_matrix", "feature_importance",
        "correlation_matrix", "patient_distribution"

    data : dict
        Dictionary containing the data required for the visualization.
        Required keys depend on the visualization type:
        - roc_curve: X_test_scaled, y_test, model
        - confusion_matrix: X_test_scaled, y_test, model
        - feature_importance: feature_importances as {feature_name: importance_value, ...}
        - correlation_matrix: features as DataFrame
        - patient_distribution: X_test_scaled, y_test, model

    figsize : tuple, optional
        Size of the figure (width, height) in inches. Default is (10, 8)

    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns:
    --------
    fig : Figure
        The matplotlib Figure containing the visualization
    """
    global fig
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Process based on visualization type
        if viz_type.lower() == "roc_curve":
            # Check required data
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None

            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']

            # Check if binary classification
            if len(model.classes_) != 2:
                return None

            # Get prediction probabilities and compute ROC curve
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")

        elif viz_type.lower() == "confusion_matrix":
            # Check required data
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None

            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']

            # Get predictions and compute confusion matrix
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title('Confusion Matrix')

        elif viz_type.lower() == "feature_importance":
            # Check if feature importances provided
            if 'feature_importances' not in data or not data['feature_importances']:
                return None

            feature_importances = data['feature_importances']

            # Sort features by importance
            sorted_features = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

            # Limit to top 20 features if there are many
            if len(sorted_features) > 20:
                top_features = dict(list(sorted_features.items())[:20])
                sorted_features = top_features
                ax.set_title('Top 20 Features by Importance')
            else:
                ax.set_title('Feature Importance')

            # Plot feature importances
            bars = ax.barh(list(sorted_features.keys()), list(sorted_features.values()))
            ax.set_xlabel('Importance')

            # Add values to the bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:.3f}', ha='left', va='center')

        elif viz_type.lower() == "correlation_matrix":
            # Check if features dataframe provided
            if 'features' not in data or not isinstance(data['features'], pd.DataFrame):
                return None

            features = data['features']

            # Calculate correlation matrix
            corr_matrix = features.corr()

            # Create heatmap with appropriate size
            if corr_matrix.shape[0] > 20:
                # If too many features, focus on highest correlations
                # Create a mask for correlations with absolute value > 0.3 (except diagonal)
                mask = np.abs(corr_matrix) < 0.3
                np.fill_diagonal(mask, False)  # Keep diagonal

                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                            vmin=-1, vmax=1, ax=ax, mask=mask)
                ax.set_title('Significant Feature Correlations (|r| >= 0.3)')
            else:
                # If manageable size, show all correlations with annotations
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                            vmin=-1, vmax=1, ax=ax, fmt='.2f')
                ax.set_title('Feature Correlation Matrix')

        elif viz_type.lower() == "patient_distribution":
            # Check required data
            if not all(k in data for k in ['X_test_scaled', 'y_test', 'model']):
                return None

            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']

            # Get actual and predicted distributions
            y_pred = model.predict(X_test_scaled)
            actual_counts = pd.Series(y_test).value_counts().sort_index()
            predicted_counts = pd.Series(y_pred).value_counts().sort_index()

            # Get unique classes and their labels
            classes = np.sort(np.unique(np.concatenate([y_test, y_pred])))
            if len(classes) == 2:
                labels = ['No Disease', 'Disease']
            else:
                labels = [f'Class {c}' for c in classes]

            # Create bar plot
            x = np.arange(len(labels))
            width = 0.35

            # Ensure actual_counts and predicted_counts have entries for all classes
            actual_values = [actual_counts.get(c, 0) for c in classes]
            predicted_values = [predicted_counts.get(c, 0) for c in classes]

            ax.bar(x - width / 2, actual_values, width, label='Actual', color='skyblue')
            ax.bar(x + width / 2, predicted_values, width, label='Predicted', color='salmon')

            ax.set_ylabel('Number of Patients')
            ax.set_title('Patient Distribution: Actual vs Predicted')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            # Add value labels on top of bars
            for i, v in enumerate(actual_values):
                ax.text(i - width / 2, v + 0.5, str(v), ha='center')
            for i, v in enumerate(predicted_values):
                ax.text(i + width / 2, v + 0.5, str(v), ha='center')

        else:
            plt.close(fig)
            return None

        # Apply tight layout
        fig.tight_layout()

        # Save figure if path provided
        if save_path:
            try:
                fig.savefig(save_path, bbox_inches='tight', dpi=300)
            except Exception:
                pass

        return fig

    except Exception:
        if 'fig' in locals():
            plt.close(fig)
        return None


class HealthcareAI:
    def __init__(self):
        self.dataset = None
        self.features = None
        self.target = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.model_accuracies = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.original_columns = []
        self.dummy_columns = []
        self.load_data()
        if self.dataset is not None:
            self.preprocess_data()
            self.train_models()

    def load_data(self):
        try:
            self.dataset = pd.read_csv("heart.csv")
            if self.dataset.empty:
                return False
            return True
        except Exception:
            return False

    def preprocess_data(self, test_size=0.2):
        if self.dataset is None:
            return False
        target_column = 'Target'
        if target_column not in self.dataset.columns:
            return False
        for col in self.dataset.columns:
            if self.dataset[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.dataset[col]):
                    self.dataset[col].fillna(self.dataset[col].median(), inplace=True)
                else:
                    self.dataset[col].fillna(self.dataset[col].mode()[0], inplace=True)
        for cat_col in CATEGORICAL_OPTIONS.keys():
            if cat_col in self.dataset.columns:
                valid_options = set(CATEGORICAL_OPTIONS[cat_col])
                actual_values = set(self.dataset[cat_col].astype(str))
                if not actual_values.issubset(valid_options):
                    self.dataset[cat_col] = self.dataset[cat_col].astype(str).apply(
                        lambda x: x if x in valid_options else valid_options.pop()
                    )
        self.target = self.dataset[target_column]
        self.features = self.dataset.drop(target_column, axis=1)
        self.original_columns = self.features.columns.tolist()
        categorical_cols = self.features.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            self.features = pd.get_dummies(self.features, columns=categorical_cols)
        self.dummy_columns = self.features.columns.tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return True

    def train_models(self):
        if not hasattr(self, 'X_train_scaled') or self.X_train_scaled is None:
            return False
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.model_accuracies = {}
        for name, model in models.items():
            try:
                model.fit(self.X_train_scaled, self.y_train)
                self.models[name] = model
                y_pred = model.predict(self.X_test_scaled)
                test_accuracy = accuracy_score(self.y_test, y_pred)
                self.model_accuracies[name] = {"test_accuracy": float(test_accuracy * 100)}
            except Exception:
                self.model_accuracies[name] = {"test_accuracy": 0.0}
        if not self.model_accuracies:
            return False
        self.best_model_name = max(self.model_accuracies, key=lambda x: self.model_accuracies[x]["test_accuracy"])
        self.best_model = self.models[self.best_model_name]
        return True

    def predict(self, patient_data, threshold=0.5):
        if self.best_model is None:
            return None
        try:
            input_df = pd.DataFrame([patient_data], columns=self.original_columns)
            categorical_cols = input_df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                input_df = pd.get_dummies(input_df, columns=categorical_cols)
            for col in self.dummy_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[self.dummy_columns]
            scaled_data = self.scaler.transform(input_df)
            probabilities = self.best_model.predict_proba(scaled_data)[0]
            probability = probabilities[1]
            prediction = 1 if probability >= threshold else 0
            return {
                "probability": probability,
                "prediction": prediction
            }
        except Exception:
            return None


# Initialize AI system
ai_system = HealthcareAI()


@app.route('/', methods=['GET', 'POST'])
def predict():
    expected_features = ['Age', 'Sex', 'CP', 'Trestbps', 'Chol', 'Fbs', 'Restecg', 'Thalach', 'Exang', 'Oldpeak',
                         'Slope', 'CA', 'Thal']
    content = PREDICT_HTML
    if ai_system.dataset is None:
        flash('Dataset not loaded. Ensure heart.csv is in the correct directory.', 'error')
    elif not hasattr(ai_system, 'features') or ai_system.features is None:
        flash('Model not trained. Check dataset and preprocessing.', 'error')
    elif not all(f in ai_system.original_columns for f in expected_features):
        flash('Dataset columns do not match expected features. Check heart.csv.', 'error')
    elif request.method == 'POST':
        patient_data = {}
        for field, value in request.form.items():
            try:
                if field in ai_system.original_columns:
                    if field in CATEGORICAL_OPTIONS:
                        if value not in CATEGORICAL_OPTIONS[field]:
                            flash(f"Invalid value for {field}. Select a valid option.", 'error')
                            break
                        patient_data[field] = int(value)
                    else:
                        patient_data[field] = float(value)
            except ValueError:
                flash(f"Invalid value for {field}. Please enter a valid number.", 'error')
                break
        else:  # No break occurred, process prediction
            result = ai_system.predict(patient_data)
            if result:
                prediction_text = 'Disease Detected' if result['prediction'] == 1 else 'No Disease Detected'
                probability = round(result['probability'] * 100, 2)
                prediction_class = 'danger' if result['prediction'] == 1 else 'success'
                result_html = render_template_string(PREDICTION_RESULT_HTML, prediction_text=prediction_text,
                                                     probability=probability, prediction_class=prediction_class)
                content = PREDICT_HTML.replace(
                    '<div id="prediction-result" style="margin-top: 20px;">\n    <h3 style="margin-bottom: 20px;">Prediction Result</h3>\n    <p>Submit the form to see the prediction result.</p>\n</div>',
                    f'<div id="prediction-result" style="margin-top: 20px;">\n{result_html}\n</div>'
                )
            else:
                flash('Prediction failed. Ensure all required features are provided correctly.', 'error')
    try:
        response = make_response(render_template_string(BASE_HTML, title="Predict Heart Disease", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        flash(f"Error rendering page: {str(e)}", 'error')
        response = make_response(render_template_string(BASE_HTML, title="Predict Heart Disease", content=PREDICT_HTML))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response


@app.route('/visualise')
def visualise():
    # Clear previous images
    images_dir = os.path.join(app.root_path, 'static', 'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    # Check if data is available
    if (ai_system.best_model is None or ai_system.X_test_scaled is None or
            ai_system.y_test is None or ai_system.features is None):
        flash('Cannot generate visualizations: Model or data not available.', 'error')
        content = "<p>No visualizations available. Please ensure the model is trained and data is loaded.</p>"
        response = make_response(render_template_string(BASE_HTML, title="Visualise Results", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    # Prepare common data
    common_data = {
        'X_test_scaled': ai_system.X_test_scaled,
        'y_test': ai_system.y_test,
        'model': ai_system.best_model
    }

    # Generate visualizations
    visualizations = {
        'roc_curve': None,
        'confusion_matrix': None,
        'feature_importance': None,
        'correlation_matrix': None,
        'patient_distribution': None
    }

    # ROC Curve
    fig = create_visualization('roc_curve', common_data, figsize=(8, 6),
                               save_path=os.path.join(images_dir, 'roc_curve.png'))
    if fig:
        visualizations['roc_curve'] = '/images/roc_curve.png'
        plt.close(fig)

    # Confusion Matrix
    fig = create_visualization('confusion_matrix', common_data, figsize=(6, 6),
                               save_path=os.path.join(images_dir, 'confusion_matrix.png'))
    if fig:
        visualizations['confusion_matrix'] = '/images/confusion_matrix.png'
        plt.close(fig)

    # Feature Importance
    feature_importances = {}
    if ai_system.best_model_name == 'Logistic Regression':
        coef = np.abs(ai_system.best_model.coef_[0])
        feature_importances = dict(zip(ai_system.dummy_columns, coef))
    elif hasattr(ai_system.best_model, 'feature_importances_'):
        feature_importances = dict(zip(ai_system.dummy_columns, ai_system.best_model.feature_importances_))

    if feature_importances:
        fig = create_visualization('feature_importance', {'feature_importances': feature_importances},
                                   figsize=(10, 8), save_path=os.path.join(images_dir, 'feature_importance.png'))
        if fig:
            visualizations['feature_importance'] = '/images/feature_importance.png'
            plt.close(fig)

    # Correlation Matrix
    fig = create_visualization('correlation_matrix', {'features': ai_system.dataset[ai_system.original_columns]},
                               figsize=(10, 8), save_path=os.path.join(images_dir, 'correlation_matrix.png'))
    if fig:
        visualizations['correlation_matrix'] = '/images/correlation_matrix.png'
        plt.close(fig)

    # Patient Distribution
    fig = create_visualization('patient_distribution', common_data, figsize=(8, 6),
                               save_path=os.path.join(images_dir, 'patient_distribution.png'))
    if fig:
        visualizations['patient_distribution'] = '/images/patient_distribution.png'
        plt.close(fig)

    try:
        content = render_template_string(VISUALISE_HTML, **visualizations)
        response = make_response(render_template_string(BASE_HTML, title="Visualise Results", content=content))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        flash(f"Error rendering visualisation page: {str(e)}", 'error')
        response = make_response(render_template_string(BASE_HTML, title="Visualise Results", content=VISUALISE_HTML))
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response


@app.route('/images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory(os.path.join(app.root_path, 'static', 'images'), filename)
    except Exception as e:
        flash(f"Error serving image: {str(e)}", 'error')
        return '', 404


@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    app.run(debug=True)