"""
Complete NSL-KDD Dataset Generation and SVM Modeling Pipeline
This script generates the NSL-KDD dataset and trains SVM models on it
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import time
import warnings
warnings.filterwarnings('ignore')

class NSLKDDDatasetGenerator:
    """
    Generate realistic NSL-KDD dataset for intrusion detection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Attack type mapping
        self.attack_mapping = {
            # DoS attacks
            'apache2': 'DoS', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
            'mailbomb': 'DoS', 'pod': 'DoS', 'processtable': 'DoS', 'smurf': 'DoS',
            'teardrop': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
            
            # R2L attacks
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'httptunnel': 'R2L',
            'imap': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L',
            'sendmail': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
            'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
            
            # Probe attacks
            'ipsweep': 'Probe', 'mscan': 'Probe', 'nmap': 'Probe',
            'portsweep': 'Probe', 'saint': 'Probe', 'satan': 'Probe',
            
            # U2R attacks
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
            'ps': 'U2R', 'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
            
            # Normal
            'normal': 'Normal'
        }
    
    def generate_nslkdd_dataset(self, n_samples=50000):
        """
        Generate synthetic NSL-KDD dataset
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        print(f"Generating NSL-KDD dataset with {n_samples} samples...")
        
        # Define the 41 features of NSL-KDD
        data = {}
        
        # Basic features (9 features)
        data['duration'] = np.random.exponential(scale=30, size=n_samples)
        data['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'], 
                                               size=n_samples, p=[0.8, 0.15, 0.05])
        data['service'] = np.random.choice(
            ['http', 'smtp', 'ftp', 'telnet', 'ssh', 'pop3', 'imap4', 'other'],
            size=n_samples, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1]
        )
        data['flag'] = np.random.choice(
            ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH'],
            size=n_samples, p=[0.6, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005]
        )
        data['src_bytes'] = np.random.exponential(scale=1000, size=n_samples)
        data['dst_bytes'] = np.random.exponential(scale=1000, size=n_samples)
        data['land'] = np.random.binomial(1, 0.001, size=n_samples)
        data['wrong_fragment'] = np.random.poisson(lam=0.1, size=n_samples)
        data['urgent'] = np.random.poisson(lam=0.05, size=n_samples)
        
        # Content features (13 features)
        data['hot'] = np.random.exponential(scale=2, size=n_samples)
        data['num_failed_logins'] = np.random.poisson(lam=0.1, size=n_samples)
        data['logged_in'] = np.random.binomial(1, 0.4, size=n_samples)
        data['num_compromised'] = np.random.exponential(scale=1, size=n_samples)
        data['root_shell'] = np.random.binomial(1, 0.1, size=n_samples)
        data['su_attempted'] = np.random.poisson(lam=0.05, size=n_samples)
        data['num_root'] = np.random.exponential(scale=1, size=n_samples)
        data['num_file_creations'] = np.random.exponential(scale=1, size=n_samples)
        data['num_shells'] = np.random.poisson(lam=0.1, size=n_samples)
        data['num_access_files'] = np.random.poisson(lam=0.1, size=n_samples)
        data['num_outbound_cmds'] = np.zeros(n_samples)  # Always 0 in NSL-KDD
        data['is_host_login'] = np.random.binomial(1, 0.001, size=n_samples)
        data['is_guest_login'] = np.random.binomial(1, 0.001, size=n_samples)
        
        # Traffic features (9 features)
        data['count'] = np.random.exponential(scale=50, size=n_samples)
        data['srv_count'] = np.random.exponential(scale=50, size=n_samples)
        data['serror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['srv_serror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['rerror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['srv_rerror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['same_srv_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['diff_srv_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['srv_diff_host_rate'] = np.random.uniform(0, 1, size=n_samples)
        
        # Host-based features (10 features)
        data['dst_host_count'] = np.random.randint(0, 256, size=n_samples)
        data['dst_host_srv_count'] = np.random.randint(0, 256, size=n_samples)
        data['dst_host_same_srv_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['dst_host_diff_srv_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['dst_host_same_src_port_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['dst_host_srv_diff_host_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['dst_host_serror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['dst_host_srv_serror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['dst_host_rerror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['dst_host_srv_rerror_rate'] = np.random.uniform(0, 1, size=n_samples)
        
        # Generate attack labels
        attack_types = list(self.attack_mapping.keys())
        # 60% normal, 40% attacks distributed among different types
        attack_probs = [0.6] + [0.4/len(attack_types[:-1])] * (len(attack_types) - 1)
        data['attack_type'] = np.random.choice(attack_types, size=n_samples, p=attack_probs)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add derived features
        df['attack_category'] = df['attack_type'].map(self.attack_mapping)
        df['is_attack'] = (df['attack_category'] != 'Normal').astype(int)
        
        print(f"Dataset generated successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Attack distribution:")
        print(df['attack_category'].value_counts())
        
        return df
    
    def preprocess_dataset(self, df):
        """
        Preprocess the generated dataset for machine learning
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("Preprocessing dataset...")
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # One-hot encode categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        for feature in categorical_features:
            if feature in processed_df.columns:
                dummies = pd.get_dummies(processed_df[feature], prefix=feature)
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df.drop(feature, axis=1, inplace=True)
        
        # Clip extreme values to handle outliers
        numerical_features = processed_df.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features 
                            if col not in ['is_attack', 'attack_type']]
        
        for feature in numerical_features:
            # Clip to 99th percentile to handle outliers
            upper_bound = np.percentile(processed_df[feature], 99)
            processed_df[feature] = np.clip(processed_df[feature], 0, upper_bound)
        
        print(f"Preprocessing completed!")
        print(f"Final dataset shape: {processed_df.shape}")
        
        return processed_df


class SVMIntrusionDetector:
    """
    SVM-based intrusion detection system
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.svm_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
    def prepare_data(self, processed_df, test_size=0.3):
        """
        Prepare data for training and testing
        
        Args:
            processed_df: Preprocessed dataset
            test_size: Proportion of data for testing
            
        Returns:
            tuple: Training and testing datasets
        """
        print("Preparing data for SVM model...")
        
        # Extract features and labels
        feature_columns = [col for col in processed_df.columns 
                          if col not in ['attack_category', 'is_attack', 'attack_type']]
        
        self.feature_names = feature_columns
        X = processed_df[feature_columns].values
        y = processed_df['is_attack'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        print(f"Class distribution in training set:")
        print(f"  Normal: {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
        print(f"  Attack: {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_svm(self, X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            kernel: SVM kernel type
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        print(f"Training SVM with {kernel} kernel...")
        start_time = time.time()
        
        # Initialize and train SVM
        self.svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=self.random_state,
            probability=True
        )
        
        self.svm_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Number of support vectors: {self.svm_model.n_support_}")
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            tuple: (metrics_dict, predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.svm_model.predict(X_test)
        y_pred_proba = self.svm_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'specificity': recall_score(y_test, y_pred, pos_label=0),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def print_evaluation_results(self, y_test, y_pred, metrics):
        """
        Print comprehensive evaluation results
        """
        target_names = ['Normal', 'Attack']
        
        print("\n" + "="*60)
        print("SVM INTRUSION DETECTION EVALUATION RESULTS")
        print("="*60)
        
        print("\nDetailed Metrics:")
        print(f"Accuracy Score        = {metrics['accuracy']:.4f}")
        print(f"Precision Score       = {metrics['precision']:.4f}")
        print(f"Recall/Sensitivity    = {metrics['recall']:.4f}")
        print(f"Specificity           = {metrics['specificity']:.4f}")
        print(f"F1 Score              = {metrics['f1_score']:.4f}")
        print(f"ROC AUC Score         = {metrics['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                   Predicted")
        print(f"Actual    Normal   Attack")
        print(f"Normal    {cm[0,0]:6d}   {cm[0,1]:6d}")
        print(f"Attack    {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    def plot_results(self, y_test, y_pred, y_pred_proba):
        """
        Create visualizations of the results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'SVM (AUC = {roc_auc:.3f})')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, 'r-', linewidth=2, label=f'SVM (AUC = {pr_auc:.3f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Class Distribution Comparison
        labels = ['Normal', 'Attack']
        true_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]
        pred_counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax4.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax4.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        ax4.set_ylabel('Count')
        ax4.set_title('Class Distribution')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()


# Complete pipeline execution
def run_complete_pipeline():
    """
    Run the complete NSL-KDD dataset generation and SVM modeling pipeline
    """
    print("Starting Complete NSL-KDD SVM Pipeline")
    print("="*50)
    
    # Step 1: Generate dataset
    generator = NSLKDDDatasetGenerator(random_state=42)
    raw_dataset = generator.generate_nslkdd_dataset(n_samples=50000)
    
    # Step 2: Preprocess dataset
    processed_dataset = generator.preprocess_dataset(raw_dataset)
    
    # Step 3: Initialize SVM detector
    svm_detector = SVMIntrusionDetector(random_state=42)
    
    # Step 4: Prepare data
    X_train, X_test, y_train, y_test = svm_detector.prepare_data(processed_dataset)
    
    # Step 5: Train SVM model
    svm_detector.train_svm(X_train, y_train, kernel='rbf', C=1.0)
    
    # Step 6: Evaluate model
    metrics, y_pred, y_pred_proba = svm_detector.evaluate_model(X_test, y_test)
    
    # Step 7: Print results
    svm_detector.print_evaluation_results(y_test, y_pred, metrics)
    
    # Step 8: Create visualizations
    svm_detector.plot_results(y_test, y_pred, y_pred_proba)
    
    # Step 9: Save the dataset for future use
    processed_dataset.to_csv('nslkdd_processed_dataset.csv', index=False)
    print(f"\nDataset saved as 'nslkdd_processed_dataset.csv'")
    
    return processed_dataset, svm_detector, metrics


# Execute the pipeline
if __name__ == "__main__":
    # Run the complete pipeline
    dataset, detector, results = run_complete_pipeline()
    
    print("\nPipeline completed successfully!")
    print(f"Final model accuracy: {results['accuracy']:.4f}")
    print(f"Final model F1-score: {results['f1_score']:.4f}")