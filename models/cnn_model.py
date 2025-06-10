import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow to use CPU if GPU is unavailable
try:
    # Try to configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" Using GPU: {len(gpus)} device(s) available")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration failed: {e}")
            print("Falling back to CPU...")
            tf.config.set_visible_devices([], 'GPU')
    else:
        print(" No GPU detected, using CPU")
        tf.config.set_visible_devices([], 'GPU')
except Exception as e:
    print(f"GPU setup error: {e}")
    print("Forcing CPU usage...")
    tf.config.set_visible_devices([], 'GPU')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class IntrusionDetectionDataset:
    """
    Creates and manages NSL-KDD style dataset for intrusion detection
    """
    
    def __init__(self):
        # Define attack categories and their specific types
        self.attack_categories = {
            'Normal': ['normal'],
            'DoS': ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
                    'processtable', 'smurf', 'teardrop', 'udpstorm'],
            'R2L': ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 
                    'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'],
            'Probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
            'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'sqlattack']
        }
        
        # Flatten attack mapping for easy lookup
        self.attack_to_category = {}
        for category, attacks in self.attack_categories.items():
            for attack in attacks:
                self.attack_to_category[attack] = category
    
    def create_network_features(self, num_samples):
        """Generate realistic network traffic features"""
        
        # Basic connection features
        duration = np.random.exponential(scale=25, size=num_samples)
        
        # Protocol distribution (TCP dominant in real networks)
        protocols = np.random.choice(['tcp', 'udp', 'icmp'], 
                                   size=num_samples, p=[0.85, 0.12, 0.03])
        
        # Service types (HTTP/HTTPS most common) - normalized probabilities
        service_probs = [0.45, 0.25, 0.08, 0.08, 0.06, 0.03, 0.02, 0.02, 0.01]
        service_probs = np.array(service_probs) / np.sum(service_probs)  # Normalize
        services = np.random.choice([
            'http', 'https', 'ftp', 'smtp', 'ssh', 'telnet', 'pop3', 'domain_u', 'other'
        ], size=num_samples, p=service_probs)
        
        # Connection flags (probabilities sum to 1.0)
        flag_probs = [0.65, 0.15, 0.08, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.003, 0.002]
        flag_probs = np.array(flag_probs) / np.sum(flag_probs)  # Normalize to ensure sum = 1
        flags = np.random.choice([
            'SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH'
        ], size=num_samples, p=flag_probs)
        
        # Data transfer features
        src_bytes = np.random.lognormal(mean=8, sigma=2, size=num_samples)
        dst_bytes = np.random.lognormal(mean=7, sigma=2.5, size=num_samples)
        
        # Binary flags
        land_flag = np.random.binomial(1, 0.0005, size=num_samples)
        wrong_fragment = np.random.poisson(lam=0.05, size=num_samples)
        urgent = np.random.poisson(lam=0.02, size=num_samples)
        
        return {
            'duration': duration,
            'protocol_type': protocols,
            'service': services,
            'flag': flags,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'land': land_flag,
            'wrong_fragment': wrong_fragment,
            'urgent': urgent
        }
    
    def create_content_features(self, num_samples):
        """Generate content-based features"""
        
        return {
            'hot': np.random.exponential(scale=1.5, size=num_samples),
            'num_failed_logins': np.random.poisson(lam=0.08, size=num_samples),
            'logged_in': np.random.binomial(1, 0.45, size=num_samples),
            'num_compromised': np.random.exponential(scale=0.5, size=num_samples),
            'root_shell': np.random.binomial(1, 0.08, size=num_samples),
            'su_attempted': np.random.poisson(lam=0.03, size=num_samples),
            'num_root': np.random.exponential(scale=0.8, size=num_samples),
            'num_file_creations': np.random.exponential(scale=0.6, size=num_samples),
            'num_shells': np.random.poisson(lam=0.05, size=num_samples),
            'num_access_files': np.random.poisson(lam=0.04, size=num_samples),
            'num_outbound_cmds': np.zeros(num_samples),  # Always 0 in NSL-KDD
            'is_host_login': np.random.binomial(1, 0.0008, size=num_samples),
            'is_guest_login': np.random.binomial(1, 0.0006, size=num_samples)
        }
    
    def create_traffic_features(self, num_samples):
        """Generate traffic-based features"""
        
        return {
            'count': np.random.exponential(scale=45, size=num_samples),
            'srv_count': np.random.exponential(scale=40, size=num_samples),
            'serror_rate': np.random.beta(1, 3, size=num_samples),
            'srv_serror_rate': np.random.beta(1, 3, size=num_samples),
            'rerror_rate': np.random.beta(1, 4, size=num_samples),
            'srv_rerror_rate': np.random.beta(1, 4, size=num_samples),
            'same_srv_rate': np.random.beta(3, 1, size=num_samples),
            'diff_srv_rate': np.random.beta(1, 2, size=num_samples),
            'srv_diff_host_rate': np.random.beta(1, 3, size=num_samples)
        }
    
    def create_host_features(self, num_samples):
        """Generate host-based features"""
        
        return {
            'dst_host_count': np.random.randint(1, 256, size=num_samples),
            'dst_host_srv_count': np.random.randint(1, 256, size=num_samples),
            'dst_host_same_srv_rate': np.random.beta(3, 1, size=num_samples),
            'dst_host_diff_srv_rate': np.random.beta(1, 2, size=num_samples),
            'dst_host_same_src_port_rate': np.random.beta(2, 1, size=num_samples),
            'dst_host_srv_diff_host_rate': np.random.beta(1, 3, size=num_samples),
            'dst_host_serror_rate': np.random.beta(1, 4, size=num_samples),
            'dst_host_srv_serror_rate': np.random.beta(1, 4, size=num_samples),
            'dst_host_rerror_rate': np.random.beta(1, 5, size=num_samples),
            'dst_host_srv_rerror_rate': np.random.beta(1, 5, size=num_samples)
        }
    
    def generate_dataset(self, total_samples=50000, normal_ratio=0.65):
        """
        Generate complete NSL-KDD style dataset
        """
        print(f"Generate {total_samples} network traffic samples...")
        
        # Calculate samples per category
        normal_samples = int(total_samples * normal_ratio)
        attack_samples = total_samples - normal_samples
        
        # Distribute attack samples among categories
        attack_distribution = {
            'DoS': 0.45,
            'Probe': 0.30,
            'R2L': 0.15,
            'U2R': 0.10
        }
        
        # Generate samples for each category
        all_data = []
        
        # Generate normal traffic
        normal_data = self._generate_category_data('Normal', normal_samples)
        all_data.extend(normal_data)
        
        # Generate attack traffic
        for attack_cat, ratio in attack_distribution.items():
            cat_samples = int(attack_samples * ratio)
            if cat_samples > 0:
                attack_data = self._generate_category_data(attack_cat, cat_samples)
                all_data.extend(attack_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dataset generated successfully!")
        print(f"Shape: {df.shape}")
        print(f"Attack distribution:")
        print(df['category'].value_counts().sort_index())
        
        return df
    
    def _generate_category_data(self, category, num_samples):
        """Generate data for a specific attack category"""
        
        # Generate base features
        network_features = self.create_network_features(num_samples)
        content_features = self.create_content_features(num_samples)
        traffic_features = self.create_traffic_features(num_samples)
        host_features = self.create_host_features(num_samples)
        
        # Combine all features
        combined_features = {**network_features, **content_features, 
                           **traffic_features, **host_features}
        
        # Add category and binary labels
        if category == 'Normal':
            attack_types = ['normal'] * num_samples
        else:
            attack_types = np.random.choice(self.attack_categories[category], 
                                          size=num_samples)
        
        # Create records
        records = []
        for i in range(num_samples):
            record = {}
            for feature, values in combined_features.items():
                record[feature] = values[i]
            
            record['attack_type'] = attack_types[i]
            record['category'] = category
            record['is_attack'] = 1 if category != 'Normal' else 0
            
            records.append(record)
        
        return records
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset for machine learning
        """
      
        # Create a copy
        processed_df = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            # One-hot encoding
            dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=False)
            processed_df = pd.concat([processed_df, dummies], axis=1)
            processed_df.drop(col, axis=1, inplace=True)
        
        # Select features (exclude target variables)
        feature_cols = [col for col in processed_df.columns 
                       if col not in ['attack_type', 'category', 'is_attack']]
        
        X = processed_df[feature_cols].values
        y = processed_df['is_attack'].values
        
        # Handle any infinite or NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
       
        return X, y, feature_cols

class CNNIntrusionDetector:
    """
    Convolutional Neural Network for intrusion detection
    """
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
    
    def build_model(self):
        """
        Build the CNN architecture for intrusion detection
        """
        print("Building CNN model for intrusion detection...")
        
        # Force CPU usage for model creation if needed
        with tf.device('/CPU:0'):
            model = Sequential([
                # First convolutional layer
                Conv1D(filters=256, kernel_size=1, activation='tanh', 
                       input_shape=(1, self.input_shape)),
                
                # Second convolutional layer
                Conv1D(filters=128, kernel_size=1, activation='tanh'),
                
                # Third convolutional layer
                Conv1D(filters=64, kernel_size=1, activation='tanh'),
                
                # Fourth convolutional layer
                Conv1D(filters=32, kernel_size=1, activation='tanh'),
                
                # Pooling layer
                MaxPooling1D(pool_size=1),
                
                # Dropout for regularization
                Dropout(0.2),
                
                # Flatten for dense layers
                Flatten(),
                
                # Dense layers
                Dense(100, activation='relu'),
                Dropout(0.3),
                
                # Output layer (binary classification)
                Dense(2, activation='softmax')
            ])
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✅ Model built successfully!")
        print(f"📊 Model summary:")
        model.summary()
        
        return model
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Prepare data for training
        """
       
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for CNN (add channel dimension)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, 2)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_reshaped, y_categorical, test_size=test_size, 
            random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=42, stratify=y_temp.argmax(axis=1)
        )
        
        print(f" Data prepared!")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f" Validation set: {X_val.shape[0]} samples")
        print(f" Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """
        Train the CNN model
        """
        
        
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                'best_cnn_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train the model with CPU/GPU handling
        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"⚠️  Training error: {e}")
            print("🔄 Retrying with CPU-only mode...")
            with tf.device('/CPU:0'):
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
        
        self.history = history
        
        print("✅ Training completed!")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        print("📊 Evaluating model performance...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Convert to class labels
        y_true_labels = y_test.argmax(axis=1)
        y_pred_labels = y_pred.argmax(axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        precision = precision_score(y_true_labels, y_pred_labels)
        recall = recall_score(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels)
        specificity = recall_score(y_true_labels, y_pred_labels, pos_label=0)
        
        print("🎯 Model Performance Metrics:")
        print(f"   Accuracy:     {accuracy:.4f}")
        print(f"   Precision:    {precision:.4f}")
        print(f"   Recall:       {recall:.4f}")
        print(f"   Specificity:  {specificity:.4f}")
        print(f"   F1-Score:     {f1:.4f}")
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels, 
                                  target_names=['Normal', 'Attack']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        print("\n🔍 Confusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_true': y_true_labels,
            'y_pred': y_pred_labels,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("❌ No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, results):
        """
        Plot ROC curve
        """
        y_true = results['y_true']
        y_scores = results['y_pred_proba'][:, 1]  # Probability of attack class
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        return roc_auc
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix heatmap
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def main():
    """
    Main function to run the complete intrusion detection pipeline
    """
    print("🚀 Starting NSL-KDD Intrusion Detection with CNN")
    print("=" * 60)
    
    # Step 1: Generate dataset
    dataset_generator = IntrusionDetectionDataset()
    dataset = dataset_generator.generate_dataset(total_samples=50000)
    
    # Step 2: Preprocess data
    X, y, feature_names = dataset_generator.preprocess_data(dataset)
    
    # Step 3: Initialize CNN model
    cnn_model = CNNIntrusionDetector(input_shape=X.shape[1])
    
    # Step 4: Build model architecture
    cnn_model.build_model()
    
    # Step 5: Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test = cnn_model.prepare_data(X, y)
    
    # Step 6: Train the model
    history = cnn_model.train_model(X_train, y_train, X_val, y_val, 
                                   epochs=50, batch_size=128)
    
    # Step 7: Evaluate the model
    results = cnn_model.evaluate_model(X_test, y_test)
    
    # Step 8: Visualize results
    print("\n📈 Generating visualizations...")
    
    # Plot training history
    cnn_model.plot_training_history()
    
    # Plot ROC curve
    roc_auc = cnn_model.plot_roc_curve(results)
    print(f"🎯 ROC AUC Score: {roc_auc:.4f}")
    
    # Plot confusion matrix
    cnn_model.plot_confusion_matrix(results['confusion_matrix'])
    
    print("=" * 60)
    print("Intrusion detection analysis is completed successfully!")
    
    return cnn_model, results, dataset

# Run the complete pipeline
if __name__ == "__main__":
    model, evaluation_results, generated_dataset = main()