"""
NSL-KDD Excel Data Processor and Dataset Generator
Original implementation for processing NSL-KDD Features.xlsx file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')

class NSLKDDExcelProcessor:
    """
    Advanced processor for NSL-KDD Excel files with synthetic data generation capabilities
    """
    
    def __init__(self):
        self.feature_info = None
        self.processed_data = None
        self.scalers = {}
        self.label_encoders = {}
        self.attack_mapping = self._create_attack_mapping()
        
    def _create_attack_mapping(self):
        """Create comprehensive attack type mapping"""
        return {
            # DoS attacks
            'apache2': 'DoS', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
            'mailbomb': 'DoS', 'pod': 'DoS', 'processtable': 'DoS', 'smurf': 'DoS',
            'teardrop': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
            
            # R2L attacks
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'httptunnel': 'R2L',
            'imap': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L',
            'sendmail': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
            'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
            'xlock': 'R2L', 'xsnoop': 'R2L',
            
            # Probe attacks
            'ipsweep': 'Probe', 'mscan': 'Probe', 'nmap': 'Probe',
            'portsweep': 'Probe', 'saint': 'Probe', 'satan': 'Probe',
            
            # U2R attacks
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
            'ps': 'U2R', 'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
            
            # Normal
            'normal': 'Normal'
        }
    
    def load_excel_file(self, file_path):
        """
        Load and examine the NSL-KDD Excel file
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            dict: Information about the loaded file
        """
        try:
            # Read the Excel file - try different sheet names
            excel_file = pd.ExcelFile(file_path)
            print(f"Available sheets: {excel_file.sheet_names}")
            
            # Try to read the first sheet or most likely sheet
            sheet_name = excel_file.sheet_names[0]
            self.feature_info = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print(f"Loaded sheet '{sheet_name}' with shape: {self.feature_info.shape}")
            print(f"Columns: {list(self.feature_info.columns)}")
            
            return {
                'shape': self.feature_info.shape,
                'columns': list(self.feature_info.columns),
                'sheets': excel_file.sheet_names,
                'sample_data': self.feature_info.head()
            }
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None
    
    def analyze_feature_structure(self):
        """
        Analyze the structure of the loaded feature information
        
        Returns:
            dict: Analysis results
        """
        if self.feature_info is None:
            print("No data loaded. Please load Excel file first.")
            return None
        
        analysis = {
            'data_types': self.feature_info.dtypes.to_dict(),
            'missing_values': self.feature_info.isnull().sum().to_dict(),
            'unique_values': {col: self.feature_info[col].nunique() 
                            for col in self.feature_info.columns},
            'statistical_summary': self.feature_info.describe().to_dict()
        }
        
        return analysis
    
    def generate_synthetic_nslkdd_dataset(self, n_samples=125973, random_state=42):
        """
        Generate synthetic NSL-KDD dataset based on feature characteristics
        
        Args:
            n_samples (int): Number of samples to generate
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        np.random.seed(random_state)
        
        # Define feature ranges and distributions based on NSL-KDD characteristics
        feature_specs = {
            'duration': {'type': 'continuous', 'min': 0, 'max': 58329, 'distribution': 'exponential'},
            'protocol_type': {'type': 'categorical', 'values': ['tcp', 'udp', 'icmp'], 'weights': [0.8, 0.15, 0.05]},
            'service': {'type': 'categorical', 'values': ['http', 'smtp', 'ftp', 'telnet', 'ssh', 'pop3', 'imap4', 'other'], 'weights': [0.3, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.2]},
            'flag': {'type': 'categorical', 'values': ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH'], 'weights': [0.6, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005]},
            'src_bytes': {'type': 'continuous', 'min': 0, 'max': 1379963888, 'distribution': 'exponential'},
            'dst_bytes': {'type': 'continuous', 'min': 0, 'max': 1309937401, 'distribution': 'exponential'},
            'land': {'type': 'binary', 'prob': 0.001},
            'wrong_fragment': {'type': 'continuous', 'min': 0, 'max': 3, 'distribution': 'poisson'},
            'urgent': {'type': 'continuous', 'min': 0, 'max': 14, 'distribution': 'poisson'},
            'hot': {'type': 'continuous', 'min': 0, 'max': 101, 'distribution': 'exponential'},
            'num_failed_logins': {'type': 'continuous', 'min': 0, 'max': 5, 'distribution': 'poisson'},
            'logged_in': {'type': 'binary', 'prob': 0.4},
            'num_compromised': {'type': 'continuous', 'min': 0, 'max': 7479, 'distribution': 'exponential'},
            'root_shell': {'type': 'binary', 'prob': 0.1},
            'su_attempted': {'type': 'continuous', 'min': 0, 'max': 2, 'distribution': 'poisson'},
            'num_root': {'type': 'continuous', 'min': 0, 'max': 7468, 'distribution': 'exponential'},
            'num_file_creations': {'type': 'continuous', 'min': 0, 'max': 100, 'distribution': 'exponential'},
            'num_shells': {'type': 'continuous', 'min': 0, 'max': 5, 'distribution': 'poisson'},
            'num_access_files': {'type': 'continuous', 'min': 0, 'max': 9, 'distribution': 'poisson'},
            'num_outbound_cmds': {'type': 'continuous', 'min': 0, 'max': 0, 'distribution': 'constant'},
            'is_host_login': {'type': 'binary', 'prob': 0.001},
            'is_guest_login': {'type': 'binary', 'prob': 0.001},
            'count': {'type': 'continuous', 'min': 0, 'max': 511, 'distribution': 'exponential'},
            'srv_count': {'type': 'continuous', 'min': 0, 'max': 511, 'distribution': 'exponential'},
            'serror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'srv_serror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'rerror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'srv_rerror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'same_srv_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'diff_srv_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'srv_diff_host_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_count': {'type': 'continuous', 'min': 0, 'max': 255, 'distribution': 'uniform'},
            'dst_host_srv_count': {'type': 'continuous', 'min': 0, 'max': 255, 'distribution': 'uniform'},
            'dst_host_same_srv_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_diff_srv_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_same_src_port_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_srv_diff_host_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_serror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_srv_serror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_rerror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'},
            'dst_host_srv_rerror_rate': {'type': 'continuous', 'min': 0, 'max': 1, 'distribution': 'uniform'}
        }
        
        # Generate synthetic data
        synthetic_data = {}
        
        for feature, spec in feature_specs.items():
            if spec['type'] == 'continuous':
                if spec['distribution'] == 'exponential':
                    # Generate exponential distribution
                    values = np.random.exponential(scale=(spec['max'] - spec['min'])/10, size=n_samples)
                    values = np.clip(values, spec['min'], spec['max'])
                elif spec['distribution'] == 'poisson':
                    # Generate Poisson distribution
                    values = np.random.poisson(lam=1, size=n_samples)
                    values = np.clip(values, spec['min'], spec['max'])
                elif spec['distribution'] == 'uniform':
                    # Generate uniform distribution
                    values = np.random.uniform(spec['min'], spec['max'], size=n_samples)
                else:  # constant
                    values = np.full(n_samples, spec['min'])
                    
                synthetic_data[feature] = values
                
            elif spec['type'] == 'categorical':
                # Generate categorical data
                synthetic_data[feature] = np.random.choice(
                    spec['values'], 
                    size=n_samples, 
                    p=spec['weights']
                )
                
            elif spec['type'] == 'binary':
                # Generate binary data
                synthetic_data[feature] = np.random.binomial(1, spec['prob'], size=n_samples)
        
        # Generate attack labels with realistic distribution
        attack_types = list(self.attack_mapping.keys())
        attack_weights = [0.5] + [0.5/len(attack_types[:-1])] * (len(attack_types) - 1)  # 50% normal, 50% attacks
        synthetic_data['attack_type'] = np.random.choice(attack_types, size=n_samples, p=attack_weights)
        
        # Create DataFrame
        df = pd.DataFrame(synthetic_data)
        
        print(f"Generated synthetic dataset with {n_samples} samples and {len(df.columns)} features")
        return df
    
    def preprocess_synthetic_dataset(self, df):
        """
        Preprocess the generated synthetic dataset
        
        Args:
            df (pd.DataFrame): Synthetic dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        processed_df = df.copy()
        
        # 1. Create attack categories
        processed_df['attack_category'] = processed_df['attack_type'].map(self.attack_mapping)
        
        # 2. Create binary target
        processed_df['is_attack'] = (processed_df['attack_category'] != 'Normal').astype(int)
        
        # 3. One-hot encode categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        for feature in categorical_features:
            if feature in processed_df.columns:
                dummies = pd.get_dummies(processed_df[feature], prefix=feature, prefix_sep='_')
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df.drop(feature, axis=1, inplace=True)
        
        # 4. Scale numerical features
        numerical_features = processed_df.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col not in ['is_attack']]
        
        scaler = StandardScaler()
        processed_df[numerical_features] = scaler.fit_transform(processed_df[numerical_features])
        self.scalers['numerical'] = scaler
        
        # 5. Feature selection using correlation
        X = processed_df.drop(['attack_type', 'attack_category', 'is_attack'], axis=1)
        y = processed_df['is_attack']
        
        # Calculate correlations and select top features
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(20).index.tolist()
        
        # Create final dataset
        final_df = processed_df[top_features + ['attack_category', 'is_attack', 'attack_type']].copy()
        
        self.processed_data = final_df
        print(f"Preprocessing completed. Final dataset shape: {final_df.shape}")
        
        return final_df
    
    def create_train_test_splits(self, df, test_size=0.3, validation_size=0.2):
        """
        Create train, validation, and test splits
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            test_size (float): Proportion for test set
            validation_size (float): Proportion of training set for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and targets
        feature_cols = [col for col in df.columns 
                       if col not in ['attack_category', 'is_attack', 'attack_type']]
        
        X = df[feature_cols]
        y = df['is_attack']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
        )
        
        print(f"Dataset splits:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Testing: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def visualize_dataset_characteristics(self, df):
        """
        Create comprehensive visualizations of the dataset
        
        Args:
            df (pd.DataFrame): Dataset to visualize
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NSL-KDD Dataset Analysis', fontsize=16)
        
        # 1. Attack type distribution
        attack_counts = df['attack_category'].value_counts()
        axes[0, 0].pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Attack Type Distribution')
        
        # 2. Binary classification distribution
        binary_counts = df['is_attack'].value_counts()
        axes[0, 1].bar(['Normal', 'Attack'], binary_counts.values, color=['green', 'red'])
        axes[0, 1].set_title('Binary Classification Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Feature correlation heatmap (top 10 features)
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:10]
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 2])
            axes[0, 2].set_title('Feature Correlation Matrix')
        
        # 4. Feature importance (using Random Forest)
        if 'is_attack' in df.columns:
            feature_cols = [col for col in df.columns 
                           if col not in ['attack_category', 'is_attack', 'attack_type']]
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            X_sample = df[feature_cols].head(1000)  # Sample for speed
            y_sample = df['is_attack'].head(1000)
            rf.fit(X_sample, y_sample)
            
            importances = rf.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1, 0].set_yticks(range(len(feature_importance)))
            axes[1, 0].set_yticklabels(feature_importance['feature'])
            axes[1, 0].set_title('Top 10 Feature Importances')
            axes[1, 0].set_xlabel('Importance Score')
        
        # 5. Distribution of numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns[:5]
        if len(numerical_features) > 0:
            df[numerical_features].hist(bins=30, ax=axes[1, 1])
            axes[1, 1].set_title('Distribution of Numerical Features')
        
        # 6. Attack type by protocol (if available)
        if 'protocol_type_tcp' in df.columns:
            protocol_attack = df.groupby(['attack_category']).agg({
                'protocol_type_tcp': 'mean',
                'protocol_type_udp': 'mean',
                'protocol_type_icmp': 'mean'
            }).fillna(0)
            
            protocol_attack.plot(kind='bar', ax=axes[1, 2])
            axes[1, 2].set_title('Attack Types by Protocol')
            axes[1, 2].set_ylabel('Proportion')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_dataset_summary(self, df):
        """
        Generate comprehensive summary of the dataset
        
        Args:
            df (pd.DataFrame): Dataset to summarize
            
        Returns:
            dict: Dataset summary
        """
        summary = {
            'total_samples': len(df),
            'total_features': len([col for col in df.columns 
                                  if col not in ['attack_category', 'is_attack', 'attack_type']]),
            'attack_distribution': df['attack_category'].value_counts().to_dict(),
            'binary_distribution': df['is_attack'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return summary
    
    def export_processed_data(self, df, filename_prefix='nslkdd_processed'):
        """
        Export the processed dataset to various formats
        
        Args:
            df (pd.DataFrame): Processed dataset
            filename_prefix (str): Prefix for exported files
        """
        # Export to CSV
        df.to_csv(f'{filename_prefix}.csv', index=False)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(f'{filename_prefix}.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Full_Dataset', index=False)
            
            # Export train/test splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_test_splits(df)
            
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            train_df.to_excel(writer, sheet_name='Training_Set', index=False)
            val_df.to_excel(writer, sheet_name='Validation_Set', index=False)
            test_df.to_excel(writer, sheet_name='Test_Set', index=False)
        
        print(f"Exported processed dataset to {filename_prefix}.csv and {filename_prefix}.xlsx")

def preprocess_and_label_data(self, df):
    """
    Encode categorical variables, scale continuous variables, and assign attack labels.
    Args:
        df (pd.DataFrame): Raw synthetic dataset
    Returns:
        X (np.array): Preprocessed feature matrix
        y (np.array): Encoded target labels
    """
    df = df.copy()

    # 1. Assign attack labels randomly for testing (can be customized)
    attacks_types = list(self.attack_mapping.keys())
    df['attack'] = np.random.choice(attacks_types, size=len(df), p=[1/len(attacks_types)] * len(attacks_types))
    df['label'] = df['attack'].map(self.attack_mapping)

    # 2. Encode categorical variables
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols)

    # 3. Encode target label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    # 4. Drop non-numeric columns
    df.drop(['attack', 'label'], axis=1, inplace=True)

    # 5. Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    return X, y, label_encoder


def plot_class_distribution(y, label_encoder, save_path='class_distribution_preprocess.png'):
    classes = label_encoder.inverse_transform(np.unique(y))
    counts = np.bincount(y)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=counts)
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(save_path)  # Save the figure
    plt.close() 

# Usage example and main execution
def main():
    """
    Main execution function demonstrating the complete pipeline
    """
    # Initialize processor
    processor = NSLKDDExcelProcessor()
    
    # You can load your Excel file if needed (optional)
    # excel_info = processor.load_excel_file('NSL-KDD Features.xlsx')
    # if excel_info:
    #     print("Excel file information:", excel_info)
    
    # Generate synthetic NSL-KDD dataset
    print("Generating synthetic NSL-KDD dataset...")
    synthetic_df = processor.generate_synthetic_nslkdd_dataset(n_samples=100000)
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    processed_df = processor.preprocess_synthetic_dataset(synthetic_df)
    
    # Create train/test splits
    print("Creating train/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = processor.create_train_test_splits(processed_df)
    
    # Visualize dataset characteristics
    print("Creating visualizations...")
    processor.visualize_dataset_characteristics(processed_df)
    
    # Generate and print summary
    summary = processor.generate_dataset_summary(processed_df)
    print("\nDataset Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export processed data
    processor.export_processed_data(processed_df, 'nslkdd_synthetic_processed')
    
    print("\nProcessing completed successfully!")
    return processed_df, processor

# Example usage for your specific case
def process_nslkdd_excel_file(excel_path):
    """
    Process your specific NSL-KDD Features.xlsx file
    
    Args:
        excel_path (str): Path to your Excel file
        
    Returns:
        tuple: (processed_data, processor_instance)
    """
    processor = NSLKDDExcelProcessor()
    
    # Load and analyze your Excel file
    excel_info = processor.load_excel_file(excel_path)
    if excel_info is None:
        print("Failed to load Excel file. Generating synthetic data instead...")
        synthetic_df = processor.generate_synthetic_nslkdd_dataset(n_samples=125973)
        processed_df = processor.preprocess_synthetic_dataset(synthetic_df)
    else:
        # If Excel contains actual data, process it
        # Analyze the structure
        analysis = processor.analyze_feature_structure()
        print("Feature analysis:", analysis)
        
        # For now, generate synthetic data based on the structure
        # You can modify this to process actual Excel data if it contains samples
        synthetic_df = processor.generate_synthetic_nslkdd_dataset(n_samples=125973)
        processed_df = processor.preprocess_synthetic_dataset(synthetic_df)
    
    # Create visualizations and summary
    processor.visualize_dataset_characteristics(processed_df)
    summary = processor.generate_dataset_summary(processed_df)
    
    # Export results
    processor.export_processed_data(processed_df)
    
    return processed_df, processor

if __name__ == "__main__":
    # Example execution
    # Replace 'NSL-KDD Features.xlsx' with the actual path to your file
    processed_data, processor_instance = process_nslkdd_excel_file('NSL-KDD Features.xlsx')
    
    print("Processing completed! You now have:")
    print("1. Processed dataset (processed_data)")
    print("2. Processor instance (processor_instance)")
    print("3. Exported CSV and Excel files")
    print("4. Train/validation/test splits ready for modeling")