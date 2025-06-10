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

# Keras/TensorFlow imports for MLP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input # Added Input layer for clarity in MLP
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')

class NSLKDDExcelProcessor:
    """
    Advanced processor for NSL-KDD Excel files with synthetic data generation capabilities
    and MLP model training/evaluation.
    """

    def __init__(self):
        self.feature_info = None
        self.processed_data = None
        self.scalers = {}
        self.label_encoders = {}
        self.attack_mapping = self._create_attack_mapping()
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = (None,) * 6
        self.mlp_model = None # Changed model attribute to mlp_model

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

                synthetic_data[feature] = values.astype(float) # Ensure float type

            elif spec['type'] == 'categorical':
                # Generate categorical data
                synthetic_data[feature] = np.random.choice(
                    spec['values'],
                    size=n_samples,
                    p=spec['weights']
                )

            elif spec['type'] == 'binary':
                # Generate binary data
                synthetic_data[feature] = np.random.binomial(1, spec['prob'], size=n_samples).astype(int)

        # Generate attack labels with realistic distribution
        attack_types = list(self.attack_mapping.keys())
        attack_weights = [0.5] + [0.5/len(attack_types[:-1])] * (len(attack_types) - 1)  # 50% normal, 50% attacks
        synthetic_data['attack_type'] = np.random.choice(attack_types, size=n_samples, p=attack_weights)

        # Create DataFrame
        df = pd.DataFrame(synthetic_data)

        # Explicitly set 'num_outbound_cmds' to numeric, as it's always 0
        if 'num_outbound_cmds' in df.columns:
            df['num_outbound_cmds'] = df['num_outbound_cmds'].astype(float)


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
        # Filter for features actually present in the DataFrame
        categorical_features_present = [f for f in categorical_features if f in processed_df.columns]

        for feature in categorical_features_present:
            dummies = pd.get_dummies(processed_df[feature], prefix=feature, prefix_sep='_')
            processed_df = pd.concat([processed_df, dummies], axis=1)
            processed_df.drop(feature, axis=1, inplace=True) # Drop original categorical column

        # 4. Scale numerical features
        # First, ensure all remaining columns that *should* be numeric are numeric
        for col in processed_df.columns:
            # Attempt to convert to numeric, coerce errors will turn non-numeric into NaN
            # Then fill NaNs if they appear in numerical features, or drop rows if critical
            processed_df[col] = pd.to_numeric(processed_df[col], errors='ignore')

        numerical_features = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col not in ['is_attack', 'attack_type', 'attack_category']]

        # Handle potential NaNs in numerical features before scaling (e.g., from `errors='coerce'`)
        # Filling with mean is a common strategy
        for col in numerical_features:
            if processed_df[col].isnull().any():
                mean_val = processed_df[col].mean()
                processed_df[col] = processed_df[col].fillna(mean_val)
                print(f"Filled {processed_df[col].isnull().sum()} NaN values in '{col}' with mean: {mean_val:.2f}")


        scaler = StandardScaler()
        if numerical_features: # Only scale if there are numerical features
            processed_df[numerical_features] = scaler.fit_transform(processed_df[numerical_features])
            self.scalers['numerical'] = scaler
        else:
            print("No numerical features found for scaling after one-hot encoding and initial cleanup.")


        # 5. Feature selection using correlation
        # Temporarily drop non-numeric and target columns to calculate correlations
        X_for_corr = processed_df.select_dtypes(include=[np.number]).drop(['is_attack'], axis=1, errors='ignore')
        y_for_corr = processed_df['is_attack']

        top_features = []
        if not X_for_corr.empty and not y_for_corr.empty:
            # Check for columns with zero variance after scaling; these can cause issues with correlation
            # and don't provide useful information.
            zero_variance_cols = X_for_corr.columns[X_for_corr.var() == 0]
            if not zero_variance_cols.empty:
                print(f"Dropping zero variance columns before correlation: {list(zero_variance_cols)}")
                X_for_corr = X_for_corr.drop(columns=zero_variance_cols)

            if not X_for_corr.empty:
                correlations = X_for_corr.corrwith(y_for_corr).abs().sort_values(ascending=False)
                top_features = correlations.head(20).index.tolist()
                print(f"Selected {len(top_features)} top features based on correlation.")
            else:
                print("X_for_corr is empty after dropping zero variance columns; cannot calculate correlations.")
        else:
            print("X_for_corr or y_for_corr is empty; skipping feature selection by correlation.")


        # Create final dataset *only with selected top features and target columns*
        # Ensure all columns in top_features actually exist in processed_df
        final_selected_features = [f for f in top_features if f in processed_df.columns]
        # Add target columns explicitly back
        final_df = processed_df[final_selected_features + ['attack_category', 'is_attack', 'attack_type']].copy()


        # Final check: Ensure all feature columns in final_df are numeric before storing
        # Identify columns that are NOT the target columns and are still 'object' type
        non_numeric_cols_in_features = final_df[final_selected_features].select_dtypes(include=['object']).columns
        if not non_numeric_cols_in_features.empty:
            print(f"Warning: Non-numeric columns found in final features: {list(non_numeric_cols_in_features)}")
            # Attempt to convert to numeric, dropping if conversion is not possible
            for col in non_numeric_cols_in_features:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            final_df = final_df.dropna(subset=non_numeric_cols_in_features)
            print("Attempted to convert non-numeric features to numeric and dropped rows with unconvertible values.")

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
        # IMPORTANT: Ensure X contains ONLY numerical features
        # Drop 'attack_type' and 'attack_category' from X as they are not features for the model
        feature_cols_candidate = [col for col in df.columns
                                  if col not in ['attack_category', 'is_attack', 'attack_type']]

        X = df[feature_cols_candidate].select_dtypes(include=[np.number]) # Explicitly select only numeric dtypes for X
        y = df['is_attack']

        if X.empty:
            raise ValueError("Feature DataFrame (X) is empty after selecting numeric dtypes. Check preprocessing steps.")

        # Ensure y is also numeric (it should be int from earlier steps)
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target variable (y) is not numeric. Check preprocessing steps for 'is_attack'.")


        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: train vs validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
        )

        print(f"Dataset splits:")
        print(f"Training: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"Validation: {self.X_val.shape[0]} samples, {self.X_val.shape[1]} features")
        print(f"Testing: {self.X_test.shape[0]} samples, {self.X_test.shape[1]} features")

        # Confirm dtypes one last time before returning
        print("X_train dtypes after splitting:", self.X_train.dtypes.value_counts())
        print("X_val dtypes after splitting:", self.X_val.dtypes.value_counts())
        print("X_test dtypes after splitting:", self.X_test.dtypes.value_counts())


        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def visualize_dataset_characteristics(self, df):
        """
        Create comprehensive visualizations of the dataset

        Args:
            df (pd.DataFrame): Dataset to visualize
        """
        # Determine how many plots are needed in the second row for numerical features
        numerical_features_for_plot = df.select_dtypes(include=[np.number]).columns
        numerical_features_for_plot = [col for col in numerical_features_for_plot if col not in ['is_attack']]

        # Take a reasonable number of numerical features to plot, e.g., max 6
        num_features_to_hist = min(len(numerical_features_for_plot), 6) # Plot up to 6 numerical features

        # Let's refactor the visualization to use separate figures for better control and clarity,
        # especially since you're hitting issues with subplots.

        # 1. Attack type distribution
        plt.figure(figsize=(8, 6))
        attack_counts = df['attack_category'].value_counts()
        plt.pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%')
        plt.title('Attack Type Distribution')
        plt.savefig('attack_type_distribution.png')
        plt.show()

        # 2. Binary classification distribution
        plt.figure(figsize=(8, 6))
        binary_counts = df['is_attack'].value_counts()
        plt.bar(['Normal', 'Attack'], binary_counts.values, color=['green', 'red'])
        plt.title('Binary Classification Distribution')
        plt.ylabel('Count')
        plt.savefig('binary_classification_distribution.png')
        plt.show()

        # 3. Feature correlation heatmap (top 10 features)
        numerical_cols_for_corr = df.select_dtypes(include=[np.number]).columns.drop('is_attack', errors='ignore')
        if len(numerical_cols_for_corr) > 1:
            plt.figure(figsize=(10, 8))
            # Select top 10 features by variance or by some other criteria if correlation is not directly available
            # For now, let's just take the first 10 numerical columns that are not 'is_attack'
            corr_features = numerical_cols_for_corr[:10]
            if len(corr_features) > 1: # Ensure there are at least two features to correlate
                corr_matrix = df[corr_features].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Feature Correlation Matrix (Top Numerical Features)')
                plt.tight_layout()
                plt.savefig('feature_correlation_heatmap.png')
                plt.show()
            else:
                print("Not enough numerical features to plot correlation heatmap.")


        # 4. Feature importance (using Random Forest)
        if 'is_attack' in df.columns:
            feature_cols = [col for col in df.columns
                            if col not in ['attack_category', 'is_attack', 'attack_type']]

            # Ensure feature_cols are all numeric before passing to RandomForest
            feature_cols_numeric = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

            if len(feature_cols_numeric) > 0:
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                # Ensure X_sample and y_sample are not empty
                if not df.empty:
                    X_sample = df[feature_cols_numeric].head(min(1000, len(df)))  # Sample for speed, adapt to df size
                    y_sample = df['is_attack'].head(min(1000, len(df)))

                    if not X_sample.empty:
                        rf.fit(X_sample, y_sample)

                        importances = rf.feature_importances_
                        feature_importance = pd.DataFrame({
                            'feature': feature_cols_numeric, # Use numeric feature names
                            'importance': importances
                        }).sort_values('importance', ascending=False).head(10)

                        plt.figure(figsize=(10, 6))
                        sns.barplot(x='importance', y='feature', data=feature_importance)
                        plt.title('Top 10 Feature Importances (Random Forest)')
                        plt.xlabel('Importance Score')
                        plt.ylabel('Feature')
                        plt.tight_layout()
                        plt.savefig('feature_importances.png')
                        plt.show()
                    else:
                        print("Cannot calculate feature importances: X_sample is empty.")
                else:
                    print("Cannot calculate feature importances: DataFrame is empty.")
            else:
                print("No numerical features available for feature importance calculation.")

        # 5. Distribution of numerical features
        if len(numerical_features_for_plot) > 0:
            # Create subplots for numerical feature distributions dynamically
            rows = int(np.ceil(num_features_to_hist / 2)) # Arrange in 2 columns
            fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
            axes = axes.flatten() # Flatten for easy iteration

            for i, col in enumerate(numerical_features_for_plot[:num_features_to_hist]):
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.savefig('numerical_feature_distributions.png')
            plt.show()
        else:
            print("No numerical features to plot distributions.")

        # 6. Attack type by protocol (if available) - this logic was a bit off before, if you want specific protocol analysis, need to adjust
        protocol_cols = [col for col in df.columns if col.startswith('protocol_type_')]
        if len(protocol_cols) > 0 and 'attack_category' in df.columns:
            plt.figure(figsize=(10, 6))
            # Aggregate based on 'attack_category' and the one-hot encoded protocols
            # We will group by attack category and sum the protocol columns to see count in each
            # Or better, use a stacked bar plot or count plot for better visualization
            df_protocol_attack = df.groupby('attack_category')[protocol_cols].sum()
            if not df_protocol_attack.empty:
                df_protocol_attack.plot(kind='bar', stacked=True, ax=plt.gca())
                plt.title('Attack Types by Protocol Distribution')
                plt.xlabel('Attack Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Protocol Type')
                plt.tight_layout()
                plt.savefig('attack_by_protocol.png')
                plt.show()
            else:
                print("Aggregated protocol attack data is empty; skipping plot.")
        else:
            print("Protocol type data or attack category not available for 'Attack Types by Protocol' plot.")


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
            # Ensure splits are already created before trying to export
            if self.X_train is not None and self.y_train is not None:
                train_df = pd.concat([self.X_train, self.y_train], axis=1)
                val_df = pd.concat([self.X_val, self.y_val], axis=1)
                test_df = pd.concat([self.X_test, self.y_test], axis=1)

                train_df.to_excel(writer, sheet_name='Training_Set', index=False)
                val_df.to_excel(writer, sheet_name='Validation_Set', index=False)
                test_df.to_excel(writer, sheet_name='Test_Set', index=False)
            else:
                print("Train/Validation/Test splits not found. Please run create_train_test_splits first.")


        print(f"Exported processed dataset to {filename_prefix}.csv and {filename_prefix}.xlsx")

    # --- MLP Model Methods ---
    def build_mlp_model(self, input_dim, num_classes):
        """
        Builds and compiles an MLP model for binary classification.

        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes (1 for binary classification with sigmoid).
        """
        model = Sequential([
            Input(shape=(input_dim,)), # Input layer, expects 2D data (samples, features)
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='sigmoid') # Sigmoid for binary classification
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.mlp_model = model # Assign to mlp_model
        print("MLP model built and compiled.")
        self.mlp_model.summary()

    def train_mlp_model(self, epochs=50, batch_size=32):
        """
        Trains the built MLP model.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        if self.mlp_model is None:
            print("MLP model not built. Please call build_mlp_model first.")
            return

        if self.X_train is None or self.y_train is None:
            print("Training data not available. Please run create_train_test_splits first.")
            return

        # MLP expects 2D input: (samples, features)
        # No reshaping is needed here as X_train and X_val are already 2D DataFrames/NumPy arrays
        X_train_flat = self.X_train.to_numpy()
        X_val_flat = self.X_val.to_numpy()


        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        print("Training MLP model...")
        history = self.mlp_model.fit( # Use mlp_model
            X_train_flat,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_flat, self.y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        print("MLP model training complete.")

        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('MLP Model Accuracy') # Changed title
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('MLP Model Loss') # Changed title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('mlp_training_history.png') # Changed filename
        plt.show()

    def evaluate_mlp_model(self):
        """
        Evaluates the trained MLP model on the test set.
        """
        if self.mlp_model is None: # Use mlp_model
            print("MLP model not trained. Please call train_mlp_model first.")
            return

        if self.X_test is None or self.y_test is None:
            print("Test data not available. Please run create_train_test_splits first.")
            return

        # MLP expects 2D input: (samples, features)
        X_test_flat = self.X_test.to_numpy()


        print("\nEvaluating MLP model on test set...")
        loss, accuracy = self.mlp_model.evaluate(X_test_flat, self.y_test, verbose=0) # Use mlp_model
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        y_pred_proba = self.mlp_model.predict(X_test_flat) # Use mlp_model
        # For binary classification with sigmoid output, y_pred_proba will have shape (samples, 1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten() # Flatten to 1D array for classification_report

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        plt.title('MLP Confusion Matrix') # Changed title
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('mlp_confusion_matrix.png') # Changed filename
        plt.show()

        # ROC Curve
        # For sigmoid output, y_pred_proba[:, 0] correctly gives the probability of the positive class
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba) # Use y_pred_proba directly for ROC
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MLP Receiver Operating Characteristic (ROC) Curve') # Changed title
        plt.legend(loc="lower right")
        plt.savefig('mlp_roc_curve.png') # Changed filename
        plt.show()


# Example usage for your specific case
def process_nslkdd_excel_file(excel_path):
    """
    Process your specific NSL-KDD Features.xlsx file
    and train/evaluate an MLP model.

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
    print("Creating visualizations...")
    processor.visualize_dataset_characteristics(processed_df)
    summary = processor.generate_dataset_summary(processed_df)
    print("\nDataset Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Create train/test splits (important for MLP training)
    processor.create_train_test_splits(processed_df)

    # Build, train, and evaluate MLP model
    # The input_dim for MLP is just the number of features
    input_dim = processor.X_train.shape[1]
    # num_classes should be 1 for binary classification with 'sigmoid' activation in the final Dense layer.
    num_classes = 1

    processor.build_mlp_model(input_dim, num_classes=num_classes) # Changed to build_mlp_model
    processor.train_mlp_model(epochs=20, batch_size=64) # Changed to train_mlp_model
    processor.evaluate_mlp_model() # Changed to evaluate_mlp_model

    # Export results
    processor.export_processed_data(processed_df)

    return processed_df, processor

if __name__ == "__main__":
    # Example execution
    # Replace 'NSL-KDD Features.xlsx' with the actual path to your file
    processed_data, processor_instance = process_nslkdd_excel_file('/home/asterbyte/machine_test/intrud_detection/Intrusion-detection-DL-ML/NSL-KDD Features.xlsx')

    print("Processing completed! You now have:")
    print("1. Processed dataset (processed_data)")
    print("2. Processor instance (processor_instance)")
    print("3. Exported CSV and Excel files")
    print("4. Train/validation/test splits ready for modeling")
    print("5. MLP model trained and evaluated with performance plots.") # Changed message