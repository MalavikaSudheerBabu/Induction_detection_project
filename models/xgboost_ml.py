import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
import xgboost as xgb

# ==========================
# Dataset Generator Class
# ==========================

class NSLKDDDatasetGenerator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

        self.attack_mapping = {
            # DoS
            'apache2': 'DoS', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
            'mailbomb': 'DoS', 'pod': 'DoS', 'processtable': 'DoS', 'smurf': 'DoS',
            'teardrop': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',

            # R2L
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'httptunnel': 'R2L',
            'imap': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L',
            'sendmail': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
            'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',

            # Probe
            'ipsweep': 'Probe', 'mscan': 'Probe', 'nmap': 'Probe',
            'portsweep': 'Probe', 'saint': 'Probe', 'satan': 'Probe',

            # U2R
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
            'ps': 'U2R', 'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',

            # Normal
            'normal': 'Normal'
        }

    def generate_nslkdd_dataset(self, n_samples=50000):
        print(f"Generating NSL-KDD dataset with {n_samples} samples...")
        data = {}

        # === Basic features
        data['duration'] = np.random.exponential(scale=30, size=n_samples)
        data['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'], size=n_samples, p=[0.8, 0.15, 0.05])
        data['service'] = np.random.choice(['http', 'smtp', 'ftp', 'telnet', 'ssh', 'pop3', 'imap4', 'other'],
                                           size=n_samples, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1])
        data['flag'] = np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH'],
                                        size=n_samples,
                                        p=[0.6, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005])
        data['src_bytes'] = np.random.exponential(scale=1000, size=n_samples)
        data['dst_bytes'] = np.random.exponential(scale=1000, size=n_samples)
        data['land'] = np.random.binomial(1, 0.001, size=n_samples)
        data['wrong_fragment'] = np.random.poisson(lam=0.1, size=n_samples)
        data['urgent'] = np.random.poisson(lam=0.05, size=n_samples)

        # === Content features
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
        data['num_outbound_cmds'] = np.zeros(n_samples)
        data['is_host_login'] = np.random.binomial(1, 0.001, size=n_samples)
        data['is_guest_login'] = np.random.binomial(1, 0.001, size=n_samples)

        # === Traffic features
        data['count'] = np.random.exponential(scale=50, size=n_samples)
        data['srv_count'] = np.random.exponential(scale=50, size=n_samples)
        data['serror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['srv_serror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['rerror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['srv_rerror_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['same_srv_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['diff_srv_rate'] = np.random.uniform(0, 1, size=n_samples)
        data['srv_diff_host_rate'] = np.random.uniform(0, 1, size=n_samples)

        # === Host-based features
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

        # === Labels
        attack_types = list(self.attack_mapping.keys())
        attack_probs = [0.6] + [0.4 / (len(attack_types) - 1)] * (len(attack_types) - 1)
        data['attack_type'] = np.random.choice(attack_types, size=n_samples, p=attack_probs)

        df = pd.DataFrame(data)
        df['attack_category'] = df['attack_type'].map(self.attack_mapping)
        df['is_attack'] = (df['attack_category'] != 'Normal').astype(int)

        print("Dataset generated successfully!")
        print(df['attack_category'].value_counts())
        return df

    def preprocess_dataset(self, df):
        print("Preprocessing dataset...")
        df = df.copy()
        categorical = ['protocol_type', 'service', 'flag']
        df = pd.get_dummies(df, columns=categorical, drop_first=True)

        features = df.drop(columns=['attack_type', 'attack_category', 'is_attack'])
        labels = df['is_attack']
        return features, labels


# Generate and preprocess dataset
gen = NSLKDDDatasetGenerator()
raw_df = gen.generate_nslkdd_dataset(50000)
X, Y = gen.preprocess_dataset(raw_df)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train XGBoost Classifier
xgb_mod = xgb.XGBClassifier(random_state=42, gpu_id=0, use_label_encoder=False, eval_metric='logloss')
xgb_mod.fit(X_train, Y_train)

# Predictions
y_pred = xgb_mod.predict(X_test)

# Metrics
print("XGBoost Classification Report")
print(classification_report(Y_test, y_pred, target_names=['normal', 'attack']))
print("Confusion Matrix")
print(confusion_matrix(Y_test, y_pred))
print("Accuracy Score        =    ", accuracy_score(Y_test, y_pred))
print("Precision Score       =    ", precision_score(Y_test, y_pred))
print("Recall/Sensitivity    =    ", recall_score(Y_test, y_pred))
print("Specificity           =    ", recall_score(Y_test, y_pred, pos_label=0))
print("F1 Score              =    ", f1_score(Y_test, y_pred))

# ROC Curve
fpr, tpr, threshold = roc_curve(Y_test, y_pred)
auc_val = auc(fpr, tpr)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='XGBoost (AUC = {:.3f})'.format(auc_val))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
