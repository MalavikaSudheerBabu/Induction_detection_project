import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape, UpSampling1D, Input, Concatenate, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Dummy data for demonstration if not running with NSLKDDExcelProcessor
# In a real scenario, X_train, Y_train etc., would come from your NSLKDDExcelProcessor
try:
    # Attempt to use data from NSLKDDExcelProcessor if it's running
    if 'processor' in locals() and processor.X_train is not None:
        X_train_raw = processor.X_train.to_numpy()
        X_val_raw = processor.X_val.to_numpy()
        X_test_raw = processor.X_test.to_numpy()
        y_train_raw = processor.y_train.to_numpy()
        y_val_raw = processor.y_val.to_numpy()
        y_test_raw = processor.y_test.to_numpy()
        print("Using data from NSLKDDExcelProcessor.")
    else:
        raise NameError # Force dummy data if processor isn't active
except NameError:
    print("NSLKDDExcelProcessor data not found. Generating dummy data for demonstration.")
    # Generate dummy data for a basic test if NSLKDDExcelProcessor isn't integrated
    # You would typically have a dataset like the preprocessed NSL-KDD
    num_features = 40 # Example number of features
    num_samples = 10000
    X_dummy = np.random.rand(num_samples, num_features) * 10
    y_dummy = np.random.randint(0, 2, num_samples) # Binary labels (0 or 1)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy)
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_train_raw, y_train_raw, test_size=0.25, random_state=42, stratify=y_train_raw) # 0.25 of 0.8 is 0.2

# Ensure Y is one-hot encoded for CGAN
# Determine num_classes (2 for binary classification)
num_classes = len(np.unique(y_train_raw))
print(f"Detected {num_classes} classes for Y.")

encoder = OneHotEncoder(sparse_output=False) # Changed sparse to sparse_output for newer sklearn
Y_train_onehot = encoder.fit_transform(y_train_raw.reshape(-1, 1))
Y_val_onehot = encoder.transform(y_val_raw.reshape(-1, 1))
Y_test_onehot = encoder.transform(y_test_raw.reshape(-1, 1))

# Reshape X data to (samples, 1, features) for Conv1D
X_train = X_train_raw.reshape(X_train_raw.shape[0], 1, X_train_raw.shape[1])
X_val = X_val_raw.reshape(X_val_raw.shape[0], 1, X_val_raw.shape[1])
X_test = X_test_raw.reshape(X_test_raw.shape[0], 1, X_test_raw.shape[1])

print(f"X_train shape: {X_train.shape}, Y_train_onehot shape: {Y_train_onehot.shape}")
print(f"X_test shape: {X_test.shape}, Y_test_onehot shape: {Y_test_onehot.shape}")

# Define latent dimension and data dimension
latent_dim = 100 # Size of the noise vector
data_dim = X_train.shape[2] # Number of features in your data


# --- Conditional Generator ---
def build_conditional_generator(latent_dim, data_dim, num_classes):
    # Input for noise vector
    noise_input = Input(shape=(latent_dim,))
    # Input for conditional label (one-hot encoded)
    label_input = Input(shape=(num_classes,))

    # Concatenate noise and label inputs
    merged_input = Concatenate()([noise_input, label_input])

    # Generator layers
    # Start with a dense layer to expand the combined input
    gen = Dense(256)(merged_input)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.BatchNormalization(momentum=0.8)(gen) # BatchNormalization
    gen = Dense(512)(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.BatchNormalization(momentum=0.8)(gen)
    gen = Dense(data_dim)(gen) # Output features
    # Reshape to (1, data_dim) for Conv1D compatibility
    output = Reshape((1, data_dim))(gen)

    model = Model([noise_input, label_input], output, name='generator')
    return model

# --- Conditional Discriminator ---
def build_conditional_discriminator(data_dim, num_classes):
    # Input for real/fake data (features)
    data_input = Input(shape=(1, data_dim))
    # Input for conditional label (one-hot encoded)
    label_input = Input(shape=(num_classes,))

    # Flatten the data input for concatenation
    flat_data = Flatten()(data_input)
    # Concatenate flattened data and label inputs
    merged_input = Concatenate()([flat_data, label_input])

    # Discriminator layers
    dis = Dense(512)(merged_input)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.3)(dis)
    dis = Dense(256)(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.3)(dis)

    # Output for real/fake classification
    validity = Dense(1, activation='sigmoid', name='discriminator_output')(dis)

    model = Model([data_input, label_input], validity, name='discriminator')
    return model

# Build Generator and Discriminator
generator = build_conditional_generator(latent_dim, data_dim, num_classes)
discriminator = build_conditional_discriminator(data_dim, num_classes)

# Compile discriminator separately
discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# --- Combined GAN Model (for Generator training) ---
# Freeze discriminator weights when training the generator
discriminator.trainable = False

# Generator's inputs: noise and label
noise_input = Input(shape=(latent_dim,))
label_input = Input(shape=(num_classes,))

# Generate fake data
generated_data = generator([noise_input, label_input])

# Discriminator's output on generated data
discriminator_output = discriminator([generated_data, label_input])

# Combined GAN model: Generator -> Discriminator
gan = Model([noise_input, label_input], discriminator_output, name='gan_model')
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

generator.summary()
discriminator.summary()
gan.summary()


# --- Training Loop (Adversarial Training) ---
epochs = 200  #5000 # Increased epochs for better GAN training (still lower than ideal for real data)
batch_size = 36  #2785 # Keep your batch size
sample_interval = 50 #500 # Print progress every X epochs

# Labels for real and fake data
real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

print("\n--- Starting Conditional GAN Training Loop ---")
for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------
    # Select a random batch of real data and corresponding labels
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_data_batch = X_train[idx]
    real_label_batch = Y_train_onehot[idx]

    # Generate a batch of fake data with random labels
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # Randomly select labels for generator to produce conditioned data
    random_labels_for_gen = Y_train_onehot[np.random.randint(0, Y_train_onehot.shape[0], batch_size)]
    fake_data_batch = generator.predict([noise, random_labels_for_gen], verbose=0)


    # Train the discriminator on real and fake data
    d_loss_real = discriminator.train_on_batch([real_data_batch, real_label_batch], real_labels)
    d_loss_fake = discriminator.train_on_batch([fake_data_batch, random_labels_for_gen], fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------
    # Generate new noise and random labels for generator training
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    random_labels_for_gen_g = Y_train_onehot[np.random.randint(0, Y_train_onehot.shape[0], batch_size)]

    # Train the generator (via the combined GAN model) to fool the discriminator
    g_loss = gan.train_on_batch([noise, random_labels_for_gen_g], real_labels)

    # Plot the progress
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

print("\n--- Conditional GAN Training Finished ---")


discriminator_feature_extractor = Model(discriminator.inputs, discriminator.layers[-2].output)
discriminator_feature_extractor.trainable = False # Freeze these layers for classification head training

# 2. Build a small classifier head on top of the discriminator's extracted features
classifier_input = Input(shape=(discriminator_feature_extractor.output.shape[1],))
classifier_output = Dense(num_classes, activation='softmax')(classifier_input) # Softmax for multi-class

classification_model = Model(classifier_input, classifier_output, name='classification_head')
classification_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classification_model.summary()

# 3. Train the classification model on real data features extracted by the discriminator
print("\n--- Training Discriminator-based Classifier ---")
# Extract features for training the classifier
X_train_features = discriminator_feature_extractor.predict([X_train, Y_train_onehot], verbose=0)
X_val_features = discriminator_feature_extractor.predict([X_val, Y_val_onehot], verbose=0)

classification_history = classification_model.fit(
    X_train_features, Y_train_onehot,
    epochs=50, # Fewer epochs for this new head
    batch_size=batch_size,
    validation_data=(X_val_features, Y_val_onehot),
    verbose=1
)
print("--- Discriminator-based Classifier Training Finished ---")

# --- Final Evaluation ---
print("\n--- Evaluating Discriminator-based Classifier ---")
X_test_features = discriminator_feature_extractor.predict([X_test, Y_test_onehot], verbose=0) # Use Y_test_onehot to get conditioned features

# Make predictions using the trained classification model
y_pred_proba = classification_model.predict(X_test_features)
y_pred_classes = np.argmax(y_pred_proba, axis=1) # Get class index

# For Y_test, use original raw labels for metrics calculation
y_true_classes = y_test_raw # Use the original 0/1 labels for comparison

target_names = [str(i) for i in range(num_classes)]
if num_classes == 2:
    target_names = ['Normal', 'Attack'] # Assuming 0 is Normal, 1 is Attack from NSL-KDD

print("\nClassification Report (Discriminator as Classifier):")
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

print("\nConfusion Matrix (Discriminator as Classifier):")
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Discriminator-based Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('discriminator_classifier_confusion_matrix.png')
plt.show()


# Metrics
print("Accuracy Score          =     ", accuracy_score(y_true_classes, y_pred_classes))
# Precision, Recall, F1 for binary classification (assuming 1 is positive class 'Attack')
if num_classes == 2:
    print("Precision Score         =     ", precision_score(y_true_classes, y_pred_classes, pos_label=1))
    print("Recall/Sensitivity      =     ", recall_score(y_true_classes, y_pred_classes, pos_label=1))
    print("Specificity             =     ", recall_score(y_true_classes, y_pred_classes, pos_label=0))
    print("F1 Score                =     ", f1_score(y_true_classes, y_pred_classes, pos_label=1))
else:
    print("Precision Score (macro) =     ", precision_score(y_true_classes, y_pred_classes, average='macro'))
    print("Recall/Sensitivity (macro) =  ", recall_score(y_true_classes, y_pred_classes, average='macro'))
    print("F1 Score (macro)        =     ", f1_score(y_true_classes, y_pred_classes, average='macro'))


# ROC Curve (for binary classification only)
if num_classes == 2:
    # Need probabilities for the 'positive' class (Attack, index 1)
    y_pred_proba_positive = y_pred_proba[:, 1]
    fpr, tpr, threshold = roc_curve(y_true_classes, y_pred_proba_positive)
    auc_v = auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Discriminator-based Classifier (area = {:.3f})'.format(auc_v))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Discriminator-based Classifier ROC curve')
    plt.legend(loc='best')
    plt.savefig('discriminator_classifier_roc_curve.png')
    plt.show()
else:
    print("\nROC curve not applicable for multi-class classification in this direct manner.")


# Plot model architectures
tf.keras.utils.plot_model(generator, to_file='GAN_generator.png', show_shapes=True, show_layer_activations=True, show_dtype=True, show_layer_names=True )
tf.keras.utils.plot_model(discriminator, to_file='GAN_discriminator.png', show_shapes=True, show_layer_activations=True, show_dtype=True, show_layer_names=True )
tf.keras.utils.plot_model(gan, to_file='GAN_combined.png', show_shapes=True, show_layer_activations=True, show_dtype=True, show_layer_names=True )