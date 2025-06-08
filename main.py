import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import joblib
import time
import preprocessing as prep
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, model_name="Model", is_keras=False):
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    
    labels = label_encoder.classes_  # Use string labels for reporting and plotting
    
    if is_keras:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)
        y_val_pred_probs = model.predict(X_val)
        y_val_pred = np.argmax(y_val_pred_probs, axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        y_test_pred_probs = model.predict(X_test)
        y_test_pred = np.argmax(y_test_pred_probs, axis=1)
        y_test_true = np.argmax(y_test, axis=1)

        val_accuracy = accuracy_score(y_val_true, y_val_pred)
        test_accuracy = accuracy_score(y_test_true, y_test_pred)
        
        print(f"\n--- {model_name} Validation Classification Report ---")
        print(classification_report(y_val_true, y_val_pred, target_names=labels))
        print(f"\n--- {model_name} Test Classification Report ---")
        print(classification_report(y_test_true, y_test_pred, target_names=labels))

        cm = confusion_matrix(y_test_true, y_test_pred, labels=label_encoder.transform(labels))
        model.save(f"models/cnn_model.keras")
        
        # Plot training and validation accuracy/loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{model_name.replace(' ', '_').lower()}_train_val_plot.png")
        plt.close()
        
    else:
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_accuracy = accuracy_score(y_val, y_val_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\n--- {model_name} Validation Classification Report ---")
        print(classification_report(y_val, y_val_pred, target_names=labels))
        print(f"\n--- {model_name} Test Classification Report ---")
        print(classification_report(y_test, y_test_pred, target_names=labels))

        cm = confusion_matrix(y_test, y_test_pred, labels=label_encoder.transform(labels))
        joblib.dump(model, f"models/{model_name.replace(' ', '_').lower()}.pkl")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name.replace(' ', '_').lower()}_cm.png")
    plt.close()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return val_accuracy, test_accuracy, training_time

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_comparison(models, val_accuracies, test_accuracies, training_times):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', color=(0.21, 0.64, 0.92, 0.8))  # Blue
    plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color=(1.0, 0.39, 0.52, 0.8))  # Red
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/model_accuracy_comparison.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, training_times, color=(0.29, 0.75, 0.75, 0.8))  # Teal
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/model_training_time_comparison.png")
    plt.close()

def main():
    base_dir = r"C:\Users\chawt\Desktop\inha_6_sem\Digital Signal Processing Final Project\ai-teng-nano-sensor\TENG-Nano-Sensor-Data\Rehabilitation monitoring"
    
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Traditional ML models
    X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test, X, y = prep.preprocessing_traditionalML(base_dir)
    
    # Encode labels for traditional ML
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    labels = label_encoder.classes_

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500),
        "Gradient Boosted Trees": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    model_names = []
    val_accuracies = []
    test_accuracies = []
    training_times = []

    for name, model in models.items():
        val_acc, test_acc, train_time = train_and_evaluate_model(
            model, X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, 
            label_encoder, model_name=name
        )
        model_names.append(name)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        training_times.append(train_time)
        print(f"{name} Training Time: {train_time:.2f} seconds")

    # CNN part
    X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test, X, y = prep.preprocessing_CNN(base_dir)
    
    # Encode labels for CNN
    y_train_enc = to_categorical(label_encoder.fit_transform(y_train))
    y_val_enc = to_categorical(label_encoder.transform(y_val))
    y_test_enc = to_categorical(label_encoder.transform(y_test))
    num_classes = y_train_enc.shape[1]

    # Build & train CNN model with EarlyStopping
    cnn = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    val_acc, test_acc, train_time = train_and_evaluate_model(
        cnn, X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, 
        label_encoder, model_name="CNN", is_keras=True
    )
    
    model_names.append("CNN")
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)
    training_times.append(train_time)
    print(f"CNN Training Time: {train_time:.2f} seconds")

    # Plot comparison of all models
    plot_comparison(model_names, val_accuracies, test_accuracies, training_times)

    # Print class distribution
    print("\nClass distribution in training set:", Counter(y_train))
    print("Validation set class distribution:", Counter(y_val))
    print("Test set class distribution:", Counter(y_test))

if __name__ == "__main__":
    main()