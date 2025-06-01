import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import joblib


import preprocessing as prep

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping  # <-- Added import
from tensorflow.python.client import device_lib


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, labels, model_name="Model", is_keras=False):
    print(f"\n--- Training {model_name} ---")
    if is_keras:
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
        y_val_pred_probs = model.predict(X_val)
        y_val_pred = np.argmax(y_val_pred_probs, axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        y_test_pred_probs = model.predict(X_test)
        y_test_pred = np.argmax(y_test_pred_probs, axis=1)
        y_test_true = np.argmax(y_test, axis=1)

        print(f"\n--- {model_name} Validation Classification Report ---")
        print(classification_report(y_val_true, y_val_pred, target_names=labels))

        print(f"\n--- {model_name} Test Classification Report ---")
        print(classification_report(y_test_true, y_test_pred, target_names=labels))

        cm = confusion_matrix(y_test_true, y_test_pred)
        model.save("models/cnn_model.keras")

    else:
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        print(f"\n--- {model_name} Validation Classification Report ---")
        print(classification_report(y_val, y_val_pred, target_names=labels))

        print(f"\n--- {model_name} Test Classification Report ---")
        print(classification_report(y_test, y_test_pred, target_names=labels))

        cm = confusion_matrix(y_test, y_test_pred, labels=labels)
        joblib.dump(model, f"models/{model_name.replace(' ', '_').lower()}.pkl")


    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"plots/{model_name.replace(' ', '_').lower()}.png")
    plt.close()

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


def main():
    base_dir = "AI-TENG-Nano-Sensor-main/TENG-Nano-Sensor-Data/Rehabilitation monitoring"

    X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test, X, y = prep.preprocessing_traditionalML(base_dir)
    labels = np.unique(y)

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

    for name, model in models.items():
        train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, labels, model_name=name)

    # CNN part
    X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test, X, y = prep.preprocessing_CNN(base_dir)
    labels = np.unique(y)


    # Encode labels
    encoder = LabelEncoder()
    y_train_enc = to_categorical(encoder.fit_transform(y_train))
    y_val_enc = to_categorical(encoder.transform(y_val))
    y_test_enc = to_categorical(encoder.transform(y_test))
    num_classes = y_train_enc.shape[1]

    # Build & train CNN model with EarlyStopping
    cnn = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    cnn.fit(X_train, y_train_enc,
            validation_data=(X_val, y_val_enc),
            epochs=50,  # Increased epochs
            batch_size=32,
            callbacks=[early_stop])

    # Evaluate on test set and print reports
    y_pred_probs = cnn.predict(X_test)
    y_pred = encoder.inverse_transform(np.argmax(y_pred_probs, axis=1))
    y_true = encoder.inverse_transform(np.argmax(y_test_enc, axis=1))

    train_and_evaluate_model(cnn, X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, labels=encoder.classes_,
                             model_name="CNN", is_keras=True)

    '''
    # For example, check training labels distribution
    counter = Counter(y_train)
    print("Class distribution in training set:")
    for cls, count in counter.items():
        print(f"{cls}: {count} samples")

    print("Training set class distribution:", Counter(y_train))
    print("Validation set class distribution:", Counter(y_val))
    print("Test set class distribution:", Counter(y_test))
    '''

if __name__ == "__main__":
    main()
