import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


import preprocessing as prep


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, labels, model_name="Model"):
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)

    # Validation
    y_val_pred = model.predict(X_val)
    print(f"\n--- {model_name} Validation Classification Report ---")
    print(classification_report(y_val, y_val_pred))

    # Test
    y_test_pred = model.predict(X_test)
    print(f"\n--- {model_name} Test Classification Report ---")
    print(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    base_dir = "AI-TENG-Nano-Sensor-main/TENG-Nano-Sensor-Data/Rehabilitation monitoring"

    X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test, X, y = prep.preprocessing(base_dir)
    labels = np.unique(y)

    # Try different models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    }

    for name, model in models.items():
        train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, labels, model_name=name)



if __name__ == "__main__":
    main()