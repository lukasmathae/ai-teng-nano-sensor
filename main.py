from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import preprocessing as prep



def main():
    base_dir = "AI-TENG-Nano-Sensor-main/TENG-Nano-Sensor-Data/Rehabilitation monitoring"

    X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test, X,y = prep.preprocessing(base_dir)

    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    print("\n--- Validation Classification Report ---")
    print(classification_report(y_val, y_val_pred))

    # 📊 Final test set evaluation
    y_test_pred = clf.predict(X_test)
    print("\n--- Test Classification Report ---")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix

    cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(y))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y), cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()