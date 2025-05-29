import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(base_dir):

    # Mapping for cleaner labels
    file_label_map = {
        "First meta volt.xlsx": "meta",
        "Heel volt.xlsx": "heel",
        "Toe volt.xlsx": "toe",
        "U First meta volt.xlsx": "meta",
        "U Heel volt.xlsx": "heel",
        "U Toe volt.xlsx": "toe"
    }

    # Final list of dataframes
    dataframes = []

    # Loop over both Person 1 and Person 2
    for person in ["Person 1", "Person 2"]:
        person_dir = os.path.join(base_dir, person)
        person_label = person.lower().replace(" ", "")  # 'person1' or 'person2'

        for filename in os.listdir(person_dir):
            if filename.endswith(".xlsx"):
                filepath = os.path.join(person_dir, filename)
                df = pd.read_excel(filepath)

                # 🧼 Remove faulty rows
                df = df.dropna(subset=["Time (s)"])

                # Create label from folder and filename
                part_label = file_label_map.get(filename, filename.replace(".xlsx", "").lower())
                label = f"{person_label}_{part_label}"

                df["Label"] = label
                dataframes.append(df)

    # Combine all into one DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def sliding_window(combined_df):

    # Parameters
    window_size = 100  # adjust based on your data rate and action duration
    stride = 50  # 50% overlap

    X = []
    y = []

    # Group by label and slide windows
    for label, group in combined_df.groupby("Label"):
        voltages = group["Voltage (V)"].values

        for start in range(0, len(voltages) - window_size + 1, stride):
            end = start + window_size
            window = voltages[start:end]
            X.append(window)
            y.append(label)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print("X shape (samples, window_size):", X.shape)
    print("y shape (samples,):", y.shape)
    print("Labels:", set(y))

    return X, y

def flatten_data(X):

    scaler = StandardScaler()

    X_scaled = np.array([scaler.fit_transform(window.reshape(-1, 1)).flatten() for window in X])
    return X_scaled

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

    print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)
    return X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test

def preprocessing(base_dir):
    print("=============================================================================")
    print("Starting preprocessing on dir: ", base_dir)
    combined_df = load_data(base_dir)
    print("Unique labels:", combined_df['Label'].unique())

    # Output shape and first few rows
    print("Combined shape:", combined_df.shape)
    print(combined_df.head())

    X,y = sliding_window(combined_df)
    X_scaled = flatten_data(X)
    X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test = split_data(X_scaled, y)
    print("=============================================================================")
    return X_train, X_temp, y_train, y_temp, X_val, X_test, y_val, y_test, X, y