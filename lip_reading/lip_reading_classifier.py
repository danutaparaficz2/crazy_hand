import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load your data
df = pd.read_csv("lip_data.csv", header=None)
labels = df.iloc[:, 0]
features = df.iloc[:, 1:]

print(f"Dataset info:")
print(f"Total samples: {len(df)}")
print(f"Classes: {labels.value_counts()}")
print(f"Features: {features.shape[1]}")

# Encode labels
label_categories = labels.astype('category').cat.categories
labels_encoded = labels.astype('category').cat.codes

# Split into train/test with stratification to ensure all classes are represented
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Scale features - important for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Try multiple classifiers
classifiers = {
    'SVM (RBF)': SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'),
    'SVM (Linear)': SVC(kernel='linear', probability=True, C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
}

best_accuracy = 0
best_clf = None
best_name = ""

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")
    
    # Train classifier
    if 'SVM' in name:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:  # Random Forest doesn't need scaling
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Keep track of best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_clf = clf
        best_name = name
        best_scaler = scaler if 'SVM' in name else None
    
    # Get unique labels present in test set for proper classification report
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    target_names_filtered = [label_categories[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names_filtered))

print(f"\nBest model: {best_name} with accuracy: {best_accuracy:.3f}")

# Save the model for later use
import joblib
joblib.dump(best_clf, "lip_classifier.pkl")

# Also save the scaler if it was used
if best_scaler is not None:
    joblib.dump(best_scaler, "lip_scaler.pkl")
    print("Saved both classifier and scaler.")
else:
    print("Saved classifier only (no scaling needed).")