import os
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

feature_path = os.path.join("..", "..", "data", "processed" , "real_features_semg.csv")
models_dir = os.path.join( "..", "models") 
reports_dir = os.path.join("..","..", "reports")

df = pd.read_csv(feature_path)

# "0" is the generated signal between performing other classes 
df = df[df["class"] !=0]

feature_cols= [c for c in df.columns if c.startswith("RMS") or c.startswith("MAV")]

X = df[feature_cols]
y = df["class"]

# Spliting Dataset into Training (80%) and Test(20%)
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 ,random_state=42, stratify=y)

# Initializing and training Random forrest classifier 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


print("--- Model Evaluation :")
print(f"Test Accuracy: {acc * 100:.2f}%\n")



model_path = os.path.join(models_dir, "gesture_rf_model.pkl")
joblib.dump(model, model_path)
print(f"\nModel successfully saved to: {model_path}")

# ==========================================
# Report
class_names = [
    "Rest",
    "Fist",
    "Flexion",
    "Extension",
    "Radial Deviation",
    "Ulnar Deviation",
]

# Confusion Matrix Image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title("sEMG Gesture Classification - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(reports_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Confusion Matrix saved to: {cm_path}")

# Text Report
report_text = classification_report(
    y_test, y_pred, target_names=class_names, digits=4
)
report_path = os.path.join(reports_dir, "classification_report.txt")

with open(report_path, "w") as f:
    f.write("sEMG Gesture Recognition - Model Evaluation\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
    f.write(report_text)

print(f"Classification Report saved to: {report_path}")