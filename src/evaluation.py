from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import X_val, y_val

model = load_model('iceberg_cnn.h5')
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)