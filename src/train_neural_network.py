# =====================================
# 📦 1. Imports
# =====================================
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from imblearn.over_sampling import SMOTE

# =====================================
# 📁 2. Load and preprocess data
# =====================================
df = pd.read_csv("data/processed/cleaned_full_data.csv", sep=',')
print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
df.info()

# Drop leakage column
drop_cols = ['duration']
X = df.drop(columns=['y'] + drop_cols)
y = (df['y'] == 'yes').astype(int)

# Separate feature types
num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["int", "float"]).columns.tolist()

# Build preprocessing pipeline
pipe_num = Pipeline([('scaler', StandardScaler())])
pipe_cat = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

preproc = ColumnTransformer([
    ('num', pipe_num, num_cols),
    ('cat', pipe_cat, cat_cols)
], remainder='drop')

# Fit and transform
X_proc = preproc.fit_transform(X)

# Get column names
cat_feature_names = preproc.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
feature_names = num_cols + list(cat_feature_names)

# Optional DataFrame
df_proc = pd.DataFrame(X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc, columns=feature_names)

# =====================================
# 🔁 3. Resampling with SMOTE
# =====================================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_proc, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# =====================================
# 💾 4. Save preprocessing artifacts
# =====================================
scaler = preproc.named_transformers_['num']['scaler']
encoder = preproc.named_transformers_['cat']['encoder']

with open("models/neural_network/scaler_nn.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/neural_network/encoder_nn.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("models/neural_network/preprocessor_nn.pkl", "wb") as f:
    pickle.dump(preproc, f)

with open("models/neural_network/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# =====================================
# 🧠 5. Define, compile and train model
# =====================================
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =====================================
# 📈 6. Evaluation
# =====================================
def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model loss over epochs')
    plt.savefig('output/neural_network/loss_curve_nn.png', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_history(history)

# Metrics
y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC score:", roc_auc_score(y_test, model.predict(X_test)))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
auc_score = roc_auc_score(y_test, model.predict(X_test))

plt.figure()
plt.plot(fpr, tpr, label=f'Neural Network (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.savefig('output/neural_network/Receiver_Operating_Characteristic_nn.png', dpi=300, bbox_inches='tight')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# =====================================
# 💾 7. Save final model
# =====================================
model.save("models/neural_network/model_nn.h5")

# =====================================
# 🔍 8. SHAP Interpretation (DeepExplainer)
# =====================================
# %% shap

import shap
import numpy as np

# Select a background sample from training set
X_background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
np.bool = np.bool_
np.int = int 
# SHAP expects dense input
if hasattr(X_background, "toarray"):
    X_background = X_background.toarray()
    X_test_dense = X_test.toarray()
else:
    X_test_dense = X_test

# Create SHAP explainer (for neural networks)
explainer = shap.Explainer(model, X_background, feature_names = preproc.get_feature_names_out())

# Compute SHAP values for test set
shap_values = explainer(X_test_dense)

# Visualize feature importance
shap.plots.bar(shap_values)
plt.savefig('output/neural_network/shap_bar_nn.png', dpi=300, bbox_inches='tight')
plt.close()
# Visualize SHAP explanation for a single prediction
# Select instance
i = 0  # change this index to explore different predictions

# # Access values directly
# shap.waterfall_plot(
#     shap_values[i].base_values,        # scalar base value
#     shap_values[i].values,             # SHAP values for instance i
#     X_test_dense[i],                   # feature values of instance i
#     feature_names=preproc.get_feature_names_out()       # <- Optional but clearer
# )
# plt.savefig('models/neural_network/figures/shap_waterfall_0_nn.png', dpi=300, bbox_inches='tight')
# plt.close()

 


 