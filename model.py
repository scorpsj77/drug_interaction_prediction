"""Load datasets"""

import pandas as pd

# Load the positive and negative drug interactions datasets
neg = 'https://media.githubusercontent.com/media/TIML-Group/HODDI/refs/heads/main/dataset/HODDI/Merged_Dataset/neg.csv'
pos = 'https://media.githubusercontent.com/media/TIML-Group/HODDI/refs/heads/main/dataset/HODDI/Merged_Dataset/pos.csv'

dfn = pd.read_csv(neg)
dfp = pd.read_csv(pos)

# Load the dictionary of drugs
dictionary = 'https://media.githubusercontent.com/media/TIML-Group/HODDI/refs/heads/main/dataset/dictionary/Drugbank_ID_SMILE_all_structure%20links.csv'

df_dict = pd.read_csv(dictionary)

# Drop unnecessary columns of the dataset
dfn.drop(['time', 'row_index', 'SE_above_0.9'], axis=1, inplace=True)
dfp.drop(['time', 'row_index', 'SE_above_0.9'], axis=1, inplace=True)

# Change the target label for negative drug interactions to 0 to
dfn.replace({'hyperedge_label': {-1: 0}}, inplace=True)

"""# Prepare the Data

Splitting the dataset
"""

from sklearn.model_selection import train_test_split

# Split: 80% train, 20% test
# Split positives
dfp_train, dfp_test = train_test_split(dfp, test_size=0.20, random_state=42)
# Split negatives
dfn_train, dfn_test = train_test_split(dfn, test_size=0.20, random_state=42)

# Combine data for training and testing
train_combined = pd.concat([dfp_train, dfn_train]).sample(frac=1, random_state=42)
test_combined = pd.concat([dfp_test, dfn_test]).sample(frac=1, random_state=42)

"""Converting all DrugBank IDs to a list"""

import ast

def convert_string_to_list(s):
    if isinstance(s, str):
        evaluated = ast.literal_eval(s)
        if isinstance(evaluated, list):
            return evaluated
        else:
            return [str(evaluated)]
    elif isinstance(s, list):
        return s

# Apply the conversion
train_combined['DrugBankID'] = train_combined['DrugBankID'].apply(convert_string_to_list)
test_combined['DrugBankID'] = test_combined['DrugBankID'].apply(convert_string_to_list)

"""Drop the unknown classes in testing"""

# List of unknown classes provided when running previous models
unknown_ids = ['DB03862', 'DB04482', 'DB04920', 'DB11050', 'DB12366', 'DB13151', 'DB14693', 'DB15270', 'DB18046']

def has_unknown(drug_ids):
    return any(d in unknown_ids for d in drug_ids)

train_combined = train_combined[~train_combined['DrugBankID'].apply(has_unknown)].reset_index(drop=True)
test_combined = test_combined[~test_combined['DrugBankID'].apply(has_unknown)].reset_index(drop=True)

print(train_combined['DrugBankID'].explode().isin(unknown_ids).any())
print(test_combined['DrugBankID'].explode().isin(unknown_ids).any())
test_combined.shape

"""# ML

Load necessary packages to set up the model
"""

!pip install catboost

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Store true labels
Y_true = test_combined['hyperedge_label'].values

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

"""Train and test the stack ensemble model of Logistic Regression, Random Forest, and CatBoost"""

# Create and fit the MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

# Transform training data
X_train = mlb.fit_transform(train_combined['DrugBankID']).astype(float)
Y_train = train_combined['hyperedge_label']

estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('cb', CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_seed=42))
]

model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),  # meta-model
    passthrough=True,  # pass original features to final_estimator
    cv=5  # cross-validation folds to train base models
)

model.fit(X_train, Y_train)

# Transform test data and make predictions
X_test = mlb.transform(test_combined['DrugBankID']).astype(float)
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)[:, 1]  # confidence score for label = 1

# Add predictions to test set
test_combined['predicted_label'] = Y_pred
test_combined['confidence_score'] = Y_prob

# Output results
results = test_combined[['report_id', 'DrugBankID', 'predicted_label', 'confidence_score']]
results.to_csv('results.csv', index=False)
results.tail()

print(classification_report(Y_true, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_true, Y_pred))
print("ROC AUC:", roc_auc_score(Y_true, Y_prob))

"""Check if the model is overfitting by comparing with training set accuracy"""

# Predict on training set
Y_train_pred = model.predict(X_train)

# Calculate accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_accuracy = accuracy_score(Y_train, Y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# More detailed metrics
print(classification_report(Y_train, Y_train_pred))
print("Training Confusion Matrix:\n", confusion_matrix(Y_train, Y_train_pred))
