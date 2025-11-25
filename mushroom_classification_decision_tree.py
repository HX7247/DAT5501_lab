from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables)


# Decision Tree Classification Pipeline
def encode_categorical(X_df, y_df):
    """Label-encode all categorical features and target."""
    X_encoded = X_df.copy()
    # encode each feature column
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    # encode target (assumes single column; adjust if multi-column)
    if isinstance(y_df, pd.DataFrame):
        y_series = y_df.iloc[:, 0]
    else:
        y_series = y_df
    
    # Map e->edible, p->poisonous before encoding
    y_series = y_series.astype(str).str.lower()
    y_series = y_series.replace({'e': 'edible', 'p': 'poisonous'})
    
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y_series)
    return X_encoded, y_encoded, le_y

def train_and_evaluate_mushroom(X, y, test_size=0.2, max_depth=5, random_state=42):
    """
    Train DecisionTreeClassifier on mushroom dataset and print precision/recall.
    """
    # Encode categorical variables
    X_enc, y_enc, label_encoder = encode_categorical(X, y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    
    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Metrics
    class_names = label_encoder.classes_
    precision = precision_score(y_test, y_pred, average='binary' if len(class_names) == 2 else 'macro')
    recall = recall_score(y_test, y_pred, average='binary' if len(class_names) == 2 else 'macro')
    
    print("\n" + "="*60)
    print("Decision Tree Classification - Mushroom Dataset")
    print("="*60)
    print(f"Max Depth: {max_depth}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Mushroom Classification')
    plt.tight_layout()
    plt.show()
    
    # Plot decision tree (limit depth for readability)
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X_enc.columns.tolist(), 
              class_names=class_names.tolist(), 
              filled=True, max_depth=3, fontsize=10)
    plt.title(f'Decision Tree (max_depth={max_depth}, visualized depth=3)')
    plt.tight_layout()
    plt.show()
    
    return clf, (X_test, y_test, y_pred)

if __name__ == "__main__":
    model, results = train_and_evaluate_mushroom(X, y, max_depth=5)
