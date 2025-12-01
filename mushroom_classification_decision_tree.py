from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def plot_roc_curve(y_test, y_pred_proba, label_encoder):
    """Plot ROC curve for binary classification."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Mushroom Classification')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    return roc_auc

def perform_cross_validation(X_enc, y_enc, max_depth=5, cv=5, random_state=42):
    """Perform k-fold cross-validation and report metrics."""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    # Cross-validate with multiple metrics
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(clf, X_enc, y_enc, cv=cv, scoring=scoring, 
                                 return_train_score=True)
    
    print("\n" + "="*60)
    print(f"{cv}-Fold Cross-Validation Results")
    print("="*60)
    
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        print(f"\n{metric.upper()}:")
        print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")
        print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")
    
    return cv_results

def create_latent_features(X_df):
    """
    Identify observed vs latent variables and engineer new features.
    
    OBSERVED VARIABLES (directly measured):
    - All 22 categorical features in mushroom dataset (cap-shape, cap-color, etc.)
    
    LATENT VARIABLES (inferred concepts):
    - Toxicity indicators (combinations of features associated with poisonous mushrooms)
    - Environmental adaptations (habitat + population combinations)
    - Physical danger signals (odor + gill-color + spore-print-color)
    """
    X_enhanced = X_df.copy()
    
    # Latent Feature 1: "Danger Score" - combines multiple warning signs
    # High-risk odors: foul (f), spicy (y), fishy (s), creosote (c)
    if 'odor' in X_enhanced.columns:
        X_enhanced['has_warning_odor'] = X_enhanced['odor'].isin(['f', 'y', 's', 'c', 'p', 'm']).astype(int)
    
    # Latent Feature 2: "Environmental Risk" - habitat-population combination
    # Some habitat-population combos correlate with poisonous species
    if 'habitat' in X_enhanced.columns and 'population' in X_enhanced.columns:
        X_enhanced['habitat_population_risk'] = (
            (X_enhanced['habitat'] == 'l') & (X_enhanced['population'].isin(['s', 'c']))
        ).astype(int)
    
    # Latent Feature 3: "Gill Danger Pattern" - gill features often indicate toxicity
    if 'gill-color' in X_enhanced.columns and 'gill-spacing' in X_enhanced.columns:
        X_enhanced['gill_warning_pattern'] = (
            (X_enhanced['gill-color'].isin(['b', 'p', 'u', 'e', 'g'])) & 
            (X_enhanced['gill-spacing'] == 'c')
        ).astype(int)
    
    # Latent Feature 4: "Spore-Stalk Correlation" - combined visual cues
    if 'spore-print-color' in X_enhanced.columns and 'stalk-color-above-ring' in X_enhanced.columns:
        X_enhanced['spore_stalk_mismatch'] = (
            X_enhanced['spore-print-color'] != X_enhanced['stalk-color-above-ring']
        ).astype(int)
    
    # Latent Feature 5: "Cap Warning Composite" - cap characteristics
    if 'cap-color' in X_enhanced.columns and 'cap-surface' in X_enhanced.columns:
        X_enhanced['cap_danger_score'] = (
            (X_enhanced['cap-color'].isin(['w', 'y', 'g'])) & 
            (X_enhanced['cap-surface'].isin(['s', 'y']))
        ).astype(int)
    
    new_features = [col for col in X_enhanced.columns if col not in X_df.columns]
    print(f"\n{'='*60}")
    print("LATENT FEATURE ENGINEERING")
    print(f"{'='*60}")
    print(f"Created {len(new_features)} latent features:")
    for feat in new_features:
        print(f"  - {feat}")
    
    return X_enhanced

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
    y_pred_proba = clf.predict_proba(X_test)
    
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
    
    # ROC Curve
    plot_roc_curve(y_test, y_pred_proba, label_encoder)
    
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
    # 1. Baseline model
    print("\n" + "="*60)
    print("BASELINE MODEL (Original Features)")
    print("="*60)
    model, results = train_and_evaluate_mushroom(X, y, max_depth=5)
    
    # 2. Cross-validation on baseline
    X_enc_base, y_enc_base, _ = encode_categorical(X, y)
    cv_results_base = perform_cross_validation(X_enc_base, y_enc_base, max_depth=5, cv=5)
    
    # 3. Enhanced model with latent features
    print("\n" + "="*60)
    print("ENHANCED MODEL (With Latent Features)")
    print("="*60)
    X_enhanced = create_latent_features(X)
    model_enhanced, results_enhanced = train_and_evaluate_mushroom(X_enhanced, y, max_depth=5)
    
    # 4. Cross-validation on enhanced model
    X_enc_enhanced, y_enc_enhanced, _ = encode_categorical(X_enhanced, y)
    cv_results_enhanced = perform_cross_validation(X_enc_enhanced, y_enc_enhanced, max_depth=5, cv=5)
    
    # 5. Comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Baseline features: {X_enc_base.shape[1]}")
    print(f"Enhanced features: {X_enc_enhanced.shape[1]}")
    print(f"\nBaseline CV ROC-AUC: {cv_results_base['test_roc_auc'].mean():.4f}")
    print(f"Enhanced CV ROC-AUC: {cv_results_enhanced['test_roc_auc'].mean():.4f}")
    print(f"Improvement: {(cv_results_enhanced['test_roc_auc'].mean() - cv_results_base['test_roc_auc'].mean()):.4f}")
