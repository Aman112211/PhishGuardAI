import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pickle

def extract_email_features(df):
    """Extract meaningful features from email data"""
    features = pd.DataFrame()
    
    # Text length features
    features['subject_length'] = df['subject'].fillna('').str.len()
    features['body_length'] = df['body'].fillna('').str.len()
    
    # URL features
    features['num_urls'] = df['urls'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0)
    features['has_suspicious_url'] = df['urls'].fillna('').astype(str).str.contains('http:', case=False, regex=False).astype(int)

    
    # Suspicious patterns in subject/body
    suspicious_words = ['click', 'urgent', 'verify', 'account', 'password', 'winner', 'prize', 'free', 'claim']
    pattern = '|'.join(suspicious_words)
    features['suspicious_subject'] = df['subject'].fillna('').str.lower().str.contains(pattern, regex=True).astype(int)
    features['suspicious_body'] = df['body'].fillna('').str.lower().str.contains(pattern, regex=True).astype(int)
    
    # Capitalization features
    features['subject_caps_ratio'] = df['subject'].fillna('').apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    features['body_caps_ratio'] = df['body'].fillna('').apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    # Exclamation and question marks
    features['subject_exclamations'] = df['subject'].fillna('').str.count('!')
    features['body_exclamations'] = df['body'].fillna('').str.count('!')
    
    # Email domain features
    features['sender_valid'] = df['sender'].fillna('').str.contains('@', regex=False).astype(int)
    features['receiver_valid'] = df['receiver'].fillna('').str.contains('@', regex=False).astype(int)
    
    return features

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv("CEAS_08.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Extract features
    print("\nExtracting features...")
    feature_df = extract_email_features(df)
    
    # Add TF-IDF features for subject and body (limited features to avoid overfitting)
    print("Computing TF-IDF features...")
    tfidf_subject = TfidfVectorizer(max_features=50, stop_words='english', min_df=2)
    tfidf_body = TfidfVectorizer(max_features=100, stop_words='english', min_df=2)
    
    subject_tfidf = tfidf_subject.fit_transform(df['subject'].fillna(''))
    body_tfidf = tfidf_body.fit_transform(df['body'].fillna(''))
    
    # Combine all features
    X_numeric = feature_df.values
    X_subject = subject_tfidf.toarray()
    X_body = body_tfidf.toarray()
    X = np.hstack([X_numeric, X_subject, X_body])
    
    y = df['label'].values
    
    print(f"Final feature shape: {X.shape}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train XGBoost with regularization to prevent overfitting
    print("\nTraining model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model and vectorizers as a dictionary
    print("\nSaving model and vectorizers...")
    model_data = {
        'model': model,
        'tfidf_subject': tfidf_subject,
        'tfidf_body': tfidf_body
    }
    
    with open("email_phishing_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Training complete! Model saved as 'email_phishing_model.pkl'")
    print("\nSaved components:")
    print("  - model: XGBoost classifier")
    print("  - tfidf_subject: TF-IDF vectorizer for subject")
    print("  - tfidf_body: TF-IDF vectorizer for body")

if __name__ == '__main__':
    main()
