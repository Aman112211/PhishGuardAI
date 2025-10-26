import arff
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load ARFF file and convert to DataFrame
def load_arff(filename):
    data = arff.load(open(filename))
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    return df

# Custom helper functions for feature extraction (if needed)
def check_if_shortener(url):
    shortening_services = [
        "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
        "buff.ly", "adf.ly", "bit.do", "mcaf.ee", "rebrand.ly", "trib.al"
    ]
    for shortener in shortening_services:
        if shortener in url:
            return 1
    return 0

def has_port(url):
    try:
        parsed_url = urlparse(url)
        port = parsed_url.port
        if port and port not in [80, 443]:
            return 1
        else:
            return 0
    except Exception:
        return 0

def detect_abnormal_patterns(url):
    suspicious_keywords = [
        "secure", "account", "update", "free", "bonus", "login", "verification", "signin", "banking"
    ]
    try:
        if ".." in url:
            return 1
        if len(url) > 75:
            return 1
        if url.count("@") > 1:
            return 1
        if re.search(r"%[0-9a-fA-F]{2}", url):
            return 1
        if any(keyword in url.lower() for keyword in suspicious_keywords):
            return 1
        return 0
    except Exception:
        return 0

def count_subdomains(url):
    try:
        domain = url.split('//')[-1].split('/')[0]
        return max(domain.count('.') - 1, 0)
    except Exception:
        return 0

def extract_features_for_model(url):
    features = {
        "having_IP_Address": ...,  # 1 or 0
        "URL_Length": ...,         # int
        "Shortining_Service": ..., # 1 or 0
        # ... all other features ...
        "Statistical_report": ...
    }
    import pandas as pd
    return pd.DataFrame([features], columns=[
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token",
        "Request_URL",
        "URL_of_Anchor",
        "Links_in_tags",
        "SFH",
        "Submitting_to_email",
        "Abnormal_URL",
        "Redirect",
        "on_mouseover",
        "RightClick",
        "popUpWidnow",
        "Iframe",
        "age_of_domain",
        "DNSRecord",
        "web_traffic",
        "Page_Rank",
        "Google_Index",
        "Links_pointing_to_page",
        "Statistical_report"
    ])


def main():
    # Load dataset (adjust filename as needed)
    df = load_arff('Training Dataset.arff')

    # If your dataset does not have actual 'url' column, you can add dummy URLs or extract features from other columns
    # For UCI Phishing Dataset typically URLs are not present, features are pre-extracted
    # So use the existing columns for features and 'result' as target

    # List all features excluding the label 'result' (adjust as per your dataset columns)
    FEATURES = [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token",
        "Request_URL",
        "URL_of_Anchor",
        "Links_in_tags",
        "SFH",
        "Submitting_to_email",
        "Abnormal_URL",
        "Redirect",
        "on_mouseover",
        "RightClick",
        "popUpWidnow",
        "Iframe",
        "age_of_domain",
        "DNSRecord",
        "web_traffic",
        "Page_Rank",
        "Google_Index",
        "Links_pointing_to_page",
        "Statistical_report"
    ]


    # Check if all features are present in the dataset columns; if not, fill missing ones with zero
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0  # default to 0

    # Prepare features and target
# Prepare features and target
    X = df[FEATURES]
    y = df['Result'].replace({'-1': 0, '1': 1}).astype(int)

    # Convert feature columns to numeric (int), replace errors with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)




    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    with open('phishing_url_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as phishing_url_model.pkl")

if __name__ == "__main__":
    main()
