from flask import Flask, render_template, request, jsonify
import os
import re
from werkzeug.utils import secure_filename
#import hashlib
from pyzbar.pyzbar import decode
from PIL import Image
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}

# Load models
url_model = pickle.load(open('phishing_url_model.pkl', 'rb'))

# Load email model and vectorizers
with open('email_phishing_model.pkl', 'rb') as f:
    email_model_data = pickle.load(f)
    email_model = email_model_data['model']
    tfidf_subject = email_model_data['tfidf_subject']
    tfidf_body = email_model_data['tfidf_body']


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
    """
    Returns 1 if URL contains suspicious patterns often used in phishing URLs, else 0.
    """
    suspicious_keywords = [
        "secure", "account", "update", "free", "bonus", "login", "verification", "signin", "banking"
    ]
    try:
        # Check for multiple consecutive dots
        if ".." in url:
            return 1

        # Check length
        if len(url) > 75:
            return 1

        # Check for multiple '@' symbols
        if url.count("@") > 1:
            return 1

        # Check for percent encoded suspicious characters
        if re.search(r"%[0-9a-fA-F]{2}", url):
            return 1

        # Check presence of suspicious keywords
        if any(keyword in url.lower() for keyword in suspicious_keywords):
            return 1

        return 0
    except Exception:
        return 0


def check_if_shortener(url):
    shortening_services = [
        "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "buff.ly",
        "adf.ly", "bit.do", "mcaf.ee", "rebrand.ly", "trib.al"
    ]
    for shortener in shortening_services:
        if shortener in url:
            return True
    return False


def count_subdomains(url):
    domain = url.split('//')[-1].split('/')[0]
    return domain.count('.') - 1


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# URL Feature Extraction
def extract_url_features(url):
    """Extract features from URL for phishing detection"""
    features = {}

    # Length of URL
    features['url_length'] = len(url)
    features['is_long'] = len(url) >= 54

    # Has IP address
    ip_pattern = r'(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])'
    features['has_ip'] = bool(re.search(ip_pattern, url))

    # Has @ symbol
    features['has_at'] = '@' in url

    # Number of dots
    features['dot_count'] = url.count('.')

    # Has double slash redirecting
    features['has_double_slash'] = url.count('//') > 1

    # Number of subdomains
    if 'http' in url:
        domain_part = url.split('//')[-1].split('/')[0]
        features['subdomain_count'] = domain_part.count('.') - 1
    else:
        features['subdomain_count'] = url.count('.') - 1

    # Has prefix/suffix with dash
    features['has_dash_in_domain'] = '-' in url.split('//')[-1].split('/')[0] if 'http' in url else '-' in url.split('/')[0]

    # HTTPS in domain part
    if 'http' in url:
        domain_part = url.split('//')[-1].split('/')[0]
        features['https_in_domain'] = 'https' in domain_part
    else:
        features['https_in_domain'] = False

    return features


def analyze_url(url):
    """Rule-based URL phishing detection"""
    features = extract_url_features(url)
    risk_score = 0
    warnings = []

    if features['is_long']:
        risk_score += 15
        warnings.append("URL is unusually long")

    if features['has_ip']:
        risk_score += 25
        warnings.append("URL contains IP address instead of domain name")

    if features['has_at']:
        risk_score += 20
        warnings.append("URL contains @ symbol (potential obfuscation)")

    if features['dot_count'] > 4:
        risk_score += 10
        warnings.append("URL has many dots (suspicious structure)")

    if features['has_double_slash']:
        risk_score += 15
        warnings.append("URL has multiple // redirections")

    if features['subdomain_count'] > 2:
        risk_score += 15
        warnings.append("URL has many subdomains")

    if features['has_dash_in_domain']:
        risk_score += 10
        warnings.append("Domain contains dashes (potential brand impersonation)")

    if features['https_in_domain']:
        risk_score += 20
        warnings.append("HTTPS token in domain part (deceptive)")

    # Determine classification
    if risk_score >= 50:
        classification = "High Risk - Likely Phishing"
        color = "danger"
    elif risk_score >= 30:
        classification = "Medium Risk - Suspicious"
        color = "warning"
    else:
        classification = "Low Risk - Appears Safe"
        color = "success"

    return {
        'url': url,
        'risk_score': "",
        'classification': classification,
        'color': color,
        'warnings': warnings,
        'features': features
    }


def extract_features_for_model(url):
    features = {
        "having_IP_Address": int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        "URL_Length": len(url),
        "Shortining_Service": int(check_if_shortener(url)),
        "having_At_Symbol": int('@' in url),
        "double_slash_redirecting": int(url.count('//') > 1),
        "Prefix_Suffix": int('-' in url.split('//')[-1].split('/')[0]),
        "having_Sub_Domain": count_subdomains(url),
        "SSLfinal_State": int(url.startswith('https://')),
        "Domain_registeration_length": 0,
        "Favicon": 0,
        "port": has_port(url),
        "HTTPS_token": int('https' in url.split('//')[-1].split('/')[0]),
        "Request_URL": 0,
        "URL_of_Anchor": 0,
        "Links_in_tags": 0,
        "SFH": 0,
        "Submitting_to_email": 0,
        "Abnormal_URL": detect_abnormal_patterns(url),
        "Redirect": 0,
        "on_mouseover": 0,
        "RightClick": 0,
        "popUpWidnow": 0,
        "Iframe": 0,
        "age_of_domain": 0,
        "DNSRecord": 0,
        "web_traffic": 0,
        "Page_Rank": 0,
        "Google_Index": 0,
        "Links_pointing_to_page": 0,
        "Statistical_report": 0
    }
    return pd.DataFrame([features])


def extract_qr_data(img_path):
    decoded_objs = decode(Image.open(img_path))
    if decoded_objs:
        return decoded_objs[0].data.decode('utf-8')
    return None


# Email Feature Extraction for ML Model
def extract_email_features_for_model(subject, sender, body):
    """Extract features from email data matching the training format"""
    # Create a temporary dataframe for feature extraction
    df = pd.DataFrame({
        'subject': [subject],
        'sender': [sender],
        'body': [body],
        'urls': ['']  # You can extract URLs from body if needed
    })
    
    features = pd.DataFrame()
    
    # Text length features
    features['subject_length'] = df['subject'].fillna('').str.len()
    features['body_length'] = df['body'].fillna('').str.len()
    
    # URL features
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_in_body = re.findall(url_pattern, body)
    features['num_urls'] = len(urls_in_body)
    features['has_suspicious_url'] = int(any('http:' in url for url in urls_in_body))
    
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
    features['receiver_valid'] = 1  # Assume receiver is always valid
    
    # TF-IDF features
    subject_tfidf = tfidf_subject.transform(df['subject'].fillna(''))
    body_tfidf = tfidf_body.transform(df['body'].fillna(''))
    
    # Combine all features
    X_numeric = features.values
    X_subject = subject_tfidf.toarray()
    X_body = body_tfidf.toarray()
    X = np.hstack([X_numeric, X_subject, X_body])
    
    return X


# Audio Call Fraud Detection Logic
def analyze_audio(filename):
    """Simulate audio fraud detection analysis"""
    import random
    import time

    time.sleep(1)

    risk_score = random.randint(0, 100)
    warnings = []

    detection_factors = [
        ("Voice Pattern Analysis", random.choice(["Natural", "Suspicious", "Synthetic Detected"])),
        ("Background Noise", random.choice(["Normal", "Unusual", "Synthetic Environment"])),
        ("Speech Rate", random.choice(["Normal", "Too Fast", "Too Slow", "Irregular"])),
        ("Emotional Tone", random.choice(["Natural", "Scripted", "Pressure Tactics"])),
        ("Audio Quality", random.choice(["Normal", "Low Quality", "Digitally Modified"]))
    ]

    for factor, result in detection_factors:
        if result in ["Suspicious", "Synthetic Detected", "Unusual", "Too Fast", "Too Slow", 
                      "Scripted", "Pressure Tactics", "Low Quality", "Digitally Modified"]:
            warnings.append(f"{factor}: {result}")

    if risk_score >= 70:
        classification = "High Risk - Likely Fraudulent"
        color = "danger"
    elif risk_score >= 40:
        classification = "Medium Risk - Suspicious"
        color = "warning"
    else:
        classification = "Low Risk - Appears Legitimate"
        color = "success"

    return {
        'filename': filename,
        'risk_score': risk_score,
        'classification': classification,
        'color': color,
        'warnings': warnings,
        'detection_factors': detection_factors
    }


# Routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/url-detection')
def url_detection():
    return render_template('url_detection.html')


@app.route('/email-detection')
def email_detection():
    return render_template('email_detection.html')


@app.route('/audio-detection')
def audio_detection():
    return render_template('audio_detection.html')


@app.route('/qr-detection')
def qr_detection():
    return render_template('qr_detection.html')


# API Endpoints
@app.route('/api/analyze-url', methods=['POST'])
def api_analyze_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        url = data['url']

        # Run rule-based analysis
        result = analyze_url(url)

        # Get ML predictions
        features = extract_features_for_model(url)
        pred_prob = url_model.predict_proba(features)[:,1][0]
        pred_class = url_model.predict(features)[0]

        # Add ML info to result
        result['ml_probability'] = float(pred_prob)
        result['ml_classification'] = 'Phishing' if pred_class == 1 else 'Legitimate'
        result["risk_score"] = f'{pred_prob*100:.2f}'

        # Override classification if ML detects phishing strongly
        if pred_class == 1:
            result['classification'] = f"High Risk - ML Model Detected Phishing probability-{pred_prob*100:.2f}%"
            result['color'] = "danger"
            result['warnings'].append("ML model detected phishing likelihood")

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-qr', methods=['POST'])
def api_analyze_qr():
    if 'qr_image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['qr_image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    qr_content = extract_qr_data(filepath)
    if not qr_content:
        os.remove(filepath)
        return jsonify({'error': 'No QR data found'}), 400

    result = analyze_url(qr_content)
    os.remove(filepath)
    result['qr_content'] = qr_content

    features = extract_features_for_model(qr_content)
    pred_prob = url_model.predict_proba(features)[:,1][0]
    pred_class = url_model.predict(features)[0]

    if 'warnings' not in result:
        result['warnings'] = []
    
    if pred_class == 1:
        result['classification'] = f"High Risk - ML Model Detected Phishing Probability-{pred_prob*100:.2f}%"
        result['color'] = "danger"
        result['warnings'].append("ML model detected phishing likelihood")

    result['ml_probability'] = float(pred_prob)
    result['ml_classification'] = 'Phishing' if pred_class == 1 else 'Legitimate'
    result['risk_score'] = f'{pred_prob*100:.2f}'

    return jsonify(result)


@app.route('/api/analyze-email', methods=['POST'])
def api_analyze_email():
    data = request.get_json()
    subject = data.get('subject', '')
    sender = data.get('sender', '')
    content = data.get('content', '')

    if not subject and not sender and not content:
        return jsonify({'error': 'At least one field is required'}), 400

    try:
        # Extract features for ML model
        features = extract_email_features_for_model(subject, sender, content)
        
        # Get ML predictions
        pred_prob = email_model.predict_proba(features)[:,1][0]
        pred_class = email_model.predict(features)[0]
        
        # Determine risk level
        if pred_prob >= 0.7:
            classification = "High Risk - Likely Phishing"
            color = "danger"
        elif pred_prob >= 0.4:
            classification = "Medium Risk - Suspicious"
            color = "warning"
        else:
            classification = "Low Risk - Appears Safe"
            color = "success"
        
        # Generate warnings based on content analysis
        warnings = []
        all_text = f"{subject} {content}".lower()
        
        if any(word in all_text for word in ['urgent', 'immediate', 'action required']):
            warnings.append("Contains urgency indicators")
        
        if any(word in all_text for word in ['verify', 'account', 'password', 'suspended']):
            warnings.append("Contains account security keywords")
        
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, content)
        if len(urls) > 2:
            warnings.append(f"Contains {len(urls)} links")
        
        if pred_class == 1:
            warnings.append("ML model detected phishing likelihood")
        
        result = {
            'risk_score': f'{pred_prob*100:.2f}',
            'classification': classification,
            'color': color,
            'warnings': warnings,
            'ml_probability': float(pred_prob),
            'ml_classification': 'Phishing' if pred_class == 1 else 'Legitimate',
            'url_count': len(urls)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-audio', methods=['POST'])
def api_analyze_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({'error': 'Invalid file type. Allowed: mp3, wav, ogg, m4a'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = analyze_audio(filename)

    # Clean up uploaded file
    try:
        os.remove(filepath)
    except:
        pass

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
