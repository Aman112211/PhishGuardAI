from flask import Flask, render_template, request, jsonify
import os
import re
from werkzeug.utils import secure_filename
import hashlib

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# URL Phishing Detection Logic
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
    """Simple rule-based URL phishing detection"""
    features = extract_url_features(url)
    risk_score = 0
    warnings = []

    # Check each feature and calculate risk
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
        'risk_score': min(risk_score, 100),
        'classification': classification,
        'color': color,
        'warnings': warnings,
        'features': features
    }

# Email/Message Phishing Detection Logic
def analyze_email(subject, sender, content):
    """Simple rule-based email phishing detection"""
    risk_score = 0
    warnings = []

    # Combine all text for analysis
    all_text = f"{subject} {sender} {content}".lower()

    # Suspicious keywords
    urgent_words = ['urgent', 'immediate', 'action required', 'verify', 'suspended', 'locked', 'expires']
    financial_words = ['bank', 'credit card', 'payment', 'account', 'invoice', 'refund', 'transfer']
    threat_words = ['warning', 'alert', 'security', 'unauthorized', 'suspicious activity']
    prize_words = ['winner', 'congratulations', 'prize', 'lottery', 'claim', 'reward']

    # Check for urgent language
    urgent_count = sum(1 for word in urgent_words if word in all_text)
    if urgent_count >= 2:
        risk_score += 20
        warnings.append("Contains multiple urgency indicators")
    elif urgent_count >= 1:
        risk_score += 10
        warnings.append("Contains urgency language")

    # Check for financial terms
    financial_count = sum(1 for word in financial_words if word in all_text)
    if financial_count >= 2:
        risk_score += 15
        warnings.append("Multiple financial terms detected")

    # Check for threats
    threat_count = sum(1 for word in threat_words if word in all_text)
    if threat_count >= 2:
        risk_score += 15
        warnings.append("Contains threatening language")

    # Check for prize/lottery scams
    prize_count = sum(1 for word in prize_words if word in all_text)
    if prize_count >= 1:
        risk_score += 20
        warnings.append("Contains prize/lottery language (common scam)")

    # Check for suspicious links
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, content)
    if len(urls) > 3:
        risk_score += 15
        warnings.append("Contains multiple links")

    # Check for IP addresses in links
    for url in urls:
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            risk_score += 20
            warnings.append("Link contains IP address")
            break

    # Check sender email
    if sender:
        # Check for suspicious sender patterns
        if re.search(r'\d{4,}', sender):
            risk_score += 10
            warnings.append("Sender email contains many numbers")

        # Check for misspelled domains
        common_domains = ['gmail', 'yahoo', 'outlook', 'hotmail']
        misspelled = ['gmai1', 'yah00', 'out1ook', 'hotmai1', 'gmial', 'yahooo']

        for domain in misspelled:
            if domain in sender.lower():
                risk_score += 25
                warnings.append("Sender email appears to mimic trusted domain")
                break

    # Check for generic greetings
    if re.search(r'\b(dear (customer|user|member|sir|madam))\b', all_text):
        risk_score += 10
        warnings.append("Uses generic greeting (not personalized)")

    # Check for spelling/grammar (simple check)
    if content.count('!!') > 0 or content.count('???') > 0:
        risk_score += 5
        warnings.append("Excessive punctuation detected")

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
        'risk_score': min(risk_score, 100),
        'classification': classification,
        'color': color,
        'warnings': warnings,
        'url_count': len(urls)
    }

# Audio Call Fraud Detection Logic
def analyze_audio(filename):
    """Simulate audio fraud detection analysis"""
    # In a real implementation, this would use ML models to analyze audio features
    # For demo purposes, we'll simulate analysis based on filename and random factors

    import random
    import time

    # Simulate processing time
    time.sleep(1)

    # Simulate analysis (in production, you'd use actual ML models)
    # This would analyze: voice patterns, background noise, speech rate, emotional tone, etc.

    risk_score = random.randint(0, 100)
    warnings = []

    # Simulate various detection factors
    detection_factors = [
        ("Voice Pattern Analysis", random.choice(["Natural", "Suspicious", "Synthetic Detected"])),
        ("Background Noise", random.choice(["Normal", "Unusual", "Synthetic Environment"])),
        ("Speech Rate", random.choice(["Normal", "Too Fast", "Too Slow", "Irregular"])),
        ("Emotional Tone", random.choice(["Natural", "Scripted", "Pressure Tactics"])),
        ("Audio Quality", random.choice(["Normal", "Low Quality", "Digitally Modified"]))
    ]

    # Adjust based on factors
    for factor, result in detection_factors:
        if result in ["Suspicious", "Synthetic Detected", "Unusual", "Too Fast", "Too Slow", 
                      "Scripted", "Pressure Tactics", "Low Quality", "Digitally Modified"]:
            warnings.append(f"{factor}: {result}")

    # Determine classification
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

# API Endpoints
@app.route('/api/analyze-url', methods=['POST'])
def api_analyze_url():
    data = request.get_json()
    url = data.get('url', '')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    result = analyze_url(url)
    return jsonify(result)

@app.route('/api/analyze-email', methods=['POST'])
def api_analyze_email():
    data = request.get_json()
    subject = data.get('subject', '')
    sender = data.get('sender', '')
    content = data.get('content', '')

    result = analyze_email(subject, sender, content)
    return jsonify(result)

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
