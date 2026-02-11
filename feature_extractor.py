import re
import urllib.parse

SUSPICIOUS_WORDS = [
    'login', 'secure', 'bank', 'verify', 'update',
    'free', 'bonus', 'confirm', 'account', 'signin'
]

def extract_features(url):
    parsed = urllib.parse.urlparse(url)

    features = {}

    features['url_length'] = len(url)
    features['dot_count'] = url.count('.')
    features['hyphen_count'] = url.count('-')
    features['at_count'] = url.count('@')
    features['question_count'] = url.count('?')
    features['percent_count'] = url.count('%')
    features['digit_count'] = sum(char.isdigit() for char in url)
    features['special_char_count'] = len(re.findall(r'[^\w]', url))
    features['has_ip'] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)))
    features['https'] = int(parsed.scheme == 'https')
    features['short_url'] = int(any(x in url for x in ['bit.ly', 'tinyurl', 'goo.gl']))
    features['suspicious_word_count'] = sum(word in url.lower() for word in SUSPICIOUS_WORDS)
    features['subdomain_count'] = parsed.netloc.count('.') - 1

    return features
