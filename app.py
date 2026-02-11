import streamlit as st
import pandas as pd
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from feature_extractor import extract_features


@st.cache_resource
def load_url_model():
    model = joblib.load("models/phishradar_model.joblib")
    columns = joblib.load("models/feature_columns.joblib")
    return model, columns


@st.cache_resource
def load_email_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "models/phishradar_email_distilbert"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/phishradar_email_distilbert"
    )
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_sms_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "models/phishradar_text_distilbert"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/phishradar_text_distilbert"
    )
    model.eval()
    return tokenizer, model


url_model, url_feature_columns = load_url_model()
email_tokenizer, email_model = load_email_model()
sms_tokenizer, sms_model = load_sms_model()

st.set_page_config(
    page_title="PhishRadar",
    page_icon="🛡️",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center;'>🛡️ PhishRadar</h1>"
    "<h4 style='text-align:center;'>Real-Time Phishing Detection System</h4>",
    unsafe_allow_html=True
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔗 URL Detection",
    "📧 Email Detection",
    "💬 SMS / Text Detection"
])

with tab1:
    st.subheader("🔗 URL Phishing Detection")
    url = st.text_input("Enter a URL", placeholder="https://example.com")

    if st.button("Analyze URL", key="url_btn"):
        if not url.strip():
            st.warning("Please enter a valid URL")
        else:
            features = extract_features(url)
            df = pd.DataFrame([features])[url_feature_columns]

            prob = url_model.predict_proba(df)[0][1]
            st.progress(min(int(prob * 100), 100))

            if prob > 0.7:
                st.error(f"🚨 Phishing URL Detected\n\nRisk Score: {prob:.2f}")
            elif prob > 0.4:
                st.warning(f"⚠️ Suspicious URL\n\nRisk Score: {prob:.2f}")
            else:
                st.success(f"✅ Safe URL\n\nRisk Score: {prob:.2f}")

with tab2:
    st.subheader("📧 Email Phishing Detection")
    email_text = st.text_area("Paste email content")

    if st.button("Detect Email", key="email_btn"):
        if not email_text.strip():
            st.warning("Please enter email content")
        else:
            inputs = email_tokenizer(
                email_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = email_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                score = probs[0][1].item()

            st.progress(int(score * 100))

            if score > 0.7:
                st.error("🚨 Phishing Email Detected")
            elif score > 0.4:
                st.warning("⚠️ Suspicious Email")
            else:
                st.success("✅ Safe Email")

with tab3:
    st.subheader("💬 SMS / Text Phishing Detection")
    sms_text = st.text_area("Paste SMS / message text")

    if st.button("Detect Message", key="sms_btn"):
        if not sms_text.strip():
            st.warning("Please enter message content")
        else:
            inputs = sms_tokenizer(
                sms_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = sms_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                score = probs[0][1].item()

            st.progress(int(score * 100))

            if score > 0.7:
                st.error("🚨 Phishing Message Detected")
            elif score > 0.4:
                st.warning("⚠️ Suspicious Message")
            else:
                st.success("✅ Safe Message")


st.divider()
st.caption("🛡️ PhishRadar • Transformer + Machine Learning Based Cybersecurity")
st.caption("Developed by Siva Kasi Raja • © 2026")
