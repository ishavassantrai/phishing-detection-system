import re
import torch
import joblib
import pandas as pd
from urllib.parse import urlparse
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import gradio as gr

# loads the presaved models
text_model_path = "roberta-model"
tokenizer = RobertaTokenizerFast.from_pretrained(text_model_path)
text_model = RobertaForSequenceClassification.from_pretrained(text_model_path)
 
url_model = joblib.load("url_classifier.pkl")
url_scaler = joblib.load("url_scaler.pkl")

behaviour_model = joblib.load("behavioural_model.pkl")
behaviour_scaler = joblib.load("behaviour_scaler.pkl")

#gets the min and max values from training model to calculate probability 
behaviour_score_range = joblib.load("behaviour_score_range.pkl")
min_beh_val = behaviour_score_range["min"]
max_beh_val = behaviour_score_range["max"]

# regular expression url pattern used to find urls 
url_pattern = r'https?://[^\s)>\"]+'
btc_pattern = r'\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[0-9a-zA-Z]{39,59})\b'


#extract features for urls
def url_ext(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path
    return {
        "url_length": len(url),
        "https_presence": int(url.startswith("https")),
        "num_dots": url.count("."),
        "num_hypens": url.count("-"),
        "has_ip": int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', hostname))),
        "has_sus_word": int(any(word in url.lower() for word in ["login", "verify", "secure", "account", "update"])),
        "path_length": len(path),
        "num_of_subdirs": path.count("/")
    }


#feature extraction for behaviour model analysis 
def behaviour_ext(text):
    num_words = len(str(text).split())
    return pd.DataFrame([{
        "len_of_body": len(text),
        "num_of_sentences": text.count(".") + text.count("!") + text.count("?"),
        "num_of_words": num_words,
        "num_of_exclamations": text.count("!"),
        "num_of_links": len(re.findall(url_pattern, text)),
        "avg_word_len": sum(len(word) for word in str(text).split()) / num_words if num_words else 0
    }])

#NLP detection analysis 
def text_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probability = torch.nn.functional.softmax(outputs.logits, dim=1)
        risk_score = probability[0][1].item()
    return round(risk_score, 2)

#returns the avg risk score for the urls and returns the highest risk score found from all the URLs
def url_prediction(urls):
    if not urls: 
        return 0.0
    
    url_features = [url_ext(url) for url in urls]
    df = pd.DataFrame(url_features)
    scaled = url_scaler.transform(df)
    
    probs = url_model.predict_proba(scaled)[:, 1]

    avg_risk = float(probs.max())
    
    return avg_risk

#Uses trained model and calculates the risk score based on feature and structure anomaly
def behaviour_prediction(text):
    features_df = behaviour_ext(text)
    scaled = behaviour_scaler.transform(features_df)
    score = behaviour_model.decision_function(scaled)[0]
    normalised_score = (score - min_beh_val) / (max_beh_val - min_beh_val + 1e-6)
    risk_score = 1-normalised_score

    return round(float(risk_score), 2) 

#returns 1 if there is bitcoin wallet pattern identified
def detect_btc_risk(text):
    matches = re.findall(btc_pattern, text)
    if matches != []:
        return 1.0
    else: 
        return 0.0


    

#analysis and displays findings of inputted message
def message_analysis(email_text):
    #stores analysis in variables
    text_risk_score = text_analysis(email_text)
    urls_found = re.findall(url_pattern, email_text) #sends to find all urls found in the msg
    url_risk_score = url_prediction(urls_found)
    
    behaviour_risk_score = behaviour_prediction(email_text)

    #if url_risk_score is 0 (theres nothing present) then it will check to see if btc is present 
    #if not then it will calc the risk score based on the two other models
    #if there is then it will calc risk score with btc instead
    if url_risk_score == 0.0:
        btc_risk_score = detect_btc_risk(email_text)
        if btc_risk_score == 0.0:
            print("im here")
            final_risk_score = (text_risk_score + behaviour_risk_score) / 2
            print(final_risk_score)
        else:
            final_risk_score = (text_risk_score + btc_risk_score + behaviour_risk_score) / 3
    
    else:
        final_risk_score = (text_risk_score + url_risk_score + behaviour_risk_score) / 3

    print(final_risk_score)
    #classification of phishing
    if final_risk_score >= 0.7: 
        classification = "PHISHING"
    elif final_risk_score >= 0.4:
        classification = "LIKLEY PHISHING"
    else: 
        classification = "NOT PHISHING"


    #displays the analysis results
    report = "**Email Phishing Analysis Report**\n"
    report += f"\n**Text Risk Score:** `{(text_risk_score *100)}%`\n"

    if urls_found:
        report += f"\n**URL Risk Score:** `{(url_risk_score*100)}%`\n"
        report+= "\n**URls Detected:**\n"
        for i, url in enumerate(urls_found, 1):
            report += f" - {url}\n"
    elif btc_risk_score == 1.0:
        report += f"\n**BTC Risk Score:** `{(btc_risk_score*100)}%`\n"
    else: 
        report += "\n**No URLs or Bitcoin Wallet detected**\n"
    
    report += f"\n**Behavioural Risk Score:** `{(behaviour_risk_score*100)}%`\n"
    report += f"\n**OVERALL CONFIDENCE SCORE:** `{round(final_risk_score*100,2)}%` ==> `{classification}`\n"

    
    
    return report

# the UI interface
gr.Interface(
    fn=message_analysis,
    inputs=gr.Textbox(lines=15, label="Paste the suspected email message below"),
    outputs=gr.Markdown(label="Phishing Detection Result"),
    title="Phishing Email Message Detection System",
    description="Detects phishing emails messages based on content, URLs, and message behavior.",
    allow_flagging="never"
).launch()
