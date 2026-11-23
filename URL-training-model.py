import pandas as pd
import re
from urllib.parse import urlparse

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler 


labels = []
urls = []

url_pattern = r'https?://[^\s)>\"]+'


def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path

    return {
        "url_length" : len(url),
        "https_presence" : int(url.startswith("https")),
        "num_dots": url.count("."),
        "num_hypens" : url.count("-"),
        "has_ip": int(bool(re.search(r'\b(?:d{1,3}\.){3}\d{1,3}\b', hostname))),#
        "has_sus_word": int(any(word in url.lower() for word in ["login", "verify", "secure", "account", "update"])),
        "path_length": len(path),
        "num_of_subdirs": path.count("/")
    }

def predict_sus_url(url):
    
    features = extract_features(url)
    df = pd.DataFrame([features])
    #calculates the probability of phishing (class 1)
    prob = model.predict_proba(df)[0][1]
    return prob



#loads the database
df = pd.read_csv("full_dataset.csv")
df = df[["Body", "label"]].dropna()

for _, row in df.iterrows():
    body_text = str(row["Body"])
    url_finds = re.findall(url_pattern, body_text)
    for url in url_finds: 
        urls.append(url)
        label_text = str(row["label"]).strip().lower()
        
        if label_text == "phishing": 
            labels.append(1)
        else:
            labels.append(0)

#creates a dataframe of urls and labels
url_df = pd.DataFrame({
    "url": urls,
    "label" : labels
})

#removes any found duplicates
urls_df = url_df.drop_duplicates().reset_index(drop=True)

#extracts all any urls with features
features = url_df["url"].apply(extract_features)
features_df = pd.DataFrame(features.tolist())
features_df["label"] = url_df["label"].values

#creating a feature matrix and target 
X = features_df.drop("label", axis=1)
y = features_df["label"]

#scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#evalution 
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "url_classifier.pkl")
joblib.dump(scaler, "url_scaler.pkl")