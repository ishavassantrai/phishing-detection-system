import pandas as pd
import re
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#loads the database
df = pd.read_csv("full_dataset.csv")
df = df[["Body", "label"]].dropna()

#maps the emails to numerical labels for training
df["label"] = df["label"].map({"phishing": 1, "non-phishing": 0})

#extracts the behavioural features withing the database
df["len_of_body"] = df["Body"].apply(len)
df["num_of_sentences"] = df["Body"].apply(lambda x: x.count(".") + x.count("!") + x.count("?"))
df["num_of_words"] = df["Body"].apply(lambda x: len(str(x).split()))
df["num_of_exclamations"] = df["Body"].apply(lambda x : x.count('!'))
df["num_of_links"] = df["Body"].apply(lambda x: len(re.findall(r'https?>://[^\s)>\"]+', str(x))))
df["avg_word_len"] = df["Body"].apply (
    lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if str(x).split() else 0
)
#prepares the feature set
X = df[["len_of_body", "num_of_sentences", "num_of_words", "num_of_exclamations", "num_of_links", "avg_word_len"]]
y = df["label"]

#normalises the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#splits the data
X_train, X_test, y_test, y_test = train_test_split(X_scaled, y , test_size=0.2, random_state=42)


#trains the isolation forest
model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
model.fit(X_train)

#predicts anomalies where -1 is anomaly and 1 is normal 
predictions = model.predict(X_test)

#converts to binary where 1 is phishing and 0 is non-phishing
predictions_binary = [0 if p ==1 else 1 for p in predictions]

#evaluates model 
print("Classification report: ")
print(classification_report(y_test, predictions_binary))

print("Confusion Model: ")
print(confusion_matrix(y_test, predictions_binary))

decision_scores = model.decision_function(X_scaled)

min_score = decision_scores.min()
max_score = decision_scores.max()

print("Min: ", min_score)
print("Max: ", max_score)


#saves the model
joblib.dump(model, "behavioural_model.pkl")
joblib.dump(scaler, "behaviour_scaler.pkl")
joblib.dump({"min": min_score, "max": max_score}, "behaviour_score_range.pkl")