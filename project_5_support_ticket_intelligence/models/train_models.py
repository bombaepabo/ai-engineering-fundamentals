# train_models.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# 1. Load the dataset
CSV_PATH = "data/tickets/customer_support_tickets.csv"
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Clean fields
df['Ticket Subject'] = df['Ticket Subject'].fillna("").str.lower()
df['Ticket Description'] = df['Ticket Description'].fillna("").str.lower()

# Combine subject and description for labeling context
df['full_text'] = df['Ticket Subject'] + " " + df['Ticket Description']

# 2. Apply Rule-Based labeling to create a clean learning target
def get_smart_category(row):
    text = row['full_text']
    
    # Billing/Refund Rules
    if any(k in text for k in ['refund', 'billing', 'charge', 'invoice', 'payment', 'price', 'fee']):
        return "Refund request" if 'refund' in text else "Billing inquiry"
    
    # Cancellation Rules
    if any(k in text for k in ['cancel', 'remove', 'close account', 'delete account']):
        return "Cancellation request"
        
    # Technical issue Rules
    if any(k in text for k in ['crash', 'error', 'bug', 'setup', 'install', 'connect', 'login', 'credentials', 'password', 'hardware', 'battery', 'not turning on']):
        return "Technical issue"
        
    # Fallback default
    return "Product inquiry"

def get_smart_priority(row):
    text = row['full_text']
    
    # Critical Keywords
    if any(k in text for k in ['urgent', 'security', 'critical', 'lost all data', 'hacked', 'emergency', 'asap']):
        return "Critical"
    # High Keywords
    if any(k in text for k in ['crash', 'error', 'broken', 'fail', 'credentials', 'invalid', 'lock']):
        return "High"
    # Medium Keywords
    if any(k in text for k in ['setup', 'configure', 'install', 'update', 'compatible']):
        return "Medium"
        
    return "Low"

def get_smart_sentiment(row):
    text = row['full_text']
    
    negative_words = [
        'angry', 'frustrated', 'bad', 'terrible', 'worst', 'useless', 
        'garbage', 'disappointed', 'fail', 'error', 'invalid', 'blocked', 
        'urgent', 'not working', 'issue', 'problem', 'immediately', 'double charged'
    ]
    
    # Expanded list of positive indicators
    positive_words = [
        'thanks', 'thank you', 'love', 'great', 'awesome', 'good', 
        'perfect', 'helpful', 'satisfied', 'resolved'
    ]
    
    if any(k in text for k in negative_words):
        return "negative"
    if any(k in text for k in positive_words):
        return "positive"
        
    return "neutral"

print("Generating logical target labels...")
df['smart_category'] = df.apply(get_smart_category, axis=1)
df['smart_priority'] = df.apply(get_smart_priority, axis=1)
df['smart_sentiment'] = df.apply(get_smart_sentiment, axis=1)

# Target configurations for training
targets = {
    "category": "smart_category",
    "priority": "smart_priority",
    "sentiment": "smart_sentiment"
}

# 3. Train models for each target
# We train on the raw Ticket Description text, letting the model learn the patterns
X = df['Ticket Subject'] + " " + df['Ticket Description']

for target_name, col_name in targets.items():
    print(f"\n--- Training Model for {target_name.upper()} ---")
    y = df[col_name]
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create an ML Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # Save the trained pipeline
    model_file = f"models/{target_name}_pipeline.joblib"
    joblib.dump(pipeline, model_file)
    print(f"Saved {target_name} model to {model_file}")

print("\nAll models trained and saved successfully with logical labels!")