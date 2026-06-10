# app/services/ticket_classifier.py
import os
import joblib

class TicketClassifier:
    """
    Service to load saved scikit-learn models and perform
    predictions on incoming tickets.
    """
    def __init__(self):
        self.category_model = None
        self.priority_model = None
        self.sentiment_model = None
        self.loaded = False

    def load_models(self):
        """
        Loads joblib model pipelines from the models/ directory.
        """
        model_paths = {
            "category": "models/category_pipeline.joblib",
            "priority": "models/priority_pipeline.joblib",
            "sentiment": "models/sentiment_pipeline.joblib",
        }

        # Verify all models exist before loading
        for name, path in model_paths.items():
            if not os.path.exists(path):
                print(f"Warning: Model file {path} not found. Running classifier in mock mode.")
                return False

        try:
            self.category_model = joblib.load(model_paths["category"])
            self.priority_model = joblib.load(model_paths["priority"])
            self.sentiment_model = joblib.load(model_paths["sentiment"])
            self.loaded = True
            print("ML Models successfully loaded into memory!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict(self, subject: str, message: str) -> dict:
        """
        Predicts category, priority, and sentiment for a given ticket.
        """
        if not self.loaded:
            return {
                "category": "Technical issue",
                "priority": "low",
                "sentiment": "neutral",
                "confidence": 0.5,
                "analysis_method": "mock"
            }
        # Combine subject and message matching the training format
        combined_text = f"{subject.lower()} {message.lower()}"
        # Predict classes
        category_pred = self.category_model.predict([combined_text])[0]
        priority_pred = self.priority_model.predict([combined_text])[0]
        sentiment_pred = self.sentiment_model.predict([combined_text])[0]
        # Calculate prediction confidence score
        cat_probs = self.category_model.predict_proba([combined_text])[0]
        prio_probs = self.priority_model.predict_proba([combined_text])[0]
        sent_probs = self.sentiment_model.predict_proba([combined_text])[0]
        confidence = float(
            (max(cat_probs) + max(prio_probs) + max(sent_probs)) / 3.0
        )
        return {
            "category": category_pred,
            "priority": priority_pred.lower(),
            "sentiment": sentiment_pred.lower(),
            "confidence": round(confidence, 2),
            "analysis_method": "ml"
        }
# Singleton instance — import and use this across the app
classifier = TicketClassifier()