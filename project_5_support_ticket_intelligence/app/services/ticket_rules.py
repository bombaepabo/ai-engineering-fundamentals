# app/services/ticket_rules.py

class RuleClassifier:
    """
    Keyword-based classifier to serve as a baseline, fallback,
    and auditing tool alongside the machine learning models.
    """
    def predict(self, subject: str, message: str) -> dict:
        text = f"{subject} {message}".lower()
        
        # --- Score categories based on keyword counts ---
        cat_scores = {
            "Refund request": sum(1 for k in ["refund", "money back", "return item"] if k in text),
            "Billing inquiry": sum(1 for k in ["billing", "charge", "invoice", "payment", "price", "fee"] if k in text),
            "Cancellation request": sum(1 for k in ["cancel", "remove", "close account", "delete account"] if k in text),
            "Technical issue": sum(1 for k in ["crash", "error", "bug", "install", "login", "password", "hardware", "battery"] if k in text),
            "Product inquiry": sum(1 for k in ["recommend", "specification", "feature", "spec", "question", "buy"] if k in text)
        }
        
        # Classify as the category with the highest keyword match score
        best_cat = max(cat_scores, key=cat_scores.get)
        category = best_cat if cat_scores[best_cat] > 0 else "Product inquiry"
        
        # --- Score priorities ---
        prio_scores = {
            "critical": sum(2 for k in ["urgent", "security", "hacked", "emergency", "asap", "lost all data"] if k in text),
            "high": sum(1 for k in ["broken", "crash", "error", "fail", "invalid", "lock"] if k in text),
            "medium": sum(1 for k in ["setup", "configure", "install", "update", "compatible"] if k in text),
            "low": 0
        }
        
        best_prio = max(prio_scores, key=prio_scores.get)
        priority = best_prio if prio_scores[best_prio] > 0 else "low"
        
        return {
            "category": category,
            "priority": priority,
            "scores": cat_scores
        }

rule_classifier = RuleClassifier()