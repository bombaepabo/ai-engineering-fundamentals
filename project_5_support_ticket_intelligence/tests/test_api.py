# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

# Initialize the test client
client = TestClient(app)

# Helper headers
headers = {"X-API-Key": "dev-secret-key"}


def test_unauthorized_access():
    """
    Checks that requests to protected endpoints without the X-API-Key header are blocked.
    """
    # Change /health to /tickets
    response = client.get("/tickets")
    assert response.status_code == 401
    assert "API Key missing" in response.json()["detail"]

def test_health_check_endpoint():
    """
    Checks that the health endpoint returns a successful status.
    """
    response = client.get("/health", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["database"] == "healthy"


def test_create_ticket_validation():
    """
    Verifies that ticket creation fails if the message is too short.
    """
    bad_ticket = {
        "subject": "Help",
        "message": "short"  # Message must be >= 10 chars
    }
    response = client.post("/tickets", json=bad_ticket, headers=headers)
    assert response.status_code == 422  # Validation error (Unprocessable Entity)


def test_create_ticket_success():
    """
    Checks that ticket creation works with correct schema inputs.
    """
    good_ticket = {
        "subject": "Trouble loading page",
        "message": "The login page hangs indefinitely on loading spinner."
    }
    response = client.post("/tickets", json=good_ticket, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] is not None
    assert data["subject"] == good_ticket["subject"]