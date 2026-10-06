import pytest
from app import app, get_recommendations


@pytest.fixture
def client():
    """Configures Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# --- Core Logic Unit Tests ---


def test_get_recommendations_valid_movie():
    """Verify core function returns the expected number of recommendations."""
    recs = get_recommendations("Batman", top_k=3)
    assert recs is not None
    assert len(recs) == 3
    assert "title" in recs[0]
    assert "similarity_score" in recs[0]
    assert "match_percent" in recs[0]


def test_get_recommendations_invalid_movie():
    """Verify function returns None when title does not exist."""
    recs = get_recommendations("NonExistentMovieXYZ_123")
    assert recs is None


# --- HTTP Endpoint Integration Tests ---


def test_health_endpoint(client):
    """Verify /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert "total_movies" in data


def test_recommend_api_success(client):
    """Verify /recommend returns 200 with JSON payload."""
    response = client.get("/recommend?title=Batman&top_k=2")
    assert response.status_code == 200

    data = response.get_json()
    assert data["query"] == "Batman"
    assert data["count"] == 2
    assert len(data["recommendations"]) == 2


def test_recommend_api_missing_param(client):
    """Verify /recommend returns 400 when title param is missing."""
    response = client.get("/recommend")
    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


def test_recommend_api_not_found(client):
    """Verify /recommend returns 404 for unknown titles."""
    response = client.get("/recommend?title=RandomMadeUpMovie")
    assert response.status_code == 404


def test_ui_page_loads(client):
    """Verify root HTML dashboard renders successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Movie Recommender" in response.data