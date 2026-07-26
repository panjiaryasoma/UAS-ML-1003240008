from fastapi.testclient import TestClient

from app.main import app


def predict(client: TestClient, text: str) -> dict:
    response = client.post(
        "/predict-teks",
        json={
            "text": text,
            "language": "id",
        },
    )

    assert response.status_code == 200
    return response.json()


def test_negation_moves_probability_toward_negative():
    with TestClient(app) as client:
        positive_result = predict(
            client,
            "aplikasinya bagus dan mudah digunakan",
        )
        negated_result = predict(
            client,
            "aplikasinya tidak bagus dan tidak mudah digunakan",
        )

    positive_before = positive_result["probabilities"]["positive"]
    positive_after = negated_result["probabilities"]["positive"]

    negative_before = positive_result["probabilities"]["negative"]
    negative_after = negated_result["probabilities"]["negative"]

    assert (
        positive_after < positive_before
        or negative_after > negative_before
    )


def test_complaint_has_higher_negative_probability_than_praise():
    with TestClient(app) as client:
        praise_result = predict(
            client,
            "makanannya enak dan pelayanannya sangat baik",
        )
        complaint_result = predict(
            client,
            "makanannya tidak enak dan pelayanannya sangat buruk",
        )

    praise_negative = praise_result["probabilities"]["negative"]
    complaint_negative = complaint_result["probabilities"]["negative"]

    assert complaint_negative > praise_negative