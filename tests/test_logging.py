import logging

from fastapi.testclient import TestClient

from app.main import app


def test_prediction_logging_records_metadata_without_raw_text(caplog):
    raw_text = "teks rahasia pengguna jangan sampai masuk log"

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.post(
                "/predict-teks",
                json={
                    "text": raw_text,
                    "language": "id",
                },
            )

    assert response.status_code == 200

    response_body = response.json()
    all_messages = [
        record.getMessage()
        for record in caplog.records
    ]

    prediction_logs = [
        message
        for message in all_messages
        if "prediction_completed" in message
    ]

    assert prediction_logs, "Log prediksi belum ditemukan"

    log_message = prediction_logs[-1]

    assert "timestamp_utc=" in log_message
    assert "language=id" in log_message
    assert f"text_length={len(raw_text)}" in log_message
    assert (
        f"prediction={response_body['prediction']}"
        in log_message
    )
    assert "confidence=" in log_message
    assert (
        f"model_version={response_body['model_version']}"
        in log_message
    )
    assert "latency_ms=" in log_message

    # Privasi: isi teks pengguna tidak boleh dicatat.
    assert raw_text not in "\n".join(all_messages)