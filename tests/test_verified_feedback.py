from global_weather.continuous.verified_feedback import eligible_sample_ids


def test_eligible_requires_admission_and_forecast_verification():
    a = "a" * 64
    b = "b" * 64
    events = [
        ("fact_admitted_verified", {"sample_id": a}),
        ("forecast_verified_sample", {"sample_id": a}),
        ("fact_admitted_verified", {"sample_id": b}),
    ]
    assert eligible_sample_ids(events) == {a}
