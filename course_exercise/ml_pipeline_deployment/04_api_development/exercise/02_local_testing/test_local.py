from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# def test_api_locally_get_items():
#     r = client.get("/items/1/5")
#     print(r.status_code)
#     assert r.status_code == 200


def test_get_path():
    r = client.get("/items/42")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 1 of 42"}


def test_get_path_query():
    r = client.get("/items/42?count=5")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 5 of 42"}


def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200