"""The API must only answer its own page: no other websites, no DNS rebinding."""
import pytest
from fastapi.testclient import TestClient

from app import db, security
from app.main import app

PORT = 8000
SMART = {"provider": "openrouter", "model": "mock/smart"}
VOTE = {"a": SMART, "b": {"provider": "openrouter", "model": "mock/weak"}, "winner": "a"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as test_client:
        yield test_client


# ---------------------------------------------------------------- header checks

@pytest.mark.parametrize("host", [
    "localhost:8000", "127.0.0.1:8000", "LOCALHOST:8000", " localhost:8000 ",
])
def test_host_allowed(host):
    assert security.host_allowed(host, PORT)


@pytest.mark.parametrize("host", [
    "evil.example", "evil.example:8000",          # DNS rebinding: attacker's name, our address
    "localhost.evil.example:8000", "127.0.0.1.evil.example:8000", "evil-localhost:8000",
    "localhost:9000", "localhost",                 # wrong port / default port 80
    "127.0.0.2:8000", "0.0.0.0:8000", "192.168.1.20:8000", "[::1]:8000",
    "user@localhost:8000", "localhost:8000/x", "localhost:8000, evil.example", "localhost:", "",
])
def test_host_rejected(host):
    assert not security.host_allowed(host, PORT)


@pytest.mark.parametrize("origin", ["http://localhost:8000", "http://127.0.0.1:8000", "HTTP://LocalHost:8000"])
def test_origin_allowed(origin):
    assert security.origin_allowed(origin, PORT)


@pytest.mark.parametrize("origin", [
    "https://evil.example", "http://evil.example:8000", "null",
    "http://localhost:3000", "http://localhost",       # another local app / default port
    "https://localhost:8000",                          # we never serve https
    "http://localhost:8000.evil.example", "http://localhost.evil.example:8000",
    "http://localhost:8000/", "http://user@localhost:8000", "file://", "",
])
def test_origin_rejected(origin):
    assert not security.origin_allowed(origin, PORT)


def test_port_80_needs_no_port_in_the_header():
    assert security.host_allowed("localhost", 80) and security.origin_allowed("http://127.0.0.1", 80)
    assert not security.host_allowed("localhost:8000", 80)


def test_unknown_server_port_still_checks_the_name():
    assert security.host_allowed("localhost:1234", None)
    assert not security.host_allowed("evil.example:1234", None)


def test_duplicate_headers_are_rejected():
    assert security.request_allowed(["localhost:8000"], [], PORT)
    assert security.request_allowed(["localhost:8000"], ["http://localhost:8000"], PORT)
    assert not security.request_allowed([], [], PORT)                                   # no Host at all
    assert not security.request_allowed(["localhost:8000", "evil.example"], [], PORT)
    assert not security.request_allowed(["localhost:8000"], ["http://localhost:8000", "https://evil.example"], PORT)


@pytest.mark.parametrize("value", ["same-origin", "none", "Same-Origin", " none "])
def test_fetch_site_allowed(value):
    assert security.fetch_site_allowed(value)


@pytest.mark.parametrize("value", [
    "cross-site",                # another website
    "same-site",                 # another port on localhost: not our page either
    "CROSS-SITE", "", "same-origin, cross-site", "anything-new",
])
def test_fetch_site_rejected(value):
    assert not security.fetch_site_allowed(value)


def test_fetch_site_is_checked_with_host_and_origin():
    host, origin = ["localhost:8000"], ["http://localhost:8000"]
    assert security.request_allowed(host, [], PORT, [])                       # header absent (curl, old browsers)
    assert security.request_allowed(host, origin, PORT, ["same-origin"])
    assert security.request_allowed(host, [], PORT, ["none"])
    assert not security.request_allowed(host, [], PORT, ["cross-site"])
    assert not security.request_allowed(host, origin, PORT, ["cross-site"])   # a good Origin does not rescue it
    assert not security.request_allowed(host, [], PORT, ["same-origin", "cross-site"])   # sent twice
    assert not security.request_allowed(["evil.example:8000"], [], PORT, ["same-origin"])
    assert not security.request_allowed(host, ["https://evil.example"], PORT, ["same-origin"])


# ---------------------------------------------------------------- through the app

def test_own_page_is_allowed(client):
    assert client.get("/api/config").status_code == 200                      # no Origin (same-origin GET)
    for origin in ("http://localhost:8000", "http://127.0.0.1:8000"):
        assert client.get("/api/config", headers={"Origin": origin}).status_code == 200
    assert client.get("/api/config", headers={"Host": "localhost:8000"}).status_code == 200
    assert client.post("/api/test-connection", json=SMART,
                       headers={"Origin": "http://localhost:8000"}).json()["ok"]


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"},
    {"Origin": "http://localhost:3000"},
    {"Origin": "null"},
    {"Host": "evil.example:8000"},
    {"Host": "localhost:9999"},
    {"Host": "evil.example:8000", "Origin": "http://evil.example:8000"},      # rebinding: looks same-origin
    {"Host": ""},
])
def test_foreign_requests_are_refused(client, headers):
    response = client.get("/api/config", headers=headers)
    assert response.status_code == 403
    assert response.json() == {"detail": security.REJECTION}


@pytest.mark.parametrize("headers, status", [
    ({"Sec-Fetch-Site": "same-origin"}, 200),                                 # fetch() from our own page
    ({"Sec-Fetch-Site": "same-origin", "Origin": "http://localhost:8000"}, 200),
    ({"Sec-Fetch-Site": "none"}, 200),                                        # typed into the address bar
    ({"Sec-Fetch-Site": "cross-site"}, 403),                                  # <img>, link or no-cors fetch elsewhere
    ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"}, 403),
    ({"Sec-Fetch-Site": "cross-site", "Origin": "http://localhost:8000"}, 403),
    ({"Sec-Fetch-Site": "same-site"}, 403),                                   # another app on localhost
    ({"Sec-Fetch-Site": "same-origin", "Origin": "https://evil.example"}, 403),
    ({"Sec-Fetch-Site": "same-origin", "Host": "evil.example:8000"}, 403),
])
def test_sec_fetch_site_is_enforced(client, headers, status):
    # A cross-site GET carries no Origin, so before this check nothing refused it.
    response = client.get("/api/runs", headers=headers)
    assert response.status_code == status
    if status == 403:
        assert response.json() == {"detail": security.REJECTION}


def test_cross_site_get_cannot_read_or_export_results(client):
    cross = {"Sec-Fetch-Site": "cross-site"}
    for url in ("/api/config", "/api/models", "/api/runs", "/api/runs/export?format=json",
                "/api/runs/export?format=csv", "/api/arena/ratings", "/api/custom-tests", "/api/benchmark/tests",
                "/api/benchmark/active"):
        assert client.get(url).status_code == 200, url
        refused = client.get(url, headers=cross)
        assert refused.status_code == 403 and "content-disposition" not in refused.headers, url


@pytest.mark.parametrize("evil", [{"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"}],
                         ids=["origin", "sec-fetch-site"])
def test_every_api_route_is_guarded(client, evil):
    api_routes = [(method, route.path) for route in app.routes
                  if getattr(route, "path", "").startswith("/api/") for method in route.methods]
    assert len(api_routes) >= 15
    for method, path in api_routes:
        url = path.replace("{run_id}", "1").replace("{test_id}", "1")
        assert client.request(method, url, headers=evil).status_code == 403, (method, path)
    assert client.options("/api/chat/stream", headers={                       # CORS preflight
        **evil, "Access-Control-Request-Method": "POST"}).status_code == 403
    assert client.get("/api/does-not-exist", headers=evil).status_code == 403


def test_refused_requests_have_no_side_effects(client):
    client.delete("/api/arena/votes")
    assert client.post("/api/arena/vote", json=VOTE).json()["votes"] == 1
    evil = {"Origin": "https://evil.example"}

    assert client.post("/api/arena/vote", json=VOTE, headers=evil).status_code == 403
    assert client.delete("/api/arena/votes", headers=evil).status_code == 403
    assert client.delete("/api/arena/votes", headers={"Host": "evil.example:8000"}).status_code == 403
    assert len(db.list_votes()) == 1

    assert client.delete("/api/arena/votes").json()["deleted"] == 1


def test_cross_site_post_cannot_spend_the_key(client):
    """A "simple" cross-site POST needs no CORS preflight, so its Origin has to stop it.

    The installed FastAPI also rejects a body without a JSON content type (422), but that is
    version-dependent; the guard must not lean on it, so the answer has to be 403 either way.
    """
    body = '{"target": {"provider": "openrouter", "model": "mock/smart"}, "messages": [{"role": "user", "content": "hi"}]}'
    for content_type in ("text/plain", "application/json"):
        response = client.post("/api/chat/stream", content=body,
                               headers={"Origin": "https://evil.example", "Content-Type": content_type})
        assert response.status_code == 403 and "data:" not in response.text
    own = client.post("/api/chat/stream", content=body, headers={"Content-Type": "application/json"})
    assert own.status_code == 200 and '"type": "done"' in own.text


def test_pages_and_static_files_are_not_guarded(client):
    # Nothing sensitive is served outside /api/, and a rebound page still cannot call the API.
    rebound = {"Host": "evil.example:8000"}
    assert client.get("/", headers=rebound).status_code == 200
    assert client.get("/static/js/main.js", headers=rebound).status_code == 200
