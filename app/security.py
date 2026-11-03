"""Local-only guard for the API.

The server holds API keys (from .env) and will spend them for anyone who can
reach /api/*. Binding to 127.0.0.1 keeps other machines out, but not the
user's own browser, which any web page can point at localhost:
  - another site can POST to http://localhost:<port>/api/... (cross-site request)
  - a hostile domain can re-point its DNS at 127.0.0.1 (DNS rebinding), after
    which the browser treats our API as same-origin with that domain
Both are visible in the request headers: a cross-site request carries the
other site's Origin, and a rebound request carries the attacker's domain in
Host. So /api/* only answers when Host - and Origin, when the browser sends
one - name localhost or 127.0.0.1 on the port we are listening on.

Browsers leave Origin off some requests (a plain GET started by a link, an
image tag or a no-cors fetch), but they label every request with
Sec-Fetch-Site, which page scripts cannot set or remove. That header is
checked as well, so a cross-site GET is refused too.
"""
from __future__ import annotations

import re

LOCAL_HOSTS = ("localhost", "127.0.0.1")

_HOSTS = "|".join(re.escape(h) for h in LOCAL_HOSTS)
_HOST = re.compile(rf"({_HOSTS})(?::(\d{{1,5}}))?", re.I)
_ORIGIN = re.compile(rf"http://({_HOSTS})(?::(\d{{1,5}}))?", re.I)   # the app is only ever served over http
_DEFAULT_PORT = 80

# Sec-Fetch-Site values that can be this app: "same-origin" is a call from our own page, "none" is
# the user opening an /api/ URL directly (address bar, bookmark). "cross-site" is another website
# and "same-site" is another port on localhost; neither is our page. Clients that do not send the
# header at all (curl, scripts, old browsers) are still held to the Host and Origin checks.
OWN_FETCH_SITES = ("same-origin", "none")

REJECTION = "Forbidden: this API only accepts requests from the Model Lab page on localhost."


def _is_local(pattern: re.Pattern[str], value: str, port: int | None) -> bool:
    match = pattern.fullmatch(value.strip())
    if not match:
        return False
    # port is None only when the server cannot tell which port it listens on (e.g. a unix socket).
    return port is None or int(match.group(2) or _DEFAULT_PORT) == port


def host_allowed(host: str, port: int | None) -> bool:
    """True if a Host header value names this server: localhost or 127.0.0.1 on our port."""
    return _is_local(_HOST, host, port)


def origin_allowed(origin: str, port: int | None) -> bool:
    """True if an Origin header value is this app's own page ("null" and other sites are not)."""
    return _is_local(_ORIGIN, origin, port)


def fetch_site_allowed(value: str) -> bool:
    """True if a Sec-Fetch-Site header value says the request came from our own page or the user."""
    return value.strip().lower() in OWN_FETCH_SITES


def request_allowed(hosts: list[str], origins: list[str], port: int | None,
                    fetch_sites: list[str] | tuple[str, ...] = ()) -> bool:
    """Decide from the raw Host / Origin / Sec-Fetch-Site values (as lists, so duplicates are caught)."""
    if len(hosts) != 1 or not host_allowed(hosts[0], port):
        return False
    if not all(fetch_site_allowed(site) for site in fetch_sites):
        return False
    return all(origin_allowed(origin, port) for origin in origins)
