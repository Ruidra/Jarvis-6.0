"""
Smart home integration (Home Assistant / Matter) — REQUIRES SETUP.

Point ``base_url`` at a local Home Assistant instance with a long-lived token.
Wire into a ``smart_home`` tool once configured.
"""


class SmartHomeClient:  # pragma: no cover - scaffold
    def __init__(self, base_url: str = "", token: str = ""):
        self.base_url = base_url
        self.token = token

    def is_ready(self) -> bool:
        return bool(self.base_url and self.token)

    def states(self) -> str:
        if not self.is_ready():
            return "Smart home not configured (needs a Home Assistant URL + token)."
        raise NotImplementedError("Implement GET /api/states via the HA REST API.")
