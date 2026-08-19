"""
Calendar integration (Google Calendar / Outlook) — REQUIRES SETUP.

Needs OAuth credentials (see ``core/oauth.py``) and the ``google-api-python-client``
or ``O365`` package.  Wire into a ``calendar`` tool once configured.
"""


class CalendarClient:  # pragma: no cover - scaffold
    def __init__(self, provider: str = "google_calendar"):
        self.provider = provider

    def is_ready(self) -> bool:
        from core.oauth import OAuthManager
        return OAuthManager().is_configured(self.provider)

    def upcoming(self, limit: int = 5) -> str:
        if not self.is_ready():
            return "Calendar not configured. Add OAuth credentials to enable."
        raise NotImplementedError("Implement list_events() via the provider SDK.")

    def add_event(self, title: str, when: str) -> str:
        if not self.is_ready():
            return "Calendar not configured. Add OAuth credentials to enable."
        raise NotImplementedError("Implement insert_event() via the provider SDK.")
