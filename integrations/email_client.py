"""
Email integration (Gmail / Outlook) — REQUIRES SETUP.

Needs OAuth credentials (see ``core/oauth.py``).  Wire into a ``send_email`` /
``read_email`` tool once configured.
"""


class EmailClient:  # pragma: no cover - scaffold
    def __init__(self, provider: str = "google_gmail"):
        self.provider = provider

    def is_ready(self) -> bool:
        from core.oauth import OAuthManager
        return OAuthManager().is_configured(self.provider)

    def send(self, to: str, subject: str, body: str) -> str:
        if not self.is_ready():
            return "Email not configured. Add OAuth credentials to enable."
        raise NotImplementedError("Implement send() via the provider SDK.")

    def unread(self, limit: int = 5) -> str:
        if not self.is_ready():
            return "Email not configured. Add OAuth credentials to enable."
        raise NotImplementedError("Implement list_messages() via the provider SDK.")
