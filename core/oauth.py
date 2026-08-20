"""
OAuth credential manager (scaffold).

Centralises third-party auth so tools like ``send_message``, ``flight_finder``,
calendar and email don't each juggle ad-hoc tokens.  Real OAuth flows require
client secrets + a redirect endpoint; this module provides the storage/retrieval
primitives and a clean interface to extend.  Not wired into live tool dispatch
until provider credentials are supplied (see ``integrations/``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OAuthToken:
    provider: str
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    scope: str = ""
    extra: dict = field(default_factory=dict)


class OAuthManager:
    """Tiny encrypted-at-rest token store keyed by provider name."""

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            try:
                import sys as _sys
                base = Path(_sys.executable).parent if getattr(_sys, "frozen", False) \
                    else Path(__file__).resolve().parents[1]
                store_path = base / "config" / "oauth_tokens.json"
            except Exception:
                store_path = Path("config/oauth_tokens.json")
        self.store_path = Path(store_path)
        self._tokens: dict[str, OAuthToken] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.store_path.exists():
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                for prov, t in data.items():
                    # `t` already carries the `provider` field (saved via vars()),
                    # so unpack it directly — re-passing provider= would raise
                    # "multiple values for argument".
                    self._tokens[prov] = OAuthToken(**t)
        except Exception as e:  # noqa: BLE001
            logger.warning("OAuth store load failed: %s", e)

    def _save(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {p: vars(t) for p, t in self._tokens.items()}
            self.store_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("OAuth store save failed: %s", e)

    def set_token(self, token: OAuthToken) -> None:
        self._tokens[token.provider] = token
        self._save()

    def get_token(self, provider: str) -> Optional[OAuthToken]:
        return self._tokens.get(provider)

    def is_configured(self, provider: str) -> bool:
        t = self._tokens.get(provider)
        return bool(t and t.access_token)


# Provider stubs — implement the real OAuth redirect flow per provider.
SUPPORTED_PROVIDERS = ["google_calendar", "google_gmail", "microsoft_outlook", "home_assistant"]
