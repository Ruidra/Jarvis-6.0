"""
Integrations package (scaffold).

These modules define the *interface* for external services the roadmap calls for.
They are intentionally NOT auto-wired into tool dispatch — each requires provider
credentials / a local server / extra dependencies.  Implement the marked methods
and register them (via ``core.oauth.OAuthManager``) when you have credentials.
"""

from __future__ import annotations

__all__ = ["calendar", "email_client", "smarthome", "biometrics"]
