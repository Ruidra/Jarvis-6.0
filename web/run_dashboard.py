"""
Headless launcher for the JARVIS Web / phone UI dashboard.

This serves the neon/glass remote-control interface (dashboard/static/app.html)
together with the /metrics health endpoint on port 8000.  It is the entrypoint
used by the Docker image so the Web UI ships alongside the phone UI.

Run directly::

    python web/run_dashboard.py

Requires: fastapi, uvicorn[standard], cryptography  (all in requirements.txt).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def main() -> None:
    # Make the project root importable when launched from any working directory.
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from dashboard.server import DashboardServer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Dashboard dependencies missing. "
            "Run: pip install fastapi 'uvicorn[standard]' cryptography\n"
            f"({exc})"
        )

    server = DashboardServer()
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":
    main()
