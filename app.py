"""Q-EWS workbench — agreed 7-screen audit UI.

    python app.py
    # http://localhost:7860
"""
from __future__ import annotations

import uvicorn

from src.application.workbench_server import create_app


if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
