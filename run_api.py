#!/usr/bin/env python3
"""Start the DigitalTwinSim API server."""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import uvicorn

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    env = os.getenv("DTS_ENV", "development")
    host = os.getenv("DTS_HOST", "0.0.0.0")
    port = int(os.getenv("DTS_PORT", "8000"))

    if env == "production":
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            workers=int(os.getenv("DTS_WORKERS", "2")),
            log_level="info",
        )
    else:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=[os.path.dirname(os.path.abspath(__file__))],
        )
