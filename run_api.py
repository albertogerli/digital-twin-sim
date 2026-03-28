#!/usr/bin/env python3
"""Start the DigitalTwinSim API server."""

import os
import sys
import uvicorn

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[os.path.dirname(os.path.abspath(__file__))],
    )
