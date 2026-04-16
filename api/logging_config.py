"""Structured logging configuration for DigitalTwinSim."""

import logging
import os
import sys

import structlog


def setup_logging():
    """Configure structlog for JSON (prod) or console (dev) output."""
    env = os.getenv("DTS_ENV", "development")
    log_level = os.getenv("DTS_LOG_LEVEL", "INFO").upper()

    if env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )


def get_logger(name: str = None):
    """Get a structlog logger with optional name binding."""
    log = structlog.get_logger()
    if name:
        log = log.bind(module=name)
    return log
