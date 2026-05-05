"""Domain plugin discovery and registration."""

from .base_domain import DomainPlugin


class DomainRegistry:
    """Registry for domain plugins."""
    _domains: dict[str, type[DomainPlugin]] = {}

    @classmethod
    def register(cls, domain_class: type[DomainPlugin]):
        cls._domains[domain_class.domain_id] = domain_class
        return domain_class

    @classmethod
    def get(cls, domain_id: str) -> DomainPlugin:
        if domain_id not in cls._domains:
            raise ValueError(
                f"Unknown domain '{domain_id}'. "
                f"Available: {', '.join(cls._domains.keys())}"
            )
        return cls._domains[domain_id]()

    @classmethod
    def list_domains(cls) -> list[str]:
        return list(cls._domains.keys())

    @classmethod
    def discover(cls):
        """Auto-discover domain plugins by importing all domain packages."""
        import importlib
        domain_packages = [
            "domains.political",
            "domains.commercial",
            "domains.marketing",
            "domains.corporate",
            "domains.public_health",
            "domains.financial",
            "domains.telecommunications",
        ]
        for pkg in domain_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                pass  # Plugin not yet implemented
