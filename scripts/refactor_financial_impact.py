import re
import os

with open("core/orchestrator/financial_impact.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Remove ORG_TICKER_MAP
text = re.sub(r"ORG_TICKER_MAP: dict\[str, list\[str\]\] = \{.*?\}\n\n", "", text, flags=re.DOTALL)
# 2. Modify _resolve_tickers_for_org to use UniverseLoader
resolve_ast = """
def _resolve_tickers_for_org(org: str) -> list[str]:
    \"\"\"Resolve tickers from an organisation name.\"\"\"
    if not org:
        return []
    org_lower = org.lower().strip()
    return UniverseLoader().resolve_org(org_lower)
"""
text = re.sub(r"def _resolve_tickers_for_org\(org: str\) -> list\[str\]:.*?return \[\]\n\n", resolve_ast, text, flags=re.DOTALL)

# 3. Remove SECTOR_BETAS and TICKER_SECTOR
text = re.sub(r"SECTOR_BETAS: dict\[str, SectorBeta\] = \{.*?\}\n\n", "", text, flags=re.DOTALL)
text = re.sub(r"# Ticker → sector mapping.*?TICKER_SECTOR: dict\[str, str\] = \{.*?\}\n\n", "", text, flags=re.DOTALL)

# 4. Insert UniverseLoader before Pair Trade Logic
loader_code = """
import json

class UniverseLoader:
    _instance = None
    _universe = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UniverseLoader, cls).__new__(cls)
            cls._instance._load()
        return cls._instance
        
    def _load(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(base_dir, "shared", "stock_universe.json")
        try:
            with open(path, "r") as f:
                self._universe = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load stock universe: {e}")
            self._universe = {"sectors": {}, "stocks": [], "org_aliases": {}}
            
    def get_beta(self, sector: str, regime: str = "default") -> SectorBeta:
        s_data = self._universe.get("sectors", {}).get(sector, {})
        betas = s_data.get("betas", {})
        target = betas.get(regime) or betas.get("default", {})
        return SectorBeta(
            sector=sector,
            political_beta=target.get("political_beta", 1.0),
            spread_beta=target.get("spread_beta", 0.0),
            crisis_alpha=target.get("crisis_alpha", 0.0),
            volatility_multiplier=target.get("volatility_multiplier", 1.0)
        )
        
    def resolve_org(self, name: str) -> list[str]:
        aliases = self._universe.get("org_aliases", {})
        for k, v in aliases.items():
            if k in name or name in k:
                return v
        return []
        
    def tickers_for_sector(self, sector: str) -> list[dict]:
        return [s for s in self._universe.get("stocks", []) if s.get("sector") == sector]

"""
text = text.replace("# ── Pair Trade Logic", loader_code + "\n# ── Pair Trade Logic")

# 5. Add new pair trades to CRISIS_PAIR_TRADES
new_trades = {
    "trade_war": {"short": ["automotive", "tech"], "long": ["defense", "utilities"], "rationale": "Trade barriers hit supply chains."},
    "central_bank_hawkish": {"short": ["real_estate", "tech"], "long": ["banking", "insurance"], "rationale": "Rates up, NIM up."},
    "tech_regulation": {"short": ["tech"], "long": ["telecom", "banking"], "rationale": "Big tech regulated, incumbents gain."},
    "pandemic": {"short": ["luxury", "automotive"], "long": ["healthcare", "tech"], "rationale": "Lockdowns help tech and health."},
    "commodity_shock": {"short": ["food_consumer", "automotive"], "long": ["energy_fossil"], "rationale": "Input costs up."},
    "currency_crisis": {"short": ["banking", "sovereign_debt"], "long": ["luxury", "tech"], "rationale": "Flight to global earners."},
    "supply_chain": {"short": ["automotive", "tech"], "long": ["infrastructure"], "rationale": "Logistics bottlenecks."},
    "geopolitical": {"short": ["banking", "energy_fossil"], "long": ["defense", "utilities"], "rationale": "War risks."}
}

for k, v in new_trades.items():
    if k not in text:
        inj = f'    "{k}": {{\n        "short": {v["short"]},\n        "long": {v["long"]},\n        "rationale": "{v["rationale"]}",\n    }},\n'
        text = re.sub(r"(CRISIS_PAIR_TRADES: dict\[str, dict\] = \{)", r"\1\n" + inj, text)


with open("core/orchestrator/financial_impact.py", "w", encoding="utf-8") as f:
    f.write(text)

