import re

with open("core/orchestrator/financial_impact.py", "r") as f:
    text = f.read()

# Replace TICKER_SECTOR and SECTOR_BETAS usages in score_round
r1 = """
        for ticker, agent_source in agent_tickers.items():
            if not any(t.ticker == ticker for t in all_ticker_impacts):
                target_stock = next((s for s in UniverseLoader()._universe.get("stocks", []) if s["ticker"] == ticker), None)
                sector = target_stock["sector"] if target_stock else "unknown"
                beta = UniverseLoader().get_beta(sector)
                impact = self._compute_ticker_impact(
                    ticker, sector, "short", base_intensity, beta, agent_source,
                )
                all_ticker_impacts.append(impact)
"""
text = re.sub(r'        for ticker, agent_source in agent_tickers.items\(\):.*?all_ticker_impacts\.append\(impact\)', r1.strip("\n"), text, flags=re.DOTALL)

# Modify signature of score_round to accept RelevantUniverse
text = text.replace("active_agents: list = None,", "active_agents: list = None,\n        relevant_universe=None,")

# Replace SECTOR_BETAS in _build_pair_trade
r2 = """
        # SHORT leg: sectors that lose
        for sector_key in pair_config["short"]:
            beta_data = UniverseLoader().get_beta(sector_key)

            # Find tickers for this sector
            sector_tickers = self._tickers_for_sector(sector_key, agent_tickers)
            for ticker in sector_tickers[:3]:  # max 3 per sector
                source = agent_tickers.get(ticker, f"pair_trade:{topic}")
                impact = self._compute_ticker_impact(
                    ticker, sector_key, "short", intensity, beta_data, source,
                )
                short_leg.append(impact)

        # LONG leg: sectors that benefit
        for sector_key in pair_config["long"]:
            beta_data = UniverseLoader().get_beta(sector_key)

            sector_tickers = self._tickers_for_sector(sector_key, agent_tickers)
            for ticker in sector_tickers[:2]:  # max 2 per long sector
                source = agent_tickers.get(ticker, f"pair_trade:{topic}")
                impact = self._compute_ticker_impact(
                    ticker, sector_key, "long", intensity, beta_data, source,
                )
                long_leg.append(impact)
"""
text = re.sub(r"        # SHORT leg: sectors that lose.*?long_leg\.append\(impact\)", r2.strip("\n"), text, flags=re.DOTALL)

# Replace _tickers_for_sector
r3 = """
    def _tickers_for_sector(
        self, sector_key: str, agent_tickers: dict[str, str],
    ) -> list[str]:
        \"\"\"Get tickers for a sector, preferring agent-sourced ones.\"\"\"
        all_stocks = UniverseLoader().tickers_for_sector(sector_key)
        all_tickers = [s["ticker"] for s in all_stocks]
        
        agent_in_sector = [t for t in agent_tickers if t in all_tickers]
        static_in_sector = [t for t in all_tickers if t not in agent_tickers]
        
        return agent_in_sector + static_in_sector
"""
text = re.sub(r"    def _tickers_for_sector\(.*?return agent_in_sector \+ static_in_sector", r3.strip("\n"), text, flags=re.DOTALL)


with open("core/orchestrator/financial_impact.py", "w") as f:
    f.write(text)
