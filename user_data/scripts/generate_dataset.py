#!/usr/bin/env python3
"""
Script to prepare dataset for CatBoost model using CatBoostPreparationStrategy.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from freqtrade.configuration import Configuration
from freqtrade.data.history import load_pair_history
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.data.dataprovider import DataProvider
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist

# Add user_data/strategies to path to ensure strategy can be found
import sys
sys.path.append("./user_data/strategies")

def main():
    # 1. Configuration
    # Minimal config for data loading
    config_path = "user_data/config.json"

    # Check if config exists, if not use a dummy config
    if not Path(config_path).exists():
        print(f"Config file not found at {config_path}. Using default config structure.")
        # Minimal dummy config
        config = {
            "user_data_dir": "user_data",
            "datadir": "user_data/data/binance", # Adjust exchange/path
            "exchange": {"name": "binance", "key": "", "secret": ""},
            "timeframe": "5m",
            "pairs": ["BTC/USDT", "ETH/USDT"], # Default pairs if not in config
            "stake_currency": "USDT",
            "dry_run": True
        }
    else:
        # Load config
        config = Configuration.from_files([config_path])
        config['user_data_dir'] = 'user_data' # Ensure correct user_data

    # Setup simple logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 2. Load Strategy
    strategy_name = "CatBoostPreparationStrategy"
    try:
        strategy_class = StrategyResolver.load_strategy_by_name(strategy_name)
        strategy = strategy_class(config=config)
    except Exception as e:
        logger.error(f"Failed to load strategy {strategy_name}: {e}")
        return

    # 3. Data Loading
    # We iterate over pairs defined in config or passed as args
    pairs = config.get("pairs", [])
    if not pairs:
        logger.warning("No pairs found in config.")
        return

    timeframe = strategy.timeframe
    data_dir = Path(config['datadir'])

    all_data = []

    for pair in pairs:
        logger.info(f"Processing pair: {pair}")
        try:
            # Load OHLCV data
            # timerange can be None to load all data
            dataframe = load_pair_history(pair=pair, timeframe=timeframe, datadir=data_dir)

            if dataframe.empty:
                logger.warning(f"No data found for {pair}")
                continue

            # 4. Feature Engineering & Target Creation
            # populate_indicators calls target_creation internally in this strategy
            dataframe = strategy.populate_indicators(dataframe, metadata={'pair': pair})

            # Add pair column for reference
            dataframe['pair'] = pair

            # Optional: Filter columns?
            # User wants "Features (X)" and "Targets (Y)"
            # We keep everything for now or drop raw OHLCV if desired.
            # Usually better to keep raw for sanity checks unless file is huge.

            all_data.append(dataframe)

        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")

    if not all_data:
        logger.error("No data processed.")
        return

    # 5. Concatenate and Save
    full_df = pd.concat(all_data, ignore_index=True)

    # Handling potential object columns (like pair name) if specific format needed
    # CatBoost handles categorical features well, but 'pair' might need to be explicitly cast if used.

    output_path = Path("user_data/data/catboost_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving dataset to {output_path} with shape {full_df.shape}")
    full_df.to_csv(output_path, index=False)

    # Optional: Save as Pickle or Feather for speed
    # full_df.to_feather("user_data/data/catboost_dataset.feather")

    logger.info(f"Done. Dataset saved to {output_path}")

if __name__ == "__main__":
    main()
