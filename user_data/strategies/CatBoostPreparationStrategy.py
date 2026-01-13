
import numpy as np
import pandas as pd
import pandas_ta as ta
from freqtrade.strategy import IStrategy

class CatBoostPreparationStrategy(IStrategy):
    """
    Strategy to prepare dataset for CatBoost model with Meta-Labeling.
    Focus on Feature Engineering and Dual-Target Labeling.
    """
    INTERFACE_VERSION = 3

    # Timeframe
    timeframe = '5m'

    # Minimal ROI and Stoploss (Placeholders)
    minimal_roi = {"0": 100}
    stoploss = -0.99

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Feature Engineering (The 'X')
        Do NOT use raw prices. Use normalized values.
        """

        # --- Helpers for Custom Indicators ---

        def calculate_rmi(series, length=14, mom=5):
            """
            Relative Momentum Index (RMI)
            RMI is an RSI variant that uses momentum (close - close[mom]) instead of 1-day change.
            """
            delta = series.diff(mom)
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)

            # Use RMA (Wilder's Smoothing) which is standard for RSI
            ema_up = up.ewm(alpha=1/length, adjust=False).mean()
            ema_down = down.ewm(alpha=1/length, adjust=False).mean()

            rs = ema_up / ema_down
            rmi = 100 - (100 / (1 + rs))
            return rmi

        def calculate_ewo(dataframe, fast=5, slow=35):
            """
            Elliott Wave Oscillator (EWO)
            """
            sma_fast = ta.sma(dataframe['close'], length=fast)
            sma_slow = ta.sma(dataframe['close'], length=slow)
            ewo = sma_fast - sma_slow
            return ewo

        def calculate_vfi(dataframe, length=130, coef=0.2, vcoef=2.5):
            """
            Volume Flow Indicator (VFI) - Approximation
            """
            typical = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
            inter = np.log(typical) - np.log(typical.shift(1))
            vinter = inter.rolling(window=30).std()
            cutoff = coef * vinter

            close_change = typical.diff()
            vave = dataframe['volume'].rolling(window=length).mean()
            vmax = vave * vcoef
            vc = dataframe['volume'].clip(upper=vmax)

            mf = np.where(inter > cutoff, vc, np.where(inter < -cutoff, -vc, 0))
            # VFI is often smoothed sum over length
            vfi = ta.ema(pd.Series(mf), length=3) # Simplified smoothing
            # Standard VFI logic is more cumulative, but this captures volume flow direction.
            # Using a simplified Money Flow approach if exact VFI is complex:
            # Let's use a robust substitute if available or this approx.
            # Given the constraints, I will use this flow calculation normalized by volume average.
            vfi_final = pd.Series(mf).rolling(window=length).sum() / vave
            return vfi_final

        def calculate_fractals(dataframe):
            """
            Williams Fractal (Shifted to avoid lookahead)
            Fractal at t is known at t+2. So we verify at t-2.
            This function returns boolean columns indicating if a fractal happened 2 bars ago.
            """
            # Pandas TA doesn't have a direct 'fractal' boolean indicator always available.
            # Implementing 5-bar fractal.
            high = dataframe['high']
            low = dataframe['low']

            # Bearish Fractal (Peak)
            is_high_fractal = (
                (high.shift(2) < high) &
                (high.shift(1) < high) &
                (high.shift(-1) < high) &
                (high.shift(-2) < high)
            )

            # Bullish Fractal (Valley)
            is_low_fractal = (
                (low.shift(2) > low) &
                (low.shift(1) > low) &
                (low.shift(-1) > low) &
                (low.shift(-2) > low)
            )

            # IMPORTANT: Shift forward by 2 to prevent lookahead bias in the feature set.
            # The event happens at 't', but is confirmed at 't+2'.
            # So at 't+2', we know 't' was a fractal.
            # The feature at 'current time' should be "Was there a fractal 2 bars ago?"
            return is_high_fractal.shift(2).fillna(False), is_low_fractal.shift(2).fillna(False)

        # --- 1. Core Indicators ---

        # RSI (Normalized 0-1)
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14) / 100.0

        # MFI (Normalized 0-1)
        dataframe['mfi'] = ta.mfi(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], length=14) / 100.0

        # CMF (Chaikin Money Flow) - Normalized -1 to 1 naturally
        dataframe['cmf'] = ta.cmf(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], length=20)

        # BBB (Bollinger Bands Bandwidth)
        bbands = ta.bbands(dataframe['close'], length=20, std=2.0)
        # bbands returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        # BBB is Bandwidth. If not directly named 'BBB', calculate it.
        if 'BBB_20_2.0' in bbands.columns:
            dataframe['bbb'] = bbands['BBB_20_2.0']
        else:
            dataframe['bbb'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0']

        # EMA Distance (Normalized)
        ema_20 = ta.ema(dataframe['close'], length=20)
        dataframe['ema_dist'] = (dataframe['close'] - ema_20) / ema_20

        # --- 2. Custom Regime & Alpha ---

        # ADX (Normalized 0-1)
        adx = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        dataframe['adx'] = adx['ADX_14'] / 100.0

        # Choppiness Index (CHOP) (Normalized 0-1)
        dataframe['chop'] = ta.chop(dataframe['high'], dataframe['low'], dataframe['close'], length=14) / 100.0

        # KAMA (Normalized distance from close)
        kama = ta.kama(dataframe['close'], length=10)
        dataframe['kama_dist'] = (dataframe['close'] - kama) / kama

        # RMI (Normalized 0-1)
        dataframe['rmi'] = calculate_rmi(dataframe['close'], length=14, mom=5) / 100.0

        # TSI (True Strength Index) (Usually -100 to 100, Normalized to -1 to 1)
        dataframe['tsi'] = ta.tsi(dataframe['close'], length=13) / 100.0

        # EWO (Elliott Wave Oscillator) (Normalized by price)
        dataframe['ewo'] = calculate_ewo(dataframe) / dataframe['close']

        # RVol (Relative Volume)
        dataframe['rvol'] = dataframe['volume'] / dataframe['volume'].rolling(window=20).mean()

        # VFI (Volume Flow Indicator)
        dataframe['vfi'] = calculate_vfi(dataframe)

        # Williams Fractal (Booleans cast to float/int)
        frac_bear, frac_bull = calculate_fractals(dataframe)
        dataframe['fractal_high'] = frac_bear.astype(int)
        dataframe['fractal_low'] = frac_bull.astype(int)

        # Fill NaNs created by rolling windows
        dataframe.fillna(0, inplace=True)

        # --- Target Creation ---
        dataframe = self.target_creation(dataframe)

        return dataframe

    def target_creation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Dual-Target Labeling (The 'Y')
        Target A: target_scalp_6 (1% profit before -0.8% loss in 6 candles)
        Target B: target_trend_12 (Ends higher or +1.5% profit without hitting SL in 12 candles)
        """

        # Parameters
        scalp_tp = 0.010
        scalp_sl = -0.008
        scalp_horizon = 6

        trend_tp = 0.015
        trend_sl = -0.008
        trend_horizon = 12

        # Vectorized implementation using rolling max/min

        # Shift close, high, low backward to look into future
        # future_high[t] is window of high[t+1]...high[t+horizon]

        # We need efficient lookup. Rolling window shift(-horizon) is one way.
        # But rolling max/min is efficient.

        # --- Target A: target_scalp_6 ---

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=scalp_horizon)

        # Future Highs and Lows (starting from t+1)
        # rolling(window=6) on shifted(-1) data gives t+1...t+6
        future_high_max = dataframe['high'].shift(-1).rolling(window=indexer).max()
        future_low_min = dataframe['low'].shift(-1).rolling(window=indexer).min()

        # Basic check: Did we hit TP? Did we hit SL?
        hit_tp = future_high_max >= dataframe['close'] * (1 + scalp_tp)
        hit_sl = future_low_min <= dataframe['close'] * (1 + scalp_sl)

        # If both hit, we need to know WHICH happened first.
        # This is hard to vectorize perfectly without checking every candle.
        # However, for a 6-candle horizon, we can use a small loop or map.
        # Given the "dataset preparation" context, correctness is key.
        # Let's iterate only on the "ambiguous" rows (where both TP and SL are hit).

        target_scalp_6 = (hit_tp & ~hit_sl).astype(int) # Win and no SL hit -> Definite Win

        ambiguous_indices = dataframe.index[hit_tp & hit_sl]

        # For ambiguous cases, we must check candle by candle
        # This is much smaller subset than all rows
        if not ambiguous_indices.empty:
            # We can use a reduced loop or apply
            # Need numerical index for faster access
            # dataframe.index might be DatetimeIndex

            # Let's use numpy for the ambiguous check
            close_vals = dataframe['close'].values
            high_vals = dataframe['high'].values
            low_vals = dataframe['low'].values

            # Map index to integer location
            # If default index (RangeIndex), straightforward. If Datetime, need to find integer loc.
            # Assuming dataframe has a RangeIndex or we reset it temporarily?
            # Freqtrade data usually has RangeIndex or DatetimeIndex.
            # Safe way: use integer indexing.

            n = len(dataframe)
            # Find integer locations of ambiguous rows
            # We can use boolean mask on numpy array range
            mask = (hit_tp & hit_sl).values
            ambiguous_locs = np.where(mask)[0]

            for i in ambiguous_locs:
                entry = close_vals[i]
                tp_price = entry * (1 + scalp_tp)
                sl_price = entry * (1 + scalp_sl)

                # Check next 6 candles
                win = False
                loss = False
                for j in range(1, scalp_horizon + 1):
                    if i + j >= n: break

                    # Check High for TP
                    if high_vals[i+j] >= tp_price:
                        # If Low also hits SL in same candle, consider it Loss (Conservative)
                        if low_vals[i+j] <= sl_price:
                            loss = True
                        else:
                            win = True
                        break

                    # Check Low for SL
                    if low_vals[i+j] <= sl_price:
                        loss = True
                        break

                if win and not loss:
                    # Update the Series using .iloc for integer access
                    # But target_scalp_6 is a Series (with original index)
                    # Use dataframe.index[i] to get label
                    idx_label = dataframe.index[i]
                    target_scalp_6.at[idx_label] = 1

        dataframe['target_scalp_6'] = target_scalp_6

        # --- Target B: target_trend_12 ---

        indexer_trend = pd.api.indexers.FixedForwardWindowIndexer(window_size=trend_horizon)

        future_high_max_12 = dataframe['high'].shift(-1).rolling(window=indexer_trend).max()
        future_low_min_12 = dataframe['low'].shift(-1).rolling(window=indexer_trend).min()

        hit_trend_tp = future_high_max_12 >= dataframe['close'] * (1 + trend_tp)
        hit_sl_12 = future_low_min_12 <= dataframe['close'] * (1 + trend_sl)

        # Logic: Label 1 if (Hits TP OR Ends Higher) AND (No SL Hit)

        # Check "Ends Higher"
        # close[i+12] > close[i]
        future_close_12 = dataframe['close'].shift(-trend_horizon)
        ends_higher = future_close_12 > dataframe['close']

        # Condition: No SL hit AND (Hit TP OR Ends Higher)
        # Note: "Hit TP" logic in prompt: "hits +1.5% profit ... without hitting Stop Loss"
        # If it hits TP *before* SL, it counts.
        # But here we check if SL was hit *at all* in the window.
        # If SL was hit, it's disqualified (Conservative/Strict).
        # "Used to filter out fake pumps". If it crashes to SL, it's a fake pump.

        # So: ( (Hit TP) OR (Ends Higher) ) AND ( NOT Hit SL )
        # Wait, if Hit TP happens *before* Hit SL, is it a win?
        # "Label 1 if price ends HIGHER ... (or hits +1.5%) without hitting Stop Loss."
        # The phrasing "without hitting Stop Loss" usually applies to the whole path or "before the success condition".
        # If it hits TP (+1.5%), we usually exit. So subsequent SL doesn't matter.
        # But if we rely on "Ends Higher" (at candle 12), we must survive 12 candles.

        # Case 1: Hits TP (+1.5%). Does it need to avoid SL *before* TP? Yes.
        # Case 2: Ends Higher (at 12). Does it need to avoid SL *during* the 12 candles? Yes.

        # Approximation:
        # If `hit_sl_12` is False, then SL was never hit. So if `hit_trend_tp` or `ends_higher`, it's a win.
        # If `hit_sl_12` is True, we must check if TP was hit *before* SL.
        # `ends_higher` requires survival till 12, so if SL hit, `ends_higher` is invalid (we would have stopped out).

        # So we only need to disambiguate if `hit_sl_12` is True AND `hit_trend_tp` is True.

        target_trend_12 = ((hit_trend_tp | ends_higher) & ~hit_sl_12).astype(int)

        ambiguous_indices_trend = dataframe.index[hit_trend_tp & hit_sl_12]

        if not ambiguous_indices_trend.empty:
            mask_trend = (hit_trend_tp & hit_sl_12).values
            ambiguous_locs_trend = np.where(mask_trend)[0]

            for i in ambiguous_locs_trend:
                entry = close_vals[i]
                tp_price = entry * (1 + trend_tp)
                sl_price = entry * (1 + trend_sl)

                win = False
                loss = False
                for j in range(1, trend_horizon + 1):
                    if i + j >= n: break

                    if low_vals[i+j] <= sl_price:
                        loss = True
                        break # SL hit first (or same candle as TP check below? Low checked first usually or if conservative)

                    if high_vals[i+j] >= tp_price:
                        win = True
                        break # TP hit before SL

                if win and not loss:
                    target_trend_12.at[dataframe.index[i]] = 1

        dataframe['target_trend_12'] = target_trend_12

        # --- Entry Tag Mapping ---
        # 'long_scalp_bounce': target_scalp_6 TRUE (Regime: Sideways)
        dataframe['tag_scalp_bounce'] = (
            (dataframe['target_scalp_6'] == 1) &
            (dataframe['adx'] < 0.25)
        ).astype(int)

        # 'long_trend_ride': target_scalp_6 AND target_trend_12 are TRUE (Regime: Trending)
        dataframe['tag_trend_ride'] = (
            (dataframe['target_scalp_6'] == 1) &
            (dataframe['target_trend_12'] == 1) &
            (dataframe['adx'] >= 0.25)
        ).astype(int)

        # 'long_trend_boost': target_scalp_6 TRUE + Strong Momentum (RMI > 70)
        dataframe['tag_trend_boost'] = (
            (dataframe['target_scalp_6'] == 1) &
            (dataframe['rmi'] > 0.70)
        ).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe
