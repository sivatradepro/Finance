import yfinance as yf
import ta
import pandas as pd
from datetime import date, timedelta, datetime
from IPython.display import clear_output

class StockBacktester:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
    
    def fetch_data(self):
        date_fmt = "%Y-%m-%d"
        start_date_buffer = datetime.strptime(self.start_date, date_fmt) - timedelta(days=365)
        start_date_buffer = start_date_buffer.strftime(date_fmt)
        
        self.data = yf.download(self.ticker, start=start_date_buffer, end=self.end_date)
    
    def _apply_strategy(self, data, strategy_func, **kwargs):
        if self.data is not None:
            return strategy_func(data, **kwargs)
        else:
            print("Data not available. Please ensure data is fetched.")
            return data
    
    def run_stock_ta_backtest(self, stop_loss_lvl=None):
        bt_df = self.data
        balance = 1000000
        pnl = 0
        position = 0

        last_signal = "hold"
        last_price = 0
        c = 0

        trade_date_start = []
        trade_date_end = []
        trade_days = []
        trade_side = []
        trade_pnl = []
        trade_ret = []

        cum_value = []

        for index, row in bt_df.iterrows():
            # check and close any positions
            if row.EXIT_LONG and last_signal == "long":
                trade_date_end.append(row.name)
                trade_days.append(c)

                pnl = (row.Open - last_price) * position
                trade_pnl.append(pnl)
                trade_ret.append((row.Open / last_price - 1) * 100)

                balance = balance + row.Open * position

                position = 0
                last_signal = "hold"

                c = 0

            elif row.EXIT_SHORT and last_signal == "short":
                trade_date_end.append(row.name)
                trade_days.append(c)

                pnl = (row.Open - last_price) * position
                trade_pnl.append(pnl)
                trade_ret.append((last_price / row.Open - 1) * 100)

                balance = balance + pnl

                position = 0
                last_signal = "hold"

                c = 0

            # check signal and enter any possible position
            if row.LONG and last_signal != "long":
                last_signal = "long"
                last_price = row.Open
                trade_date_start.append(row.name)
                trade_side.append("long")

                position = int(balance / row.Open)
                cost = position * row.Open
                balance = balance - cost

                c = 0

            elif row.SHORT and last_signal != "short":
                last_signal = "short"
                last_price = row.Open
                trade_date_start.append(row.name)
                trade_side.append("short")

                position = int(balance / row.Open) * -1

                c = 0

            if stop_loss_lvl:
                # check stop loss
                if (
                    last_signal == "long"
                    and (row.Low / last_price - 1) * 100 <= stop_loss_lvl
                ):
                    c = c + 1

                    trade_date_end.append(row.name)
                    trade_days.append(c)

                    stop_loss_price = last_price + round(
                        last_price * (stop_loss_lvl / 100), 4
                    )

                    pnl = (stop_loss_price - last_price) * position
                    trade_pnl.append(pnl)
                    trade_ret.append((stop_loss_price / last_price - 1) * 100)

                    balance = balance + stop_loss_price * position

                    position = 0
                    last_signal = "hold"

                    c = 0

                elif (
                    last_signal == "short"
                    and (last_price / row.Low - 1) * 100 <= stop_loss_lvl
                ):
                    c = c + 1

                    trade_date_end.append(row.name)
                    trade_days.append(c)

                    stop_loss_price = last_price - round(
                        last_price * (stop_loss_lvl / 100), 4
                    )

                    pnl = (stop_loss_price - last_price) * position
                    trade_pnl.append(pnl)
                    trade_ret.append((last_price / stop_loss_price - 1) * 100)

                    balance = balance + pnl

                    position = 0
                    last_signal = "hold"

                    c = 0

            # compute market value and count days for any possible poisition
            if last_signal == "hold":
                market_value = balance
            elif last_signal == "long":
                c = c + 1
                market_value = position * row.Close + balance
            else:
                c = c + 1
                market_value = (row.Close - last_price) * position + balance

            cum_value.append(market_value)

        # generate analysis
        # performance over time
        cum_ret_df = pd.DataFrame(cum_value, index=bt_df.index, columns=["CUM_RET"])
        cum_ret_df["CUM_RET"] = (cum_ret_df.CUM_RET / 1000000 - 1) * 100
        cum_ret_df["BUY_HOLD"] = (bt_df.Close / bt_df.Open.iloc[0] - 1) * 100
        cum_ret_df["ZERO"] = 0

        # trade stats
        size = min(len(trade_date_start), len(trade_date_end))

        tarde_dict = {
            "START": trade_date_start[:size],
            "END": trade_date_end[:size],
            "SIDE": trade_side[:size],
            "DAYS": trade_days[:size],
            "PNL": trade_pnl[:size],
            "RET": trade_ret[:size],
        }

        trade_df = pd.DataFrame(tarde_dict)

        num_trades = trade_df.groupby("SIDE").count()[["START"]]
        num_trades_win = trade_df[trade_df.PNL > 0].groupby("SIDE").count()[["START"]]

        avg_days = trade_df.groupby("SIDE").mean()[["DAYS"]]

        avg_ret = trade_df.groupby("SIDE").mean()[["RET"]]
        avg_ret_win = trade_df[trade_df.PNL > 0].groupby("SIDE").mean()[["RET"]]
        avg_ret_loss = trade_df[trade_df.PNL < 0].groupby("SIDE").mean()[["RET"]]

        std_ret = trade_df.groupby("SIDE").std()[["RET"]]

        detail_df = pd.concat(
            [
                num_trades,
                num_trades_win,
                avg_days,
                avg_ret,
                avg_ret_win,
                avg_ret_loss,
                std_ret,
            ],
            axis=1,
            sort=False,
        )

        detail_df.columns = [
            "NUM_TRADES",
            "NUM_TRADES_WIN",
            "AVG_DAYS",
            "AVG_RET",
            "AVG_RET_WIN",
            "AVG_RET_LOSS",
            "STD_RET",
        ]

        detail_df.round(2)

        # max drawdown
        mv_df = pd.DataFrame(cum_value, index=bt_df.index, columns=["MV"])

        days = len(mv_df)

        roll_max = mv_df.MV.rolling(window=days, min_periods=1).max()
        drawdown_val = mv_df.MV - roll_max
        drawdown_pct = (mv_df.MV / roll_max - 1) * 100

        # return all stats
        return {
            "cum_ret_df": cum_ret_df,
            "max_drawdown": {
                "value": round(drawdown_val.min(), 0),
                "pct": round(drawdown_pct.min(), 2),
            },
            "trade_stats": detail_df,
        }
    
    def strategy_MA(self, df, **kwargs):
        n = kwargs.get("n", 50)
        ma_type = kwargs.get("ma_type", "sma")
        ma_type = ma_type.strip().lower()
        data = df.copy()

        if ma_type == "sma":
            sma = ta.trend.SMAIndicator(data.Close, n)
            data["MA"] = sma.sma_indicator().round(4)
        elif ma_type == "ema":
            ema = ta.trend.EMAIndicator(data.Close, n)
            data["MA"] = ema.ema_indicator().round(4)

        data["CLOSE_PREV"] = data.Close.shift(1)

        data["LONG"] = (data.Close > data.MA) & (data.CLOSE_PREV <= data.MA)
        data["EXIT_LONG"] = (data.Close < data.MA) & (data.CLOSE_PREV >= data.MA)

        data["SHORT"] = (data.Close < data.MA) & (data.CLOSE_PREV >= data.MA)
        data["EXIT_SHORT"] = (data.Close > data.MA) & (data.CLOSE_PREV <= data.MA)

        data.LONG = data.LONG.shift(1)
        data.EXIT_LONG = data.EXIT_LONG.shift(1)
        data.SHORT = data.SHORT.shift(1)
        data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

        return data
    
    def strategy_MACD(self, df, **kwargs):
        n_slow = kwargs.get("n_slow", 26)
        n_fast = kwargs.get("n_fast", 12)
        n_sign = kwargs.get("n_sign", 9)
        data = df.copy()

        macd = ta.trend.MACD(data.Close, n_slow, n_fast, n_sign)

        data["MACD_DIFF"] = macd.macd_diff().round(4)
        data["MACD_DIFF_PREV"] = data.MACD_DIFF.shift(1)

        data["LONG"] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)
        data["EXIT_LONG"] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)

        data["SHORT"] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)
        data["EXIT_SHORT"] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)

        data.LONG = data.LONG.shift(1)
        data.EXIT_LONG = data.EXIT_LONG.shift(1)
        data.SHORT = data.SHORT.shift(1)
        data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

        return data
    
    def strategy_RSI(df, **kwargs):
        n = kwargs.get("n", 14)
        data = df.copy()

        rsi = ta.momentum.RSIIndicator(data.Close, n)

        data["RSI"] = rsi.rsi().round(4)
        data["RSI_PREV"] = data.RSI.shift(1)

        data["LONG"] = (data.RSI > 30) & (data.RSI_PREV <= 30)
        data["EXIT_LONG"] = (data.RSI < 70) & (data.RSI_PREV >= 70)

        data["SHORT"] = (data.RSI < 70) & (data.RSI_PREV >= 70)
        data["EXIT_SHORT"] = (data.RSI > 30) & (data.RSI_PREV <= 30)

        data.LONG = data.LONG.shift(1)
        data.EXIT_LONG = data.EXIT_LONG.shift(1)
        data.SHORT = data.SHORT.shift(1)
        data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

        return data

    def strategy_Stochastic_fast(self, df, **kwargs):
        k = kwargs.get("k", 20)
        d = kwargs.get("d", 5)
        data = df.copy()

        sto = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, k, d)

        data["K"] = sto.stoch().round(4)
        data["D"] = sto.stoch_signal().round(4)
        data["DIFF"] = data["K"] - data["D"]
        data["DIFF_PREV"] = data.DIFF.shift(1)

        data["LONG"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
        data["EXIT_LONG"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

        data["SHORT"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
        data["EXIT_SHORT"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

        data.LONG = data.LONG.shift(1)
        data.EXIT_LONG = data.EXIT_LONG.shift(1)
        data.SHORT = data.SHORT.shift(1)
        data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

        return data

    def strategy_Stochastic_slow(self, df, **kwargs):
        k = kwargs.get("k", 20)
        d = kwargs.get("d", 5)
        dd = kwargs.get("dd", 3)
        data = df.copy()

        sto = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, k, d)

        data["K"] = sto.stoch().round(4)
        data["D"] = sto.stoch_signal().round(4)

        ma = ta.trend.SMAIndicator(data.D, dd)
        data["DD"] = ma.sma_indicator().round(4)

        data["DIFF"] = data["D"] - data["DD"]
        data["DIFF_PREV"] = data.DIFF.shift(1)

        data["LONG"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
        data["EXIT_LONG"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

        data["SHORT"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
        data["EXIT_SHORT"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

        data.LONG = data.LONG.shift(1)
        data.EXIT_LONG = data.EXIT_LONG.shift(1)
        data.SHORT = data.SHORT.shift(1)
        data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

        return data
    
    def strategy_Ichmoku(df, **kwargs):
        n_conv = kwargs.get("n_conv", 9)
        n_base = kwargs.get("n_base", 26)
        n_span_b = kwargs.get("n_span_b", 26)
        data = df.copy()

        ichmoku = ta.trend.IchimokuIndicator(data.High, data.Low, n_conv, n_base, n_span_b)

        data["BASE"] = ichmoku.ichimoku_base_line().round(4)
        data["CONV"] = ichmoku.ichimoku_conversion_line().round(4)

        data["DIFF"] = data["CONV"] - data["BASE"]
        data["DIFF_PREV"] = data.DIFF.shift(1)

        data["LONG"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
        data["EXIT_LONG"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

        data["SHORT"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
        data["EXIT_SHORT"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

        data.LONG = data.LONG.shift(1)
        data.EXIT_LONG = data.EXIT_LONG.shift(1)
        data.SHORT = data.SHORT.shift(1)
        data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

        return data
        
    def apply_technical_indicators(self):
        if self.data is not None:
            self.data['SMA'] = ta.trend.sma_indicator(self.data['Close'], window=20)
        else:
            print("Data not fetched. Please call fetch_data() first.")
    
    def apply_keltner_channel_strategy(self):
        if self.data is not None:
            self.data["CLOSE_PREV"] = self.data.Close.shift(1)

            k_band = ta.volatility.KeltnerChannel(self.data.High, self.data.Low, self.data.Close, 10)

            self.data["K_BAND_UB"] = k_band.keltner_channel_hband().round(4)
            self.data["K_BAND_LB"] = k_band.keltner_channel_lband().round(4)

            self.data["LONG"] = (self.data.Close <= self.data.K_BAND_LB) & (self.data.CLOSE_PREV > self.data.K_BAND_LB)
            self.data["EXIT_LONG"] = (self.data.Close >= self.data.K_BAND_UB) & (self.data.CLOSE_PREV < self.data.K_BAND_UB)

            self.data["SHORT"] = (self.data.Close >= self.data.K_BAND_UB) & (self.data.CLOSE_PREV < self.data.K_BAND_UB)
            self.data["EXIT_SHORT"] = (self.data.Close <= self.data.K_BAND_LB) & (self.data.CLOSE_PREV > self.data.K_BAND_LB)

            # Adjusting signal markers to avoid lookahead bias
            self.data.LONG = self.data.LONG.shift(1)
            self.data.EXIT_LONG = self.data.EXIT_LONG.shift(1)
            self.data.SHORT = self.data.SHORT.shift(1)
            self.data.EXIT_SHORT = self.data.EXIT_SHORT.shift(1)
        else:
            print("Data not available. Please ensure data is fetched.")
    
    def display_results(self):
        if self.data is not None:
            clear_output(wait=True)
            print(self.data[["LONG", "EXIT_LONG", "SHORT", "EXIT_SHORT"]].dropna().head())
        else:
            print("No results to display. Please run the strategy first.")

# Example usage, Input: Ticker, Start Date, End Date
ticker = "FSLY"
start_date = "2023-10-23"
end_date = "2024-03-31"

backtester = StockBacktester(ticker, start_date, end_date)
backtester.fetch_data()
backtester.apply_technical_indicators()
backtester.apply_keltner_channel_strategy()
result = backtester.run_stock_ta_backtest(stop_loss_lvl=5)
result["cum_ret_df"].plot(figsize=(15, 5))
print("Cumulative Return:", result["cum_ret_df"].CUM_RET[-1], "%")
print("Max Drawdown:", result["max_drawdown"]["pct"], "%")
print("Trade Stats:", result["trade_stats"])
backtester.display_results()
