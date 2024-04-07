import unittest
import pandas as pd
from backtest_all_indicators import prepare_stock_ta_backtest_data, run_stock_ta_backtest

class TestBacktestAllIndicators(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Open': [1, 2, 3, 4, 5],
            'High': [1.5, 2.5, 3.5, 4.5, 5.5],
            'Low': [0.5, 1.5, 2.5, 3.5, 4.5],
            'Close': [1.2, 2.3, 3.4, 4.5, 5.6]
        })
        self.start_date = '2022-01-01'
        self.end_date = '2022-12-31'
        self.func = None  # replace with actual function
        self.param_dict = {}  # replace with actual parameters
        self.stop_loss_lvl = 0.05

    def test_prepare_stock_ta_backtest_data(self):
        result = prepare_stock_ta_backtest_data(self.df, self.start_date, self.end_date, self.func, **self.param_dict)
        self.assertIsInstance(result, pd.DataFrame)

    def test_run_stock_ta_backtest(self):
        bt_df = prepare_stock_ta_backtest_data(self.df, self.start_date, self.end_date, self.func, **self.param_dict)
        result = run_stock_ta_backtest(bt_df, stop_loss_lvl=self.stop_loss_lvl)
        self.assertIsInstance(result, dict)
        self.assertIn('cum_ret_df', result)
        self.assertIn('max_drawdown', result)

if __name__ == '__main__':
    unittest.main()