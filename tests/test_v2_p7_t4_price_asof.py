import pandas as pd

from scripts.v2_p7.t4.price import PriceBook


class _Loader:
    def __init__(self, frames):
        self.frames = frames

    def load_clean_daily_by_date(self, day, auto_ensure=False):
        return self.frames.get(day)


def _frame(open_price, close_price):
    return pd.DataFrame(
        [
            {
                "ts_code": "AAA",
                "open": open_price,
                "close": close_price,
                "is_suspended": 0,
                "is_limit_up": 0,
                "is_limit_down": 0,
            }
        ]
    )


def test_last_known_excludes_today_close_for_open_valuation():
    book = PriceBook(
        _Loader(
            {
                "20240102": _frame(10.0, 10.0),
                "20240103": _frame(0.0, 20.0),
            }
        )
    )
    book.begin_day("20240102")
    book.begin_day("20240103", tracked=["AAA"])
    assert book.last_known("AAA") == (10.0, "20240102")
