import unittest
import datetime
import Duration_calculator as dc

class TestDaysUntilDateIso(unittest.TestCase):
    def test_today(self):
        today = datetime.date.today()
        self.assertEqual(dc.days_until_date_iso(today.isoformat()), 0)

    def test_tomorrow(self):
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        self.assertEqual(dc.days_until_date_iso(tomorrow.isoformat()), 1)

    def test_yesterday(self):
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        self.assertEqual(dc.days_until_date_iso(yesterday.isoformat()), -1)

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            dc.days_until_date_iso('20-10-2025')

if __name__ == '__main__':
    unittest.main()