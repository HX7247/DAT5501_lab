import unittest

def calculate_change(initial, final):
    # Convert inputs to float and compute percentage change
    numerator = float(final) - float(initial)      # Difference between final and initial
    denominator = float(initial)                   # Original value
    change = (numerator / denominator) * 100       # Percentage change formula
    return change

class TestCalculateChange(unittest.TestCase):
    def test_positive_change(self):
        # Increase: 50 → 75 = +50%
        self.assertAlmostEqual(calculate_change(50, 75), 50.0)
    
    def test_negative_change(self):
        # Decrease: 100 → 50 = -50%
        self.assertAlmostEqual(calculate_change(100, 50), -50.0)
    
    def test_no_change(self):
        # No change: 30 → 30 = 0%
        self.assertAlmostEqual(calculate_change(30, 30), 0.0)
    
    def test_large_increase(self):
        # Increase: 20 → 80 = +300%
        self.assertAlmostEqual(calculate_change(20, 80), 300.0)
    
    def test_large_decrease(self):
        # Decrease: 80 → 20 = -75%
        self.assertAlmostEqual(calculate_change(80, 20), -75.0)

if __name__ == "__main__":
    unittest.main()   # Run all tests
