import unittest

def calculate_change(initial, final):
    numerator = float(final) - float(initial)
    denominator = float(initial)
    change = (numerator / denominator) * 100
    return change

class TestCalculateChange(unittest.TestCase):
    def test_positive_change(self):
        self.assertAlmostEqual(calculate_change(50, 75), 50.0)
    
    def test_negative_change(self):
        self.assertAlmostEqual(calculate_change(100, 50), -50.0)
    
    def test_no_change(self):
        self.assertAlmostEqual(calculate_change(30, 30), 0.0)
    
    def test_large_increase(self):
        self.assertAlmostEqual(calculate_change(20, 80), 300.0)
    
    def test_large_decrease(self):
        self.assertAlmostEqual(calculate_change(80, 20), -75.0)

if __name__ == "__main__":
    unittest.main()
