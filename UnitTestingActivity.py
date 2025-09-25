import unittest

def add(a, b):
    return a + b

class TestFunction(unittest.TestCase):
    def test_positive_addition(self):
        result = add(39, 12)
        self.assertEqual(result, 51)

    def test_negative_addition(self):
        result = add(-8, -11)
        self.assertEqual(result, -19)

    def test_decimal_addition(self):
        result = add(3.5, 5.5)
        self.assertAlmostEqual(result, 9.0, places=2)

if __name__ == "__main__":
    unittest.main()
