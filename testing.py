import unittest
from process_xml import get_metadata

class TestMetaParsing(unittest.TestCase):
    def test_stuff(self):
        metadata = get_metadata('billtext/BILLS-118hr1eh.txt')
        self.assertEqual(metadata.get('chamber'), 'House')

if __name__ == '__main__':
    unittest.main()