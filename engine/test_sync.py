import importlib.util
import unittest
from pathlib import Path
SPEC = importlib.util.spec_from_file_location("sync", Path(__file__).with_name("sync.py"))
sync = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(sync)

class SyncTests(unittest.TestCase):
    def test_categories(self):
        row = sync.parse_draw({"expect":"123","opencode":"9,9,2"})
        self.assertEqual(row[4:], (20,"大","双","大双"))
    def test_small_odd(self):
        row = sync.parse_draw({"issue":"124","code":"1+2+4"})
        self.assertEqual(row[4:], (7,"小","单","小单"))
    def test_reject_bad_code(self):
        with self.assertRaises(ValueError): sync.parse_draw({"issue":1,"code":"1+2"})
    def test_api_wrappers(self):
        self.assertEqual(sync.extract_rows({"data":{"list":[{"x":1}]}}), [{"x":1}])

if __name__ == "__main__": unittest.main()
