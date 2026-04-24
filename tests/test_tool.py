import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest.mock import patch, MagicMock
import requests

from agent.tools.countries_api import fetch_country_data, extract_fields

MOCK_COUNTRY_RESPONSE = {
    "name": {"common": "Germany"},
    "population": 83240525,
    "capital": ["Berlin"],
    "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
    "languages": {"deu": "German"},
    "area": 357114.0,
    "region": "Europe",
    "subregion": "Western Europe",
    "flags": {"png": "https://flagcdn.com/w320/de.png"},
    "borders": ["AUT", "BEL", "CZE"],
    "timezones": ["UTC+01:00"],
}


class TestFetchCountryData(unittest.TestCase):

    @patch("agent.tools.countries_api.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [MOCK_COUNTRY_RESPONSE]
        mock_get.return_value = mock_response

        data, error = fetch_country_data("Germany")

        self.assertIsNotNone(data)
        self.assertIsNone(error)
        self.assertEqual(data["population"], 83240525)

    @patch("agent.tools.countries_api.requests.get")
    def test_country_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        data, error = fetch_country_data("Narnia")

        self.assertIsNone(data)
        self.assertIsNotNone(error)
        self.assertIn("Narnia", error)

    @patch("agent.tools.countries_api.requests.get")
    def test_timeout(self, mock_get):
        mock_get.side_effect = requests.Timeout

        data, error = fetch_country_data("Germany")

        self.assertIsNone(data)
        self.assertIsNotNone(error)
        self.assertIn("timed out", error.lower())

    @patch("agent.tools.countries_api.requests.get")
    def test_request_exception(self, mock_get):
        mock_get.side_effect = requests.RequestException("Network error")

        data, error = fetch_country_data("Germany")

        self.assertIsNone(data)
        self.assertIsNotNone(error)


class TestExtractFields(unittest.TestCase):

    def test_extract_population(self):
        result = extract_fields(MOCK_COUNTRY_RESPONSE, ["population"])
        self.assertEqual(result["population"], 83240525)

    def test_extract_capital(self):
        result = extract_fields(MOCK_COUNTRY_RESPONSE, ["capital"])
        self.assertEqual(result["capital"], "Berlin")

    def test_extract_currencies(self):
        result = extract_fields(MOCK_COUNTRY_RESPONSE, ["currencies"])
        self.assertIn("EUR", result["currencies"])
        self.assertEqual(result["currencies"]["EUR"], "Euro")

    def test_extract_languages(self):
        result = extract_fields(MOCK_COUNTRY_RESPONSE, ["languages"])
        self.assertIn("German", result["languages"])

    def test_extract_multiple_fields(self):
        result = extract_fields(MOCK_COUNTRY_RESPONSE, ["population", "capital", "region"])
        self.assertEqual(result["population"], 83240525)
        self.assertEqual(result["capital"], "Berlin")
        self.assertEqual(result["region"], "Europe")

    def test_unknown_field_returns_none(self):
        result = extract_fields(MOCK_COUNTRY_RESPONSE, ["unknown_field"])
        self.assertIsNone(result["unknown_field"])

    def test_missing_field_in_response_returns_none(self):
        incomplete_data = {"population": 83240525}
        result = extract_fields(incomplete_data, ["population", "capital"])
        self.assertEqual(result["population"], 83240525)
        self.assertIsNone(result["capital"])

    def test_empty_requested_fields(self):
        result = extract_fields(MOCK_COUNTRY_RESPONSE, [])
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()