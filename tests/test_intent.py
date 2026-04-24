import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import unittest
from unittest.mock import patch, MagicMock

from agent.nodes.intent import intent_node


def make_llm_response(country_name, requested_fields, is_valid):
    """Helper to build a mock LLM response."""
    return json.dumps({
        "country_name": country_name,
        "requested_fields": requested_fields,
        "is_valid": is_valid
    })


def mock_llm(content: str):
    """Helper to build a mock ChatGroq instance."""
    mock_instance = MagicMock()
    mock_instance.invoke.return_value.content = content
    return mock_instance


class TestIntentNode(unittest.TestCase):

    @patch("agent.nodes.intent.ChatGroq")
    def test_valid_single_field(self, mock_groq):
        mock_groq.return_value = mock_llm(
            make_llm_response("Germany", ["population"], True)
        )
        result = intent_node({"question": "What is the population of Germany?"})

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["country_name"], "Germany")
        self.assertIn("population", result["requested_fields"])

    @patch("agent.nodes.intent.ChatGroq")
    def test_valid_multiple_fields(self, mock_groq):
        mock_groq.return_value = mock_llm(
            make_llm_response("Brazil", ["capital", "population"], True)
        )
        result = intent_node({"question": "What is the capital and population of Brazil?"})

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["country_name"], "Brazil")
        self.assertIn("capital", result["requested_fields"])
        self.assertIn("population", result["requested_fields"])

    @patch("agent.nodes.intent.ChatGroq")
    def test_invalid_question_no_country(self, mock_groq):
        mock_groq.return_value = mock_llm(
            make_llm_response(None, [], False)
        )
        result = intent_node({"question": "Tell me about dogs"})

        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["country_name"])
        self.assertEqual(result["requested_fields"], [])

    @patch("agent.nodes.intent.ChatGroq")
    def test_hallucinated_fields_are_filtered(self, mock_groq):
        mock_groq.return_value = mock_llm(
            make_llm_response("Japan", ["population", "fake_field"], True)
        )
        result = intent_node({"question": "What is the population of Japan?"})

        self.assertIn("population", result["requested_fields"])
        self.assertNotIn("fake_field", result["requested_fields"])

    @patch("agent.nodes.intent.ChatGroq")
    def test_llm_returns_malformed_json(self, mock_groq):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value.content = "this is not json"
        mock_groq.return_value = mock_instance

        result = intent_node({"question": "What is the capital of France?"})

        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["country_name"])
        self.assertEqual(result["requested_fields"], [])

    @patch("agent.nodes.intent.ChatGroq")
    def test_is_valid_true_but_missing_country(self, mock_groq):
        mock_groq.return_value = mock_llm(
            make_llm_response(None, ["population"], True)
        )
        result = intent_node({"question": "What is the population?"})

        # Secondary validation should override is_valid to False
        self.assertFalse(result["is_valid"])

    @patch("agent.nodes.intent.ChatGroq")
    def test_llm_exception_handled_gracefully(self, mock_groq):
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("Unexpected error")
        mock_groq.return_value = mock_instance

        result = intent_node({"question": "What is the capital of India?"})

        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["country_name"])
        self.assertEqual(result["requested_fields"], [])


if __name__ == "__main__":
    unittest.main()