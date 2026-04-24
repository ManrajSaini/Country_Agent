import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import unittest
from unittest.mock import patch, MagicMock

from agent.graph import build_graph

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

INITIAL_STATE = {
    "question": "",
    "country_name": None,
    "requested_fields": [],
    "is_valid": False,
    "raw_country_data": None,
    "tool_error": None,
    "final_answer": None,
}


def mock_intent_response(country, fields, is_valid):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value.content = json.dumps({
        "country_name": country,
        "requested_fields": fields,
        "is_valid": is_valid
    })
    return mock_instance


def mock_synthesize_response(answer: str):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value.content = answer
    return mock_instance


class TestGraph(unittest.TestCase):

    @patch("agent.nodes.synthesize.ChatGroq")
    @patch("agent.tools.countries_api.requests.get")
    @patch("agent.nodes.intent.ChatGroq")
    def test_happy_path(self, mock_intent_groq, mock_requests, mock_synth_groq):
        # Mock intent
        mock_intent_groq.return_value = mock_intent_response(
            "Germany", ["population"], True
        )

        # Mock API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [MOCK_COUNTRY_RESPONSE]
        mock_requests.return_value = mock_response

        # Mock synthesis
        mock_synth_groq.return_value = mock_synthesize_response(
            "The population of Germany is approximately 83 million."
        )

        graph = build_graph()
        result = graph.invoke({
            **INITIAL_STATE,
            "question": "What is the population of Germany?"
        })

        self.assertIsNotNone(result["final_answer"])
        self.assertEqual(result["country_name"], "Germany")
        self.assertIsNone(result["tool_error"])

    @patch("agent.nodes.synthesize.ChatGroq")
    @patch("agent.nodes.intent.ChatGroq")
    def test_invalid_question_short_circuits_tool(self, mock_intent_groq, mock_synth_groq):
        # Mock intent returning invalid
        mock_intent_groq.return_value = mock_intent_response(
            None, [], False
        )

        # Mock synthesis
        mock_synth_groq.return_value = mock_synthesize_response(
            "I couldn't understand your question. Please ask about a specific country."
        )

        graph = build_graph()

        with patch("agent.tools.countries_api.requests.get") as mock_requests:
            result = graph.invoke({
                **INITIAL_STATE,
                "question": "Tell me about dogs"
            })

            # Tool node should never have been called
            mock_requests.assert_not_called()

        self.assertIsNotNone(result["final_answer"])
        self.assertFalse(result["is_valid"])

    @patch("agent.nodes.synthesize.ChatGroq")
    @patch("agent.tools.countries_api.requests.get")
    @patch("agent.nodes.intent.ChatGroq")
    def test_country_not_found(self, mock_intent_groq, mock_requests, mock_synth_groq):
        # Mock intent
        mock_intent_groq.return_value = mock_intent_response(
            "Narnia", ["population"], True
        )

        # Mock 404 response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests.return_value = mock_response

        # Mock synthesis
        mock_synth_groq.return_value = mock_synthesize_response(
            "I couldn't find any information about Narnia."
        )

        graph = build_graph()
        result = graph.invoke({
            **INITIAL_STATE,
            "question": "What is the population of Narnia?"
        })

        self.assertIsNotNone(result["final_answer"])
        self.assertIsNotNone(result["tool_error"])
        self.assertIn("Narnia", result["tool_error"])

    @patch("agent.nodes.synthesize.ChatGroq")
    @patch("agent.tools.countries_api.requests.get")
    @patch("agent.nodes.intent.ChatGroq")
    def test_multiple_fields(self, mock_intent_groq, mock_requests, mock_synth_groq):
        mock_intent_groq.return_value = mock_intent_response(
            "Germany", ["population", "capital", "currencies"], True
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [MOCK_COUNTRY_RESPONSE]
        mock_requests.return_value = mock_response

        mock_synth_groq.return_value = mock_synthesize_response(
            "Germany has a population of 83 million, capital Berlin, and uses the Euro."
        )

        graph = build_graph()
        result = graph.invoke({
            **INITIAL_STATE,
            "question": "What is the population, capital and currency of Germany?"
        })

        self.assertIsNotNone(result["final_answer"])
        self.assertEqual(len(result["requested_fields"]), 3)
        self.assertIsNone(result["tool_error"])


if __name__ == "__main__":
    unittest.main()