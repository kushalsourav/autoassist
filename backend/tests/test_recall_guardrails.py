import unittest
from agent.recall_guardrails import ensure_unverified_recall_status


class RecallGuardrailsTests(unittest.TestCase):
    def test_actual_llm_response_claim_is_corrected(self):
        answer = "The open recall does not explain the symptom you're describing."
        safe = ensure_unverified_recall_status(answer, True)
        self.assertNotIn("The open recall", safe)
        self.assertIn("VIN status unverified", safe)
        self.assertIn("**Recall status:**", safe)

    def test_recall_open_statement_is_corrected(self):
        safe = ensure_unverified_recall_status("The recall is still open.", True)
        self.assertNotIn("is still open", safe)
        self.assertIn("unverified", safe)

    def test_preserve_advice_to_check_open_recalls(self):
        safe = ensure_unverified_recall_status("Check your VIN for open recalls.", True)
        self.assertIn("Check your VIN for open recalls.", safe)
        self.assertIn("**Recall status:**", safe)

    def test_no_model_year_data_no_extra_notice(self):
        message = "No recall evidence was retrieved."
        self.assertEqual(ensure_unverified_recall_status(message, False), message)

    def test_notice_not_duplicated(self):
        once = ensure_unverified_recall_status("A relevant campaign is listed.", True)
        twice = ensure_unverified_recall_status(once, True)
        self.assertEqual(once, twice)

    def test_unrepaired_recall_claim_is_corrected(self):
        safe = ensure_unverified_recall_status("This unrepaired recall needs attention.", True)
        self.assertNotIn("unrepaired recall needs", safe)
        self.assertIn("repair status unverified", safe)


if __name__ == '__main__':
    unittest.main()
