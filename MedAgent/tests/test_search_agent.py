import unittest

from medagent.contracts import Evidence, SearchTask, Usage
from medagent.search_agent import ActionResponse, SearchAgent


class SearchAgentLoopTest(unittest.TestCase):
    def make_agent(self, replies, retrievals):
        self.contexts = []
        self.queries = []
        replies = iter(replies)
        retrievals = iter(retrievals)

        def request_action(context):
            self.contexts.append(context)
            return ActionResponse(next(replies), Usage(None, None, None, False))

        def retrieve(query, config, timeout_s, top_k=3):
            self.queries.append(query)
            result = next(retrievals)
            if isinstance(result, Exception):
                raise result
            return result

        agent = SearchAgent(request_action, retrieve, {}, 1)
        task = SearchTask("original question", "planner query", 0, ())
        return agent, task

    def test_search_result_is_shown_to_next_action_and_returned_with_answer(self):
        document = Evidence("retrieved medical passage", "title", 0.8)
        agent, task = self.make_agent(
            ["<search>focused search</search>", "<answer>candidate</answer>"],
            [[document]],
        )

        result = agent.run(task)

        self.assertEqual(self.contexts[0], "planner query")
        self.assertEqual(self.queries, ["focused search"])
        self.assertIn("<information>retrieved medical passage</information>", self.contexts[1])
        self.assertEqual(result.candidate_answer, "candidate")
        self.assertEqual(result.evidence, [document])
        self.assertIsInstance(result.evidence, list)

    def test_two_searches_still_allow_a_final_answer(self):
        first = Evidence("first passage")
        second = Evidence("second passage")
        agent, task = self.make_agent(
            ["<search>first</search>", "<search>second</search>", "<answer>done</answer>"],
            [[first], [second]],
        )

        result = agent.run(task)

        self.assertEqual(self.queries, ["first", "second"])
        self.assertEqual(len(self.contexts), 3)
        self.assertIn("first passage", self.contexts[2])
        self.assertIn("second passage", self.contexts[2])
        self.assertEqual(result.candidate_answer, "done")
        self.assertEqual(result.evidence, [first, second])

    def test_search_continuation_is_returned_with_final_action(self):
        first_action = "<think>first private prefix</think>\n<search> first </search>"
        second_action = "<think>second private prefix</think>\n<search> second </search>"
        final_action = "<think>answer prefix</think>\n<answer>done</answer>"
        agent, task = self.make_agent(
            [first_action + " ignored tail", second_action, final_action + " ignored tail"],
            [[Evidence("first passage")], [Evidence("second passage")]],
        )

        result = agent.run(task)

        first_context = task.query + first_action + "<information>first passage</information>"
        self.assertEqual(self.contexts[1], first_context)
        self.assertEqual(
            self.contexts[2], first_context + second_action + "<information>second passage</information>"
        )
        self.assertEqual(self.queries, ["first", "second"])
        self.assertEqual(result.candidate_answer, "done")
        self.assertEqual(result.context, self.contexts[2] + final_action)

    def test_empty_retrieval_is_explicitly_shown_without_invented_evidence(self):
        agent, task = self.make_agent(
            ["<search>missing</search>", "<answer>uncertain</answer>"],
            [[]],
        )

        result = agent.run(task)

        self.assertIn("<information>No results.</information>", self.contexts[1])
        self.assertEqual(result.evidence, [])

    def test_invalid_action_stops_without_retrieval(self):
        agent, task = self.make_agent(["<search>unclosed"], [])

        result = agent.run(task)

        self.assertEqual(result.candidate_answer, "")
        self.assertEqual(self.queries, [])
        self.assertEqual(len(self.contexts), 1)

    def test_third_search_action_does_not_exceed_retrieval_limit(self):
        agent, task = self.make_agent(
            ["<search>first</search>", "<search>second</search>", "<search>third</search>"],
            [[Evidence("one")], [Evidence("two")]],
        )

        result = agent.run(task)

        self.assertEqual(result.candidate_answer, "")
        self.assertEqual(self.queries, ["first", "second"])
        self.assertEqual(len(self.contexts), 3)
        self.assertEqual(result.context, self.contexts[2])

    def test_retrieval_failure_is_not_reported_as_an_empty_answer(self):
        agent, task = self.make_agent(["<search>first</search>"], [TimeoutError("service timed out")])

        with self.assertRaisesRegex(TimeoutError, "service timed out"):
            agent.run(task)


if __name__ == "__main__":
    unittest.main()
