from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from medagent.contracts import TraceEvent, Usage
from medagent.trace import TraceStore, redact, replay


class RedactTest(unittest.TestCase):
    def test_keeps_only_existing_allowed_top_level_fields(self):
        payload = {
            "run_id": "run-1",
            "state_after": "VERIFIED",
            "api_key": "secret",
            "hidden_reasoning": "private",
        }

        result = redact(payload, ("run_id", "state_after", "missing"))

        self.assertEqual(
            result,
            {"run_id": "run-1", "state_after": "VERIFIED"},
        )

    def test_does_not_mutate_input(self):
        payload = {"run_id": "run-1", "nested": {"value": 1}}
        original = copy.deepcopy(payload)

        redact(payload, ("run_id",))

        self.assertEqual(payload, original)

    def test_empty_allowlist_returns_empty_dict(self):
        self.assertEqual(redact({"run_id": "run-1"}, ()), {})


class TraceEventTest(unittest.TestCase):
    def make_event(self) -> TraceEvent:
        return TraceEvent(
            run_id="run-1",
            sequence="1",
            actor="Verifier",
            state_before="VERIFYING",
            state_after="VERIFIED",
            call_ids="call-1",
            usage=Usage(10, 5, 15, True),
            input_refs=("search-result-1",),
            output_refs=("verification-1",),
            error=None,
        )

    def test_contains_documented_audit_fields(self):
        event = self.make_event()

        self.assertEqual(event.sequence, "1")
        self.assertEqual(event.call_ids, "call-1")
        self.assertEqual(event.state_before, "VERIFYING")
        self.assertEqual(event.state_after, "VERIFIED")
        self.assertEqual(event.usage.total_tokens, 15)

    def test_is_frozen(self):
        event = self.make_event()
        with self.assertRaises(FrozenInstanceError):
            event.sequence = 2  # type: ignore[misc]


class TraceStoreStartTest(unittest.TestCase):
    def test_creates_run_directory_and_preserves_manifest(self):
        manifest = {
            "run_id": "run-001",
            "model": "假模型",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            store = TraceStore(root)

            store.start(manifest)

            self.assertEqual(store.run_dir, root / "run-001")
            saved = json.loads(
                (store.run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved, manifest)

    def test_existing_run_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run-001"
            run_dir.mkdir()
            existing = run_dir / "manifest.json"
            existing.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                TraceStore(Path(directory)).start({"run_id": "run-001"})

            self.assertEqual(existing.read_text(encoding="utf-8"), "existing")

    def test_rejects_invalid_manifest_and_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(TypeError):
                TraceStore(root).start([])  # type: ignore[arg-type]

            for manifest in ({}, {"run_id": None}, {"run_id": "   "}):
                with self.subTest(manifest=manifest):
                    with self.assertRaises(ValueError):
                        TraceStore(root).start(manifest)

    def test_same_store_cannot_start_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            store.start({"run_id": "run-001"})

            with self.assertRaises(RuntimeError):
                store.start({"run_id": "run-002"})

            self.assertFalse((Path(directory) / "run-002").exists())

    def test_unserializable_manifest_fails_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaises(TypeError):
                TraceStore(root).start({"run_id": "run-001", "bad": object()})

            self.assertFalse((root / "run-001").exists())


class TraceStoreAppendTest(unittest.TestCase):
    def make_event(self, sequence: str = "1") -> TraceEvent:
        return TraceEvent(
            run_id="run-001",
            sequence=sequence,
            actor="Verifier",
            state_before="VERIFYING",
            state_after="VERIFIED",
            call_ids="call-1",
            usage=Usage(10, 5, 15, True),
            input_refs=("search-result-1",),
            output_refs=("verification-1",),
            error=None,
        )

    def test_appends_two_json_lines_with_nested_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            store.start({"run_id": "run-001"})

            store.append(self.make_event("1"))
            store.append(self.make_event("2"))

            lines = (store.run_dir / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 2)
            first, second = map(json.loads, lines)
            self.assertEqual(first["sequence"], "1")
            self.assertEqual(first["usage"]["total_tokens"], 15)
            self.assertEqual(second["sequence"], "2")

    def test_requires_start_and_trace_event(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            with self.assertRaises(RuntimeError):
                store.append(self.make_event())

            store.start({"run_id": "run-001"})
            with self.assertRaises(TypeError):
                store.append({})  # type: ignore[arg-type]

    def test_rejects_event_for_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            store.start({"run_id": "run-001"})
            event = TraceEvent(
                **{**self.make_event().__dict__, "run_id": "run-002"}
            )

            with self.assertRaisesRegex(ValueError, "run_id"):
                store.append(event)

            self.assertFalse((store.run_dir / "events.jsonl").exists())

    def test_rejects_duplicate_sequence_without_writing_again(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            store.start({"run_id": "run-001"})
            event = self.make_event()
            store.append(event)

            with self.assertRaisesRegex(ValueError, "sequence"):
                store.append(event)

            lines = (store.run_dir / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 1)


class TraceStoreFinishTest(unittest.TestCase):
    def test_writes_answer_and_summary_json_without_changing_events(self):
        result = {
            "answer": {"text": "医学回答", "citations": ["evidence-1"]},
            "summary": {"status": "COMPLETED", "attempts": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            store.start({"run_id": "run-001"})
            events_path = store.run_dir / "events.jsonl"
            events_path.write_text('{"sequence":"1"}\n', encoding="utf-8")
            original_events = events_path.read_bytes()

            store.finish(result)

            answer = json.loads(
                (store.run_dir / "answer.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (store.run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(answer, result["answer"])
            self.assertEqual(summary, result["summary"])
            self.assertEqual(events_path.read_bytes(), original_events)

    def test_requires_start_dictionary_and_both_structured_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            with self.assertRaises(RuntimeError):
                store.finish({"answer": {}, "summary": {}})

            store.start({"run_id": "run-001"})
            with self.assertRaises(TypeError):
                store.finish([])  # type: ignore[arg-type]

            for result in ({}, {"answer": {}}, {"summary": {}}):
                with self.subTest(result=result):
                    with self.assertRaises(ValueError):
                        store.finish(result)

            for result in (
                {"answer": "text", "summary": {}},
                {"answer": {}, "summary": "text"},
            ):
                with self.subTest(result=result):
                    with self.assertRaises(TypeError):
                        store.finish(result)

    def test_unserializable_result_creates_neither_output(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            store.start({"run_id": "run-001"})

            with self.assertRaises(TypeError):
                store.finish({"answer": {}, "summary": {"bad": object()}})

            self.assertFalse((store.run_dir / "answer.json").exists())
            self.assertFalse((store.run_dir / "summary.json").exists())

    def test_second_finish_does_not_overwrite_first_result(self):
        first = {"answer": {"text": "first"}, "summary": {"status": "ok"}}
        second = {"answer": {"text": "second"}, "summary": {"status": "bad"}}
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(Path(directory))
            store.start({"run_id": "run-001"})
            store.finish(first)

            with self.assertRaises(FileExistsError):
                store.finish(second)

            answer = json.loads(
                (store.run_dir / "answer.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (store.run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(answer, first["answer"])
            self.assertEqual(summary, first["summary"])


class ReplayTest(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        run_dir = root / "run-001"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": "run-001", "model": "fake"}),
            encoding="utf-8",
        )
        return run_dir

    def test_replays_completed_run_in_recorded_event_order(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_run(Path(directory))
            events = [
                {"run_id": "run-001", "sequence": "1", "actor": "SearchAgent"},
                {"run_id": "run-001", "sequence": "2", "actor": "Verifier"},
            ]
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            answer = {"text": "医学回答"}
            summary = {"status": "COMPLETED"}
            (run_dir / "answer.json").write_text(
                json.dumps(answer), encoding="utf-8"
            )
            (run_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            result = replay(run_dir)

            self.assertEqual(result.manifest["run_id"], "run-001")
            self.assertEqual(result.events, tuple(events))
            self.assertEqual(result.answer, answer)
            self.assertEqual(result.summary, summary)

    def test_replays_interrupted_run_without_fabricating_final_result(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_run(Path(directory))

            result = replay(run_dir)

            self.assertEqual(result.events, ())
            self.assertIsNone(result.answer)
            self.assertIsNone(result.summary)

    def test_rejects_incomplete_final_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self.make_run(Path(directory))
            (run_dir / "answer.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                replay(run_dir)


if __name__ == "__main__":
    unittest.main()
