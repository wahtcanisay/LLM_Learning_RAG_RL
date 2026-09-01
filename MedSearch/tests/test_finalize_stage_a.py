from MedSearch.tools.finalize_stage_a import PROJECT_ROOT, finalize_rows, update_manifest


def test_project_root_points_to_medsearch_directory():
    assert PROJECT_ROOT.name == "MedSearch"
    assert (PROJECT_ROOT / "PROJECT_BOOK.md").exists()


def record(row_id=None, group="single", index=0, text="ok"):
    row = {
        "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "answer"}],
        "mixture_group": group,
        "provenance": {"index": index},
    }
    if row_id is not None:
        row["id"] = row_id
    return row


def test_reasoning_id_is_stable_from_provenance_index():
    rows = finalize_rows([record(group="reasoning", index=17)], set(), [])
    assert rows[0]["id"] == "medical_o1_17"


def test_bad_general_record_is_replaced_without_changing_count():
    rows = [record("bad", group="general"), record("keep")]
    replacement = record("replacement", group="general")
    result = finalize_rows(rows, {"bad"}, [replacement])
    assert [row["id"] for row in result] == ["keep", "replacement"]


def test_rejects_replacement_with_unicode_replacement_character():
    rows = [record("bad", group="general")]
    replacement = record("replacement", group="general", text="broken \ufffd text")
    try:
        finalize_rows(rows, {"bad"}, [replacement])
    except ValueError as error:
        assert "U+FFFD" in str(error)
    else:
        raise AssertionError("corrupt replacement was accepted")


def test_manifest_records_finalization_policy():
    manifest = update_manifest({"seed": 7})
    assert manifest["seed"] == 7
    assert manifest["finalization"]["reasoning_id_format"] == "medical_o1_{provenance.index}"
    assert manifest["finalization"]["corrupt_general_replacements"] == 2
