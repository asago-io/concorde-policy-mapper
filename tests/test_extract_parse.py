import tempfile
import types
from pathlib import Path

from asago_policy_mapper.extract.parse import (
    Chunk,
    ParsedDocument,
    _serialize_table,
    chunk_documents,
    parse_document,
)


def test_parsed_document_fields():
    doc = ParsedDocument(source="test.md", content="hello world", doc=None)
    assert doc.source == "test.md"
    assert doc.content == "hello world"
    assert doc.doc is None


def test_chunk_fields():
    chunk = Chunk(text="some text", source="test.md", index=0)
    assert chunk.text == "some text"
    assert chunk.index == 0


def test_parse_plain_text():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("This is a plain text policy document.\nIt has multiple lines.")
        f.flush()
        result = parse_document(Path(f.name))
    assert "plain text policy document" in result.content
    assert result.doc is None
    assert result.source == f.name


def test_parse_markdown():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(
            "# Policy\n\nThis is a markdown policy with enough words to pass"
            " the minimum threshold for docling conversion output."
        )
        f.flush()
        result = parse_document(Path(f.name))
    assert "Policy" in result.content
    assert result.doc is not None


def test_chunk_plain_text_documents():
    docs = [
        ParsedDocument(source="a.txt", content="Short document.", doc=None),
        ParsedDocument(source="b.txt", content="Another short document.", doc=None),
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 2
    assert chunks[0].source == "a.txt"
    assert chunks[0].index == 0
    assert chunks[1].source == "b.txt"
    assert chunks[1].index == 0


def test_chunk_preserves_content():
    content = "A " * 500
    docs = [ParsedDocument(source="big.txt", content=content, doc=None)]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 1
    joined = " ".join(c.text for c in chunks)
    assert "A" in joined


def test_chunk_has_provenance_fields():
    chunk = Chunk(
        text="Some policy text.",
        source="policy.pdf",
        index=0,
        page=3,
        section="Section 4: Oversight",
    )
    assert chunk.page == 3
    assert chunk.section == "Section 4: Oversight"


def test_chunk_provenance_defaults_to_none():
    chunk = Chunk(text="Text.", source="doc.md", index=0)
    assert chunk.page is None
    assert chunk.section is None


# ---------------------------------------------------------------------------
# _serialize_table tests
# ---------------------------------------------------------------------------


def _make_cell(text, column_header=False, start_row=0, end_row=1, start_col=0, end_col=1):
    return types.SimpleNamespace(
        text=text,
        column_header=column_header,
        start_row_offset_idx=start_row,
        end_row_offset_idx=end_row,
        start_col_offset_idx=start_col,
        end_col_offset_idx=end_col,
    )


def _make_table_data(cells, num_cols):
    return types.SimpleNamespace(table_cells=cells, num_cols=num_cols)


def test_serialize_table_basic():
    cells = [
        _make_cell("Name", column_header=True),
        _make_cell("Age", column_header=True),
        _make_cell("Alice", start_row=0, end_row=1, start_col=0, end_col=1),
        _make_cell("30", start_row=0, end_row=1, start_col=1, end_col=2),
    ]
    data = _make_table_data(cells, num_cols=2)
    result = _serialize_table(data)

    assert "Name: Alice" in result
    assert "Age: 30" in result
    assert result.count("```") == 2  # one opening + one closing for the single row


def test_serialize_table_with_caption():
    cells = [
        _make_cell("Col", column_header=True),
        _make_cell("val", start_row=0, end_row=1, start_col=0, end_col=1),
    ]
    data = _make_table_data(cells, num_cols=1)
    result = _serialize_table(data, caption="My Caption")

    assert "<!-- table: My Caption -->" in result


def test_serialize_table_no_caption():
    cells = [
        _make_cell("Col", column_header=True),
        _make_cell("val", start_row=0, end_row=1, start_col=0, end_col=1),
    ]
    data = _make_table_data(cells, num_cols=1)
    result = _serialize_table(data)

    assert result.startswith("<!-- table -->")
    assert "<!-- table:" not in result


def test_serialize_table_no_headers_generates_col_names():
    cells = [
        _make_cell("a", start_row=0, end_row=1, start_col=0, end_col=1),
        _make_cell("b", start_row=0, end_row=1, start_col=1, end_col=2),
    ]
    data = _make_table_data(cells, num_cols=2)
    result = _serialize_table(data)

    assert "col_0: a" in result
    assert "col_1: b" in result


def test_serialize_table_fewer_headers_than_cols():
    cells = [
        _make_cell("OnlyHeader", column_header=True),
        _make_cell("v0", start_row=0, end_row=1, start_col=0, end_col=1),
        _make_cell("v1", start_row=0, end_row=1, start_col=1, end_col=2),
        _make_cell("v2", start_row=0, end_row=1, start_col=2, end_col=3),
    ]
    data = _make_table_data(cells, num_cols=3)
    result = _serialize_table(data)

    assert "OnlyHeader: v0" in result
    assert "col_1: v1" in result
    assert "col_2: v2" in result


def test_serialize_table_multirow():
    cells = [
        _make_cell("X", column_header=True),
        _make_cell("r0", start_row=0, end_row=1, start_col=0, end_col=1),
        _make_cell("r1", start_row=1, end_row=2, start_col=0, end_col=1),
        _make_cell("r2", start_row=2, end_row=3, start_col=0, end_col=1),
    ]
    data = _make_table_data(cells, num_cols=1)
    result = _serialize_table(data)

    # 3 rows = 3 code blocks = 6 triple-backtick fences
    assert result.count("```") == 6
    assert "X: r0" in result
    assert "X: r1" in result
    assert "X: r2" in result


def test_serialize_table_span_cell():
    cells = [
        _make_cell("A", column_header=True),
        _make_cell("B", column_header=True),
        # cell spanning columns 0 and 1
        _make_cell("wide", start_row=0, end_row=1, start_col=0, end_col=2),
    ]
    data = _make_table_data(cells, num_cols=2)
    result = _serialize_table(data)

    assert "A: wide" in result
    assert "B: wide" in result


def test_serialize_table_missing_cell():
    cells = [
        _make_cell("A", column_header=True),
        _make_cell("B", column_header=True),
        # only column 0 has data; column 1 is missing
        _make_cell("present", start_row=0, end_row=1, start_col=0, end_col=1),
    ]
    data = _make_table_data(cells, num_cols=2)
    result = _serialize_table(data)

    assert "A: present" in result
    assert "B: " in result
