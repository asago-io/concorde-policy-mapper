from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DOCLING_EXTENSIONS = {
    ".pdf", ".docx", ".html", ".htm", ".pptx", ".xlsx", ".md",
}

_converter_cache: dict[bool, object] = {}


@dataclass(frozen=True)
class ParsedDocument:
    source: str
    content: str
    doc: object | None


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int
    page: int | None = None
    section: str | None = None


def _get_converter(ocr: bool = False):
    from docling.document_converter import DocumentConverter

    if ocr not in _converter_cache:
        if ocr:
            _converter_cache[ocr] = DocumentConverter()
        else:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions(do_ocr=False)
            _converter_cache[ocr] = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
    return _converter_cache[ocr]


def parse_document(path: Path, ocr: bool = False) -> ParsedDocument:
    suffix = path.suffix.lower()

    if suffix in _DOCLING_EXTENSIONS:
        converter = _get_converter(ocr=ocr)
        result = converter.convert(str(path))
        doc = result.document
        content = doc.export_to_markdown()
        if len(content.split()) < 10:
            raise ValueError(
                f"Docling conversion of {path.name} produced near-empty output "
                f"({len(content.split())} words). Check that the file is readable."
            )
        return ParsedDocument(source=str(path), content=content, doc=doc)

    return ParsedDocument(
        source=str(path),
        content=path.read_text(),
        doc=None,
    )


def _chunk_plain_text(text: str, max_tokens: int = 512) -> list[str]:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start:
                end = newline + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def _get_serializer_provider():
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
    )
    from docling_core.transforms.serializer.base import (
        BaseDocSerializer,
        BaseSerializerProvider,
        BaseTableSerializer,
        SerializationResult,
    )
    from docling_core.transforms.serializer.common import create_ser_result
    from docling_core.transforms.serializer.markdown import (
        ImageRefMode,
        MarkdownParams,
    )
    from docling_core.types.doc.document import (
        DOCUMENT_TOKENS_EXPORT_LABELS,
        DoclingDocument,
        TableData,
        TableItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    _CHUNKING_LABELS = DOCUMENT_TOKENS_EXPORT_LABELS - {
        DocItemLabel.DOCUMENT_INDEX,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
    }

    class KVTableSerializer(BaseTableSerializer):
        def serialize(
            self,
            *,
            item: TableItem,
            doc_serializer: BaseDocSerializer,
            doc: DoclingDocument,
            **kwargs,
        ) -> SerializationResult:
            parts: list[SerializationResult] = []

            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                parts.append(cap_res)

            if item.data:
                caption = cap_res.text.strip() if cap_res.text else ""
                table_text = _serialize_table(item.data, caption=caption)
                parts.append(create_ser_result(text=table_text, span_source=item))

            text_res = "\n\n".join([r.text for r in parts if r.text])
            return create_ser_result(text=text_res, span_source=parts)

    class KVChunkingDocSerializer(ChunkingDocSerializer):
        table_serializer: BaseTableSerializer = KVTableSerializer()
        params: MarkdownParams = MarkdownParams(
            labels=_CHUNKING_LABELS,
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="",
            escape_underscores=False,
            escape_html=False,
        )

    class KVSerializerProvider(BaseSerializerProvider):
        def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
            return KVChunkingDocSerializer(doc=doc)

    return KVSerializerProvider()


def _serialize_table(data, caption: str = "") -> str:
    headers: list[str] = []
    rows: dict[int, dict[int, str]] = {}

    for cell in data.table_cells:
        if cell.column_header:
            headers.append(cell.text)
        else:
            for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
                row = rows.setdefault(r, {})
                for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                    row[c] = cell.text

    if not headers:
        headers = [f"col_{i}" for i in range(data.num_cols)]
    elif len(headers) < data.num_cols:
        for i in range(len(headers), data.num_cols):
            headers.append(f"col_{i}")

    comment = f"<!-- table: {caption} -->" if caption else "<!-- table -->"
    parts = [comment, ""]

    for row_idx in sorted(rows):
        row_data = rows[row_idx]
        lines = []
        for col_idx in range(len(headers)):
            val = row_data.get(col_idx, "")
            lines.append(f"{headers[col_idx]}: {val}")
        parts.append("```")
        parts.extend(lines)
        parts.append("```")
        parts.append("")

    parts.append("<!-- /table -->")
    return "\n".join(parts)


def chunk_documents(
    docs: list[ParsedDocument],
    max_tokens: int = 512,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for doc in docs:
        if doc.doc is not None:
            from docling_core.transforms.chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer

            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                max_tokens=max_tokens,
            )
            chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
                serializer_provider=_get_serializer_provider(),
            )
            doc_chunks = list(chunker.chunk(dl_doc=doc.doc))

            if not doc_chunks:
                chunks.append(Chunk(text=doc.content, source=doc.source, index=0))
                continue

            for i, dc in enumerate(doc_chunks):
                page = None
                section = None
                if hasattr(dc, "meta") and dc.meta:
                    pages = []
                    for item in getattr(dc.meta, "doc_items", []) or []:
                        for prov_item in getattr(item, "prov", []) or []:
                            page_no = getattr(prov_item, "page_no", None)
                            if page_no is not None:
                                pages.append(page_no)
                    if pages:
                        page = min(pages)
                    headings = getattr(dc.meta, "headings", []) or []
                    if headings:
                        section = headings[-1]
                chunks.append(Chunk(text=dc.text, source=doc.source, index=i, page=page, section=section))
        else:
            text_chunks = _chunk_plain_text(doc.content, max_tokens=max_tokens)
            for i, text in enumerate(text_chunks):
                chunks.append(Chunk(text=text, source=doc.source, index=i))

    return chunks
