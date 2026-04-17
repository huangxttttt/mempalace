import gc
import os
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import chromadb
import yaml

from mempalace.miner import (
    _extract_doc_text_from_streams,
    mine,
    read_supported_text,
    scan_project,
)


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_docx(path: Path, paragraphs: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(
            f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
            for paragraph in paragraphs
        )
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


def write_pdf(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    content_stream = "BT /F1 12 Tf 72 720 Td " + " T* ".join(
        f"({line}) Tj" for line in lines
    ) + " ET"
    pdf = "\n".join(
        [
            "%PDF-1.4",
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            (
                "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj"
            ),
            "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
            f"5 0 obj << /Length {len(content_stream)} >> stream",
            content_stream,
            "endstream endobj",
        ]
    )
    offsets = [0]
    body_parts = []
    current = len("%PDF-1.4\n")
    for obj in pdf.split("\n")[1:]:
        body_parts.append(obj + "\n")
        offsets.append(current)
        current += len(obj) + 1
    xref_offset = current
    xref = ["xref", "0 6", "0000000000 65535 f "]
    for offset in offsets[1:6]:
        xref.append(f"{offset:010d} 00000 n ")
    trailer = "\n".join(
        [
            "\n".join(xref),
            "trailer << /Size 6 /Root 1 0 R >>",
            f"startxref\n{xref_offset}",
            "%%EOF",
        ]
    )
    path.write_bytes(("%PDF-1.4\n" + "".join(body_parts) + trailer).encode("latin-1"))


def write_eml(path: Path, headers: dict[str, str], body: str, content_type: str = "text/plain"):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value}" for key, value in headers.items()]
    lines.append(f"Content-Type: {content_type}; charset=utf-8")
    lines.append("Content-Transfer-Encoding: 8bit")
    lines.append("")
    lines.append(body)
    path.write_text("\n".join(lines), encoding="utf-8")


def scanned_files(project_root: Path, **kwargs):
    files = scan_project(str(project_root), **kwargs)
    return sorted(path.relative_to(project_root).as_posix() for path in files)


def test_project_mining():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        os.makedirs(project_root / "backend")

        write_file(
            project_root / "backend" / "app.py", "def main():\n    print('hello world')\n" * 20
        )
        with open(project_root / "mempalace.yaml", "w") as f:
            yaml.dump(
                {
                    "wing": "test_project",
                    "rooms": [
                        {"name": "backend", "description": "Backend code"},
                        {"name": "general", "description": "General"},
                    ],
                },
                f,
            )

        palace_path = project_root / "palace"
        mine(str(project_root), str(palace_path))

        client = chromadb.PersistentClient(path=str(palace_path))
        col = client.get_collection("mempalace_drawers")
        assert col.count() > 0
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_includes_docx():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        write_docx(project_root / "docs" / "notes.docx", ["hello", "world"])

        assert scanned_files(project_root, respect_gitignore=False) == ["docs/notes.docx"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_scan_project_includes_doc_and_pdf():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        write_file(project_root / "docs" / "notes.doc", "legacy binary placeholder")
        write_pdf(project_root / "docs" / "notes.pdf", ["hello", "world"])

        assert scanned_files(project_root, respect_gitignore=False) == [
            "docs/notes.doc",
            "docs/notes.pdf",
        ]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_scan_project_includes_eml():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        write_eml(
            project_root / "mail" / "notes.eml",
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Launch plan",
                "Date": "Sat, 17 Apr 2026 09:00:00 +0800",
            },
            "We should ship the launch checklist this week.",
        )

        assert scanned_files(project_root, respect_gitignore=False) == ["mail/notes.eml"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_supported_text_extracts_docx_paragraphs():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        docx_path = project_root / "meeting.docx"
        write_docx(docx_path, ["First paragraph", "Second paragraph"])

        content = read_supported_text(docx_path)

        assert "First paragraph" in content
        assert "Second paragraph" in content
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_supported_text_extracts_pdf_text():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        pdf_path = project_root / "meeting.pdf"
        write_pdf(pdf_path, ["First paragraph", "Second paragraph"])

        content = read_supported_text(pdf_path)

        assert "First paragraph" in content
        assert "Second paragraph" in content
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_supported_text_extracts_eml_headers_and_body():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        eml_path = project_root / "meeting.eml"
        write_eml(
            eml_path,
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "Quarterly planning",
                "Date": "Sat, 17 Apr 2026 09:00:00 +0800",
            },
            "We chose PostgreSQL because it fits the deployment model.",
        )

        content = read_supported_text(eml_path)

        assert "Subject: Quarterly planning" in content
        assert "From: alice@example.com" in content
        assert "We chose PostgreSQL" in content
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_read_supported_text_extracts_eml_html_body():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        eml_path = project_root / "meeting-html.eml"
        write_eml(
            eml_path,
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Subject": "HTML update",
                "Date": "Sat, 17 Apr 2026 09:00:00 +0800",
            },
            "<html><body><p>Launch update</p><p>Checklist approved</p></body></html>",
            content_type="text/html",
        )

        content = read_supported_text(eml_path)

        assert "Subject: HTML update" in content
        assert "Launch update" in content
        assert "Checklist approved" in content
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_extract_doc_text_from_streams_handles_utf16_and_ascii():
    content = _extract_doc_text_from_streams(
        [
            "First legacy paragraph".encode("utf-16le"),
            b"\x00\x01Second legacy paragraph\x00\x02",
        ]
    )

    assert "First legacy paragraph" in content
    assert "Second legacy paragraph" in content


def test_read_supported_text_extracts_doc_text_via_olefile(monkeypatch):
    class FakeStream:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

    class FakeOleFile:
        def __init__(self, _path: str):
            self._streams = {
                ("WordDocument",): "First paragraph".encode("utf-16le"),
                ("1Table",): b"\x00Second paragraph\x00",
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def listdir(self):
            return list(self._streams.keys())

        def openstream(self, name):
            return FakeStream(self._streams[tuple(name)])

    monkeypatch.setitem(sys.modules, "olefile", types.SimpleNamespace(OleFileIO=FakeOleFile))

    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        doc_path = project_root / "meeting.doc"
        doc_path.write_bytes(b"placeholder")

        content = read_supported_text(doc_path)

        assert "First paragraph" in content
        assert "Second paragraph" in content
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_project_mining_supports_docx():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        write_docx(
            project_root / "docs" / "brief.docx",
            [
                "Quarterly planning notes and milestones.",
                "We chose PostgreSQL because it fits the deployment model.",
            ],
        )
        with open(project_root / "mempalace.yaml", "w") as f:
            yaml.dump(
                {
                    "wing": "test_project",
                    "rooms": [
                        {"name": "docs", "description": "Documentation"},
                        {"name": "general", "description": "General"},
                    ],
                },
                f,
            )

        palace_path = project_root / "palace"
        mine(str(project_root), str(palace_path))

        client = chromadb.PersistentClient(path=str(palace_path))
        col = client.get_collection("mempalace_drawers")
        docs = col.get(where={"source_file": str(project_root / "docs" / "brief.docx")})
        assert docs["documents"]
        assert "Quarterly planning notes" in docs["documents"][0]
        del docs
        del col
        del client
        gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_project_mining_supports_pdf():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        write_pdf(
            project_root / "docs" / "brief.pdf",
            [
                "Quarterly planning notes and milestones.",
                "We chose PostgreSQL because it fits the deployment model.",
            ],
        )
        with open(project_root / "mempalace.yaml", "w") as f:
            yaml.dump(
                {
                    "wing": "test_project",
                    "rooms": [
                        {"name": "docs", "description": "Documentation"},
                        {"name": "general", "description": "General"},
                    ],
                },
                f,
            )

        palace_path = project_root / "palace"
        mine(str(project_root), str(palace_path))

        client = chromadb.PersistentClient(path=str(palace_path))
        col = client.get_collection("mempalace_drawers")
        docs = col.get(where={"source_file": str(project_root / "docs" / "brief.pdf")})
        assert docs["documents"]
        assert "Quarterly planning notes" in docs["documents"][0]
        del docs
        del col
        del client
        gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_project_mining_supports_eml():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        write_eml(
            project_root / "mail" / "brief.eml",
            {
                "From": "alice@example.com",
                "To": "team@example.com",
                "Subject": "Quarterly planning notes",
                "Date": "Sat, 17 Apr 2026 09:00:00 +0800",
            },
            "We chose PostgreSQL because it fits the deployment model.",
        )
        with open(project_root / "mempalace.yaml", "w") as f:
            yaml.dump(
                {
                    "wing": "test_project",
                    "rooms": [
                        {"name": "mail", "description": "Email"},
                        {"name": "general", "description": "General"},
                    ],
                },
                f,
            )

        palace_path = project_root / "palace"
        mine(str(project_root), str(palace_path))

        client = chromadb.PersistentClient(path=str(palace_path))
        col = client.get_collection("mempalace_drawers")
        docs = col.get(where={"source_file": str(project_root / "mail" / "brief.eml")})
        assert docs["documents"]
        assert "Subject: Quarterly planning notes" in docs["documents"][0]
        assert "We chose PostgreSQL" in docs["documents"][0]
        del docs
        del col
        del client
        gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_scan_project_respects_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "ignored.py\ngenerated/\n")
        write_file(project_root / "src" / "app.py", "print('hello')\n" * 20)
        write_file(project_root / "ignored.py", "print('ignore me')\n" * 20)
        write_file(project_root / "generated" / "artifact.py", "print('artifact')\n" * 20)

        assert scanned_files(project_root) == ["src/app.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_respects_nested_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "*.log\n")
        write_file(project_root / "subrepo" / ".gitignore", "tasks/\n")
        write_file(project_root / "subrepo" / "src" / "main.py", "print('main')\n" * 20)
        write_file(project_root / "subrepo" / "tasks" / "task.py", "print('task')\n" * 20)
        write_file(project_root / "subrepo" / "debug.log", "debug\n" * 20)

        assert scanned_files(project_root) == ["subrepo/src/main.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_allows_nested_gitignore_override():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "*.csv\n")
        write_file(project_root / "subrepo" / ".gitignore", "!keep.csv\n")
        write_file(project_root / "drop.csv", "a,b,c\n" * 20)
        write_file(project_root / "subrepo" / "keep.csv", "a,b,c\n" * 20)

        assert scanned_files(project_root) == ["subrepo/keep.csv"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_allows_gitignore_negation_when_parent_dir_is_visible():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/*\n!generated/keep.py\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root) == ["generated/keep.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_does_not_reinclude_file_from_ignored_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/\n!generated/keep.py\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root) == []
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_disable_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "data/\n")
        write_file(project_root / "data" / "stuff.csv", "a,b,c\n" * 20)

        assert scanned_files(project_root, respect_gitignore=False) == ["data/stuff.csv"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_ignored_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "docs/\n")
        write_file(project_root / "docs" / "guide.md", "# Guide\n" * 20)

        assert scanned_files(project_root, include_ignored=["docs"]) == ["docs/guide.md"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_specific_ignored_file():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root, include_ignored=["generated/keep.py"]) == [
            "generated/keep.py"
        ]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_exact_file_without_known_extension():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "README\n")
        write_file(project_root / "README", "hello\n" * 20)

        assert scanned_files(project_root, include_ignored=["README"]) == ["README"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_include_override_beats_skip_dirs():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)

        assert scanned_files(
            project_root,
            respect_gitignore=False,
            include_ignored=[".pytest_cache"],
        ) == [".pytest_cache/cache.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_skip_dirs_still_apply_without_override():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)
        write_file(project_root / "main.py", "print('main')\n" * 20)

        assert scanned_files(project_root, respect_gitignore=False) == ["main.py"]
    finally:
        shutil.rmtree(tmpdir)
