from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cocoindex_code import query as query_mod


def test_turboquant_query_loads_index_without_explicit_codebook_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE code_chunks (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            language TEXT,
            content TEXT,
            start_line INTEGER,
            end_line INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT INTO code_chunks
            (id, file_path, language, content, start_line, end_line)
        VALUES
            (42, 'src/example.py', 'python', 'print(42)', 1, 1)
        """
    )

    index_path = tmp_path / "target_turboquant.tvim"
    index_path.write_bytes(b"exists")
    monkeypatch.setattr(
        query_mod,
        "target_turboquant_index_path",
        lambda project_root: index_path,
    )

    class FakeIndex:
        def search(
            self,
            query_array: Any,
            k: int,
            *,
            allowlist: Any | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            assert query_array.shape == (1, 3)
            assert k == 1
            assert allowlist is None
            return (
                np.asarray([[0.75]], dtype=np.float32),
                np.asarray([[42]], dtype=np.uint64),
            )

    loaded_paths: list[Path] = []
    monkeypatch.setattr(
        query_mod.turboquant,
        "load_index",
        lambda path: loaded_paths.append(path) or FakeIndex(),
    )

    results = query_mod._query_turboquant(
        conn,
        tmp_path,
        4,
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        1,
        0,
        None,
        None,
        None,
    )

    assert loaded_paths == [index_path]
    assert len(results) == 1
    assert results[0].file_path == "src/example.py"
    assert results[0].score == 0.75
