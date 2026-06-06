"""CocoIndex app for indexing codebases."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite, turboquant
from cocoindex.connectors.sqlite import Vec0TableDef
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk, TextPosition
from cocoindex.resources.id import IdGenerator

from .chunking import CHUNKER_REGISTRY
from .file_walk import build_matcher
from .settings import load_project_settings, target_turboquant_index_path
from .shared import (
    CODEBASE_DIR,
    EMBEDDER,
    INDEXING_EMBED_PARAMS,
    SQLITE_DB,
    CodeChunk,
    CodeChunkMetadata,
)

# Chunking configuration
CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 250
CHUNK_OVERLAP = 150
MAX_EMBED_CHARS = 6000

# Chunking splitter (stateless, can be module-level)
splitter = RecursiveSplitter()


def _position_in_chunk(chunk: Chunk, char_offset: int) -> TextPosition:
    prefix = chunk.text[:char_offset]
    line_offset = prefix.count("\n")
    if line_offset:
        column = len(prefix.rsplit("\n", maxsplit=1)[-1]) + 1
    else:
        column = chunk.start.column + char_offset
    return TextPosition(
        byte_offset=chunk.start.byte_offset + len(prefix.encode()),
        char_offset=chunk.start.char_offset + char_offset,
        line=chunk.start.line + line_offset,
        column=column,
    )


def _split_large_chunk(chunk: Chunk) -> list[Chunk]:
    if len(chunk.text) <= MAX_EMBED_CHARS:
        return [chunk]

    chunks: list[Chunk] = []
    step = MAX_EMBED_CHARS - CHUNK_OVERLAP
    start = 0
    text_len = len(chunk.text)
    while start < text_len:
        end = min(start + MAX_EMBED_CHARS, text_len)
        chunks.append(
            Chunk(
                text=chunk.text[start:end],
                start=_position_in_chunk(chunk, start),
                end=_position_in_chunk(chunk, end),
            )
        )
        if end == text_len:
            break
        start += step
    return chunks


def _split_large_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    return [sub_chunk for chunk in chunks for sub_chunk in _split_large_chunk(chunk)]


@coco.fn(memo=True)
async def process_file(
    file: localfs.File,
    table: sqlite.TableTarget[Any],
    vector_index: turboquant.IndexTarget[Any] | None = None,
) -> None:
    """Process a single file: chunk, embed, and store."""
    embedder = coco.use_context(EMBEDDER)
    indexing_params = coco.use_context(INDEXING_EMBED_PARAMS)

    try:
        content = await file.read_text()
    except UnicodeDecodeError:
        return

    if not content.strip():
        return

    project_root = coco.use_context(CODEBASE_DIR)
    suffix = file.file_path.path.suffix
    ps = load_project_settings(project_root)
    ext_lang_map = {f".{lo.ext}": lo.lang for lo in ps.language_overrides}
    language = (
        ext_lang_map.get(suffix)
        or detect_code_language(filename=file.file_path.path.name)
        or "text"
    )

    chunker_registry = coco.use_context(CHUNKER_REGISTRY)
    chunker = chunker_registry.get(suffix)
    if chunker is not None:
        language_override, chunks = chunker(Path(file.file_path.path), content)
        if language_override is not None:
            language = language_override
    else:
        chunks = splitter.split(
            content,
            chunk_size=CHUNK_SIZE,
            min_chunk_size=MIN_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            language=language,
        )
    chunks = _split_large_chunks(chunks)

    id_gen = IdGenerator()

    async def process(chunk: Chunk) -> None:
        chunk_id = await id_gen.next_id(chunk.text)
        embedding = await embedder.embed(chunk.text, **indexing_params)
        metadata = CodeChunkMetadata(
            id=chunk_id,
            file_path=file.file_path.path.as_posix(),
            language=language,
            content=chunk.text,
            start_line=chunk.start.line,
            end_line=chunk.end.line,
        )
        if vector_index is None:
            table.declare_row(
                row=CodeChunk(
                    **metadata.__dict__,
                    embedding=embedding,
                )
            )
        else:
            table.declare_row(row=metadata)
            vector_index.declare_vector(id=chunk_id, vector=embedding)

    await coco.map(process, chunks)


async def _mount_sqlite_vec_table() -> sqlite.TableTarget[CodeChunk]:
    return await sqlite.mount_table_target(
        db=SQLITE_DB,
        table_name="code_chunks_vec",
        table_schema=await sqlite.TableSchema.from_class(
            CodeChunk,
            primary_key=["id"],
        ),
        virtual_table_def=Vec0TableDef(
            partition_key_columns=["language"],
            auxiliary_columns=["file_path", "content", "start_line", "end_line"],
        ),
    )


async def _mount_turboquant_targets(
    project_root: Path,
    bit_width: int,
) -> tuple[sqlite.TableTarget[CodeChunkMetadata], turboquant.IndexTarget[Any]]:
    table = await sqlite.mount_table_target(
        db=SQLITE_DB,
        table_name="code_chunks",
        table_schema=await sqlite.TableSchema.from_class(
            CodeChunkMetadata,
            primary_key=["id"],
        ),
    )
    vector_index = await turboquant.mount_index_target(
        target_turboquant_index_path(project_root),
        bit_width=bit_width,
    )
    return table, vector_index


@coco.fn
async def indexer_main() -> None:
    """Main indexing function - walks files and processes each."""
    project_root = coco.use_context(CODEBASE_DIR)
    ps = load_project_settings(project_root)

    vector_index: turboquant.IndexTarget[Any] | None = None
    if ps.vector_search.backend == "turboquant":
        table, vector_index = await _mount_turboquant_targets(
            project_root,
            ps.vector_search.bit_width,
        )
    else:
        table = await _mount_sqlite_vec_table()

    matcher = build_matcher(
        project_root, ps.include_patterns, ps.exclude_patterns, ps.max_file_size
    )

    files = localfs.walk_dir(
        CODEBASE_DIR,
        recursive=True,
        path_matcher=matcher,
    )

    await coco.mount_each(
        coco.component_subpath(coco.Symbol("process_file")),
        process_file,
        files.items(),
        table,
        vector_index,
    )
