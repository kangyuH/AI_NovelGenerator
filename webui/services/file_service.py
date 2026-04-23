# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from typing import Iterable

from utils import clear_file_content, read_file, save_string_to_txt

PROJECT_TEXT_FILES = {
    "architecture": "Novel_architecture.txt",
    "directory": "Novel_directory.txt",
    "character_state": "character_state.txt",
    "global_summary": "global_summary.txt",
    "plot_arcs": "plot_arcs.txt",
}


def safe_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def require_project_path(filepath: str) -> str:
    filepath = (filepath or "").strip()
    if not filepath:
        raise ValueError("请先设置保存路径。")
    return filepath


def ensure_project_dir(filepath: str) -> str:
    filepath = require_project_path(filepath)
    os.makedirs(filepath, exist_ok=True)
    return filepath


def read_project_file(filepath: str, file_key: str) -> str:
    filepath = require_project_path(filepath)
    filename = PROJECT_TEXT_FILES[file_key]
    return read_file(os.path.join(filepath, filename))


def save_project_file(filepath: str, file_key: str, content: str) -> str:
    filepath = ensure_project_dir(filepath)
    filename = os.path.join(filepath, PROJECT_TEXT_FILES[file_key])
    clear_file_content(filename)
    save_string_to_txt(content or "", filename)
    return filename


def chapters_dir(filepath: str) -> str:
    return os.path.join(require_project_path(filepath), "chapters")


def list_chapters(filepath: str) -> list[str]:
    directory = chapters_dir(filepath)
    if not os.path.isdir(directory):
        return []

    chapter_nums = []
    for filename in os.listdir(directory):
        match = re.fullmatch(r"chapter_(\d+)\.txt", filename)
        if match:
            chapter_nums.append(match.group(1))
    return sorted(chapter_nums, key=lambda item: int(item))


def load_chapter(filepath: str, chapter_number: str | int) -> str:
    chapter_number = safe_int(chapter_number, 0)
    if chapter_number <= 0:
        raise ValueError("请选择有效章节。")
    path = os.path.join(chapters_dir(filepath), f"chapter_{chapter_number}.txt")
    return read_file(path)


def save_chapter(filepath: str, chapter_number: str | int, content: str) -> str:
    filepath = ensure_project_dir(filepath)
    chapter_number = safe_int(chapter_number, 0)
    if chapter_number <= 0:
        raise ValueError("请选择有效章节。")

    directory = os.path.join(filepath, "chapters")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"chapter_{chapter_number}.txt")
    clear_file_content(path)
    save_string_to_txt(content or "", path)
    return path


def next_chapter_number(chapters: Iterable[str], current: str, delta: int) -> str:
    chapters = list(chapters)
    if not chapters:
        return ""
    if current not in chapters:
        return chapters[0]
    index = chapters.index(current)
    return chapters[max(0, min(len(chapters) - 1, index + delta))]


def read_uploaded_text(file_obj) -> str:
    if file_obj is None:
        raise ValueError("请先选择文件。")

    path = getattr(file_obj, "name", file_obj)
    path = Path(path)
    if not path.exists():
        raise ValueError(f"文件不存在: {path}")

    if path.suffix.lower() == ".docx":
        from docx import Document

        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "cp936", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def write_temp_utf8_copy(content: str, suffix: str = ".txt") -> str:
    import tempfile

    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=suffix)
    try:
        handle.write(content or "")
        return handle.name
    finally:
        handle.close()

