# -*- coding: utf-8 -*-
import os
from typing import Callable

from consistency_checker import check_consistency
from novel_generator import (
    Chapter_blueprint_generate,
    Novel_architecture_generate,
    build_chapter_prompt,
    clear_vector_store,
    enrich_chapter_text,
    finalize_chapter,
    generate_chapter_draft,
    import_knowledge_file,
)
from utils import clear_file_content, read_file, save_string_to_txt

from .config_service import get_embedding_config, get_llm_config_for_role
from .file_service import ensure_project_dir, read_uploaded_text, safe_int, write_temp_utf8_copy


def _llm_kwargs(config: dict, role_key: str) -> dict:
    llm_config = get_llm_config_for_role(config, role_key)
    return {
        "interface_format": llm_config.get("interface_format", "OpenAI"),
        "api_key": llm_config.get("api_key", ""),
        "base_url": llm_config.get("base_url", ""),
        "model_name": llm_config.get("model_name", ""),
        "temperature": llm_config.get("temperature", 0.7),
        "max_tokens": safe_int(llm_config.get("max_tokens"), 8192),
        "timeout": safe_int(llm_config.get("timeout"), 600),
    }


def _embedding_kwargs(config: dict) -> dict:
    embedding_config = get_embedding_config(config)
    return {
        "embedding_api_key": embedding_config.get("api_key", ""),
        "embedding_url": embedding_config.get("base_url", ""),
        "embedding_interface_format": embedding_config.get("interface_format", "OpenAI"),
        "embedding_model_name": embedding_config.get("model_name", ""),
        "embedding_retrieval_k": safe_int(embedding_config.get("retrieval_k"), 4),
    }


def _role_names(value: str) -> list[str]:
    names = []
    for part in (value or "").replace("\n", ",").split(","):
        part = part.strip()
        if part:
            names.append(part)
    return names


def _inject_role_library_content(filepath: str, prompt_text: str, characters_involved: str) -> str:
    role_names = set(_role_names(characters_involved))
    if not role_names:
        return prompt_text

    role_root = os.path.join(filepath, "角色库")
    if not os.path.isdir(role_root):
        return prompt_text

    role_contents = []
    for root, _, files in os.walk(role_root):
        for filename in files:
            if filename.endswith(".txt") and os.path.splitext(filename)[0] in role_names:
                try:
                    role_contents.append(read_file(os.path.join(root, filename)).strip())
                except Exception:
                    continue

    if not role_contents:
        return prompt_text

    role_content = "\n".join(role_contents)
    placeholder_variations = [
        "核心人物(可能未指定)：{characters_involved}",
        "核心人物：{characters_involved}",
        "核心人物(可能未指定):{characters_involved}",
        "核心人物:{characters_involved}",
    ]
    for placeholder in placeholder_variations:
        if placeholder in prompt_text:
            return prompt_text.replace(placeholder, f"核心人物：\n{role_content}")

    lines = prompt_text.splitlines()
    for index, line in enumerate(lines):
        if "核心人物" in line and ("：" in line or ":" in line):
            lines[index] = f"核心人物：\n{role_content}"
            return "\n".join(lines)
    return f"{prompt_text}\n\n核心人物：\n{role_content}"


def generate_architecture(config: dict, params: dict, log: Callable[[str], None] | None = None) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    llm = _llm_kwargs(config, "architecture_llm")
    if log:
        log("开始生成小说架构...")
    Novel_architecture_generate(
        interface_format=llm["interface_format"],
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        llm_model=llm["model_name"],
        topic=params.get("topic", ""),
        genre=params.get("genre", ""),
        number_of_chapters=safe_int(params.get("num_chapters"), 10),
        word_number=safe_int(params.get("word_number"), 3000),
        filepath=filepath,
        temperature=llm["temperature"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
        user_guidance=params.get("user_guidance", ""),
    )
    return read_file(os.path.join(filepath, "Novel_architecture.txt"))


def generate_blueprint(config: dict, params: dict, log: Callable[[str], None] | None = None) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    llm = _llm_kwargs(config, "chapter_outline_llm")
    if log:
        log("开始生成章节蓝图...")
    Chapter_blueprint_generate(
        interface_format=llm["interface_format"],
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        llm_model=llm["model_name"],
        number_of_chapters=safe_int(params.get("num_chapters"), 10),
        filepath=filepath,
        temperature=llm["temperature"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
        user_guidance=params.get("user_guidance", ""),
    )
    return read_file(os.path.join(filepath, "Novel_directory.txt"))


def build_draft_prompt(config: dict, params: dict) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    llm = _llm_kwargs(config, "prompt_draft_llm")
    embedding = _embedding_kwargs(config)
    chapter_num = safe_int(params.get("chapter_num"), 1)
    word_number = safe_int(params.get("word_number"), 3000)

    prompt = build_chapter_prompt(
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        model_name=llm["model_name"],
        filepath=filepath,
        novel_number=chapter_num,
        word_number=word_number,
        temperature=llm["temperature"],
        user_guidance=params.get("user_guidance", ""),
        characters_involved=params.get("characters_involved", ""),
        key_items=params.get("key_items", ""),
        scene_location=params.get("scene_location", ""),
        time_constraint=params.get("time_constraint", ""),
        embedding_api_key=embedding["embedding_api_key"],
        embedding_url=embedding["embedding_url"],
        embedding_interface_format=embedding["embedding_interface_format"],
        embedding_model_name=embedding["embedding_model_name"],
        embedding_retrieval_k=embedding["embedding_retrieval_k"],
        interface_format=llm["interface_format"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
    )
    return _inject_role_library_content(filepath, prompt, params.get("characters_involved", ""))


def generate_draft(config: dict, params: dict, edited_prompt: str, log: Callable[[str], None] | None = None) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    llm = _llm_kwargs(config, "prompt_draft_llm")
    embedding = _embedding_kwargs(config)
    chapter_num = safe_int(params.get("chapter_num"), 1)
    word_number = safe_int(params.get("word_number"), 3000)
    if log:
        log(f"开始生成第 {chapter_num} 章草稿...")
    return generate_chapter_draft(
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        model_name=llm["model_name"],
        filepath=filepath,
        novel_number=chapter_num,
        word_number=word_number,
        temperature=llm["temperature"],
        user_guidance=params.get("user_guidance", ""),
        characters_involved=params.get("characters_involved", ""),
        key_items=params.get("key_items", ""),
        scene_location=params.get("scene_location", ""),
        time_constraint=params.get("time_constraint", ""),
        embedding_api_key=embedding["embedding_api_key"],
        embedding_url=embedding["embedding_url"],
        embedding_interface_format=embedding["embedding_interface_format"],
        embedding_model_name=embedding["embedding_model_name"],
        embedding_retrieval_k=embedding["embedding_retrieval_k"],
        interface_format=llm["interface_format"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
        custom_prompt_text=edited_prompt or None,
    )


def finalize_current_chapter(
    config: dict,
    params: dict,
    chapter_text: str,
    auto_enrich: bool = False,
    log: Callable[[str], None] | None = None,
) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    llm = _llm_kwargs(config, "final_chapter_llm")
    embedding = _embedding_kwargs(config)
    chapter_num = safe_int(params.get("chapter_num"), 1)
    word_number = safe_int(params.get("word_number"), 3000)
    edited_text = (chapter_text or "").strip()
    if not edited_text:
        raise ValueError("当前章节内容为空，无法定稿。")

    if auto_enrich and len(edited_text) < 0.7 * word_number:
        if log:
            log(f"第 {chapter_num} 章低于目标字数 70%，开始扩写...")
        edited_text = enrich_chapter_text(
            chapter_text=edited_text,
            word_number=word_number,
            api_key=llm["api_key"],
            base_url=llm["base_url"],
            model_name=llm["model_name"],
            temperature=llm["temperature"],
            interface_format=llm["interface_format"],
            max_tokens=llm["max_tokens"],
            timeout=llm["timeout"],
        )

    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)
    chapter_path = os.path.join(chapters_dir, f"chapter_{chapter_num}.txt")
    clear_file_content(chapter_path)
    save_string_to_txt(edited_text, chapter_path)

    if log:
        log(f"开始定稿第 {chapter_num} 章...")
    finalize_chapter(
        novel_number=chapter_num,
        word_number=word_number,
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        model_name=llm["model_name"],
        temperature=llm["temperature"],
        filepath=filepath,
        embedding_api_key=embedding["embedding_api_key"],
        embedding_url=embedding["embedding_url"],
        embedding_interface_format=embedding["embedding_interface_format"],
        embedding_model_name=embedding["embedding_model_name"],
        interface_format=llm["interface_format"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
    )
    return read_file(chapter_path)


def run_consistency_check(config: dict, params: dict) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    llm = _llm_kwargs(config, "consistency_review_llm")
    chapter_num = safe_int(params.get("chapter_num"), 1)
    chapter_text = read_file(os.path.join(filepath, "chapters", f"chapter_{chapter_num}.txt"))
    if not chapter_text.strip():
        raise ValueError("当前章节文件为空或不存在，无法审校。")
    return check_consistency(
        novel_setting=read_file(os.path.join(filepath, "Novel_architecture.txt")),
        character_state=read_file(os.path.join(filepath, "character_state.txt")),
        global_summary=read_file(os.path.join(filepath, "global_summary.txt")),
        chapter_text=chapter_text,
        api_key=llm["api_key"],
        base_url=llm["base_url"],
        model_name=llm["model_name"],
        temperature=llm["temperature"],
        interface_format=llm["interface_format"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
        plot_arcs=read_file(os.path.join(filepath, "plot_arcs.txt")),
    )


def batch_generate(
    config: dict,
    params: dict,
    start: int,
    end: int,
    word_number: int,
    min_words: int,
    auto_enrich: bool,
    log: Callable[[str], None] | None = None,
    progress: Callable | None = None,
) -> str:
    start = safe_int(start, 1)
    end = safe_int(end, start)
    if start > end:
        raise ValueError("起始章节不能大于结束章节。")
    params = dict(params)
    params["word_number"] = safe_int(word_number, params.get("word_number", 3000))
    min_words = safe_int(min_words, params["word_number"])
    generated = []
    total = end - start + 1
    for offset, chapter_num in enumerate(range(start, end + 1), start=1):
        if progress:
            progress(offset / total, desc=f"生成第 {chapter_num} 章")
        params["chapter_num"] = chapter_num
        prompt = build_draft_prompt(config, params)
        draft = generate_draft(config, params, prompt, log=log)
        if auto_enrich and len(draft or "") < 0.7 * min_words:
            draft = finalize_current_chapter(config, params, draft, auto_enrich=True, log=log)
        else:
            draft = finalize_current_chapter(config, params, draft, auto_enrich=False, log=log)
        generated.append(f"第 {chapter_num} 章完成，字数 {len(draft or '')}")
    return "\n".join(generated)


def import_knowledge(config: dict, params: dict, file_obj) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    embedding = _embedding_kwargs(config)
    content = read_uploaded_text(file_obj)
    temp_path = write_temp_utf8_copy(content)
    try:
        import_knowledge_file(
            embedding_api_key=embedding["embedding_api_key"],
            embedding_url=embedding["embedding_url"],
            embedding_interface_format=embedding["embedding_interface_format"],
            embedding_model_name=embedding["embedding_model_name"],
            file_path=temp_path,
            filepath=filepath,
        )
        return "知识库文件导入完成。"
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def clear_knowledge_vectorstore(params: dict) -> str:
    filepath = ensure_project_dir(params.get("filepath", ""))
    if clear_vector_store(filepath):
        return "已清空向量库。"
    return f"未能清空向量库，请关闭程序后手动删除 {filepath} 下的 vectorstore 文件夹。"


def default_batch_start(filepath: str) -> int:
    chapters = []
    chapter_dir = os.path.join((filepath or "").strip(), "chapters")
    if os.path.isdir(chapter_dir):
        for filename in os.listdir(chapter_dir):
            if filename.startswith("chapter_") and filename.endswith(".txt"):
                number = filename.replace("chapter_", "").replace(".txt", "")
                if number.isdigit():
                    chapters.append(int(number))
    return max(chapters) + 1 if chapters else 1
