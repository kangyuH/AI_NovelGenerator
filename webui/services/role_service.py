# -*- coding: utf-8 -*-
import os
import re
import shutil
from pathlib import Path

from llm_adapters import create_llm_adapter
from prompt_definitions import Character_Import_Prompt

from .config_service import get_llm_config
from .file_service import read_uploaded_text

ALL_CATEGORY = "全部"
TEMP_CATEGORY = "临时角色库"
ROLE_ATTRIBUTES = ["物品", "能力", "状态", "主要角色间关系网", "触发或加深的事件"]


def role_root(filepath: str) -> str:
    filepath = (filepath or "").strip()
    if not filepath:
        raise ValueError("请先设置保存路径。")
    return os.path.join(filepath, "角色库")


def ensure_role_library(filepath: str) -> str:
    root = role_root(filepath)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, ALL_CATEGORY), exist_ok=True)
    return root


def list_categories(filepath: str) -> list[str]:
    root = ensure_role_library(filepath)
    categories = [ALL_CATEGORY]
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name != ALL_CATEGORY:
            categories.append(name)
    return categories


def _category_path(filepath: str, category: str) -> str:
    root = ensure_role_library(filepath)
    category = category or ALL_CATEGORY
    path = os.path.join(root, category)
    os.makedirs(path, exist_ok=True)
    return path


def list_roles(filepath: str, category: str = ALL_CATEGORY) -> list[str]:
    root = ensure_role_library(filepath)
    roles = set()
    if category == ALL_CATEGORY:
        categories = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    else:
        categories = [category]

    for item in categories:
        path = os.path.join(root, item)
        if not os.path.isdir(path):
            continue
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                roles.add(os.path.splitext(filename)[0])
    return sorted(roles)


def find_role_path(filepath: str, role_name: str, category: str = ALL_CATEGORY) -> str | None:
    root = ensure_role_library(filepath)
    role_name = normalize_role_name(role_name)
    if not role_name:
        return None

    candidates = [category] if category and category != ALL_CATEGORY else [ALL_CATEGORY] + [
        name for name in os.listdir(root) if name != ALL_CATEGORY and os.path.isdir(os.path.join(root, name))
    ]
    for item in candidates:
        path = os.path.join(root, item, f"{role_name}.txt")
        if os.path.exists(path):
            return path
    return None


def normalize_role_name(role_name: str) -> str:
    value = (role_name or "").strip()
    for colon in (":", "："):
        value = value.split(colon)[0].strip()
    return value


def read_role(filepath: str, role_name: str, category: str = ALL_CATEGORY) -> dict:
    path = find_role_path(filepath, role_name, category)
    if not path:
        raise ValueError(f"找不到角色文件: {role_name}")
    content = _read_text_fallback(path)
    parsed = parse_role_content(content)
    parsed["category"] = os.path.basename(os.path.dirname(path))
    parsed["raw"] = content
    return parsed


def parse_role_content(content: str) -> dict:
    lines = (content or "").splitlines()
    if not lines:
        return {"name": "", "attributes": {name: [] for name in ROLE_ATTRIBUTES}}

    name = normalize_role_name(lines[0])
    attributes = {attr: [] for attr in ROLE_ATTRIBUTES}
    current_attr = None
    for line in lines[1:]:
        stripped = line.strip()
        if "──" in stripped and re.match(r"^[├└]──", stripped):
            attr_part = stripped.split("──", 1)[1]
            attr_name = re.split(r"[:：]", attr_part, 1)[0].strip()
            current_attr = attr_name if attr_name in attributes else None
            continue
        if current_attr and stripped.startswith(("│", "├", "└", "-")):
            item = re.sub(r"^[│├└─\s-]*", "", stripped).strip()
            if item:
                attributes[current_attr].append(item)
    return {"name": name, "attributes": attributes}


def build_role_content(role_name: str, attributes: dict[str, list[str]]) -> str:
    role_name = normalize_role_name(role_name)
    if not role_name:
        raise ValueError("角色名称不能为空。")

    lines = [f"{role_name}："]
    for attr_name in ROLE_ATTRIBUTES:
        lines.append(f"├──{attr_name}：")
        items = [str(item).strip() for item in attributes.get(attr_name, []) if str(item).strip()]
        if not items:
            lines.append("│  └──待补充")
            continue
        for index, item in enumerate(items):
            prefix = "└──" if index == len(items) - 1 else "├──"
            lines.append(f"│  {prefix}{item}")
    return "\n".join(lines)


def create_role(filepath: str, category: str = ALL_CATEGORY, role_name: str = "未命名") -> dict:
    directory = _category_path(filepath, category or ALL_CATEGORY)
    base_name = normalize_role_name(role_name) or "未命名"
    final_name = base_name
    counter = 1
    while os.path.exists(os.path.join(directory, f"{final_name}.txt")):
        final_name = f"{base_name}{counter}"
        counter += 1
    content = build_role_content(final_name, {})
    Path(os.path.join(directory, f"{final_name}.txt")).write_text(content, encoding="utf-8")
    return read_role(filepath, final_name, category)


def save_role(filepath: str, category: str, current_name: str, new_name: str, attributes: dict[str, list[str]]) -> dict:
    category = category or ALL_CATEGORY
    current_name = normalize_role_name(current_name)
    new_name = normalize_role_name(new_name)
    if not new_name:
        raise ValueError("角色名称不能为空。")

    old_path = find_role_path(filepath, current_name, category) if current_name else None
    target_category = category if category != ALL_CATEGORY else (os.path.basename(os.path.dirname(old_path)) if old_path else ALL_CATEGORY)
    target_dir = _category_path(filepath, target_category)
    target_path = os.path.join(target_dir, f"{new_name}.txt")
    conflict = find_role_path(filepath, new_name, ALL_CATEGORY)
    if conflict and os.path.abspath(conflict) != os.path.abspath(old_path or ""):
        raise ValueError(f"角色名称 '{new_name}' 已存在。")

    content = build_role_content(new_name, attributes)
    Path(target_path).write_text(content, encoding="utf-8")
    if old_path and os.path.abspath(old_path) != os.path.abspath(target_path):
        os.remove(old_path)
    return read_role(filepath, new_name, target_category)


def delete_role(filepath: str, category: str, role_name: str) -> list[str]:
    path = find_role_path(filepath, role_name, category)
    if not path:
        raise ValueError(f"找不到角色文件: {role_name}")
    os.remove(path)
    return list_roles(filepath, category)


def move_role(filepath: str, role_name: str, from_category: str, to_category: str) -> dict:
    old_path = find_role_path(filepath, role_name, from_category)
    if not old_path:
        raise ValueError(f"找不到角色文件: {role_name}")
    target_dir = _category_path(filepath, to_category or ALL_CATEGORY)
    new_path = os.path.join(target_dir, os.path.basename(old_path))
    if os.path.abspath(old_path) != os.path.abspath(new_path):
        if os.path.exists(new_path):
            raise ValueError(f"目标分类中已存在角色: {role_name}")
        shutil.move(old_path, new_path)
    return read_role(filepath, role_name, to_category)


def add_category(filepath: str, category_name: str) -> list[str]:
    category_name = (category_name or "").strip()
    if not category_name:
        raise ValueError("分类名称不能为空。")
    if category_name == ALL_CATEGORY:
        raise ValueError("不能创建保留分类。")
    _category_path(filepath, category_name)
    return list_categories(filepath)


def rename_category(filepath: str, old_name: str, new_name: str) -> list[str]:
    old_name = (old_name or "").strip()
    new_name = (new_name or "").strip()
    if old_name == ALL_CATEGORY:
        raise ValueError("不能重命名保留分类。")
    if not old_name or not new_name:
        raise ValueError("分类名称不能为空。")
    root = ensure_role_library(filepath)
    old_path = os.path.join(root, old_name)
    new_path = os.path.join(root, new_name)
    if not os.path.isdir(old_path):
        raise ValueError("原分类不存在。")
    if os.path.exists(new_path):
        raise ValueError("目标分类已存在。")
    os.rename(old_path, new_path)
    return list_categories(filepath)


def delete_category(filepath: str, category_name: str, mode: str = "move") -> list[str]:
    category_name = (category_name or "").strip()
    if category_name == ALL_CATEGORY:
        raise ValueError("不能删除保留分类。")
    root = ensure_role_library(filepath)
    category_path = os.path.join(root, category_name)
    if not os.path.isdir(category_path):
        raise ValueError("分类不存在。")
    if mode == "move":
        all_path = _category_path(filepath, ALL_CATEGORY)
        for filename in os.listdir(category_path):
            if filename.endswith(".txt"):
                shutil.move(os.path.join(category_path, filename), os.path.join(all_path, filename))
    shutil.rmtree(category_path)
    return list_categories(filepath)


def parse_items_text(value: str) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def attrs_from_texts(items: str, abilities: str, status: str, relations: str, events: str) -> dict[str, list[str]]:
    return {
        "物品": parse_items_text(items),
        "能力": parse_items_text(abilities),
        "状态": parse_items_text(status),
        "主要角色间关系网": parse_items_text(relations),
        "触发或加深的事件": parse_items_text(events),
    }


def attrs_to_texts(attributes: dict[str, list[str]]) -> tuple[str, str, str, str, str]:
    return tuple("\n".join(attributes.get(name, [])) for name in ROLE_ATTRIBUTES)


def analyze_roles_from_text(config: dict, llm_config_name: str, text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        raise ValueError("请先提供需要分析的文本。")
    llm_config = get_llm_config(config, llm_config_name)
    adapter = create_llm_adapter(
        interface_format=llm_config.get("interface_format", "OpenAI"),
        base_url=llm_config.get("base_url", ""),
        model_name=llm_config.get("model_name", ""),
        api_key=llm_config.get("api_key", ""),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 8192),
        timeout=llm_config.get("timeout", 600),
    )
    prompt = f"{Character_Import_Prompt}\n<<待分析小说文本开始>>\n{text}\n<<待分析小说文本结束>>"
    return parse_llm_response(_invoke_with_cleaning(adapter, prompt))


def parse_llm_response(response: str) -> list[dict]:
    roles = []
    current_role = None
    current_attr = None
    current_subattr = None
    attribute_pattern = re.compile(r"^([├└]──)([\w\u4e00-\u9fa5]+)\s*[:：]")
    item_pattern = re.compile(r"^│\s+([├└]──)\s*(.*)")

    for line in (response or "").splitlines():
        line = line.rstrip()
        role_match = re.match(r"^\s*([\u4e00-\u9fa5a-zA-Z0-9_ -]+)\s*[:：]\s*$", line)
        if role_match:
            current_role = role_match.group(1).strip()
            roles.append({"name": current_role, "attributes": {name: [] for name in ROLE_ATTRIBUTES}})
            current_attr = None
            current_subattr = None
            continue

        if not current_role:
            continue

        attr_match = attribute_pattern.match(line)
        if attr_match:
            current_attr = attr_match.group(2).strip()
            roles[-1]["attributes"].setdefault(current_attr, [])
            current_subattr = None
            continue

        item_match = item_pattern.match(line)
        if item_match and current_attr:
            content = item_match.group(2).strip()
            if not content:
                continue
            if ":" in content or "：" in content:
                parts = re.split(r"[:：]", content, 1)
                if len(parts) > 1:
                    current_subattr = parts[0].strip()
                    value = parts[1].strip()
                    if value:
                        roles[-1]["attributes"].setdefault(current_attr, []).append(f"{current_subattr}: {value}")
                    continue
            if current_subattr and roles[-1]["attributes"].get(current_attr):
                roles[-1]["attributes"][current_attr][-1] += f"，{content}"
            else:
                roles[-1]["attributes"].setdefault(current_attr, []).append(content)
    return roles


def import_roles(filepath: str, roles: list[dict], selected_names: list[str], category: str = TEMP_CATEGORY) -> list[str]:
    if not selected_names:
        raise ValueError("请至少选择一个角色。")
    selected = {normalize_role_name(name) for name in selected_names}
    for role in roles or []:
        if normalize_role_name(role.get("name", "")) in selected:
            save_role(filepath, category, "", role["name"], role.get("attributes", {}))
    return list_roles(filepath, category)


def uploaded_role_source_text(file_obj) -> str:
    return read_uploaded_text(file_obj)


def _read_text_fallback(path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "cp936", "cp1252"):
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _invoke_with_cleaning(adapter, prompt: str) -> str:
    response = adapter.invoke(prompt)
    return re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL).strip()
