# -*- coding: utf-8 -*-
import os

import gradio as gr

from config_manager import (
    EMBEDDING_INTERFACE_OPTIONS,
    LLM_INTERFACE_OPTIONS,
    load_config,
)
from webui.services import config_service, file_service, role_service, webdav_service

CSS = """
.compact textarea { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }
.scroll-text textarea {
    overflow-y: auto !important;
    resize: vertical;
    min-height: 420px;
    max-height: 70vh;
    overscroll-behavior: contain;
}
.scroll-preview textarea {
    overflow-y: auto !important;
    resize: vertical;
    min-height: 300px;
    max-height: 60vh;
    overscroll-behavior: contain;
}
"""

_GENERATION_SERVICE = None


def _generation_service():
    global _GENERATION_SERVICE
    if _GENERATION_SERVICE is None:
        from webui.services import generation_service

        _GENERATION_SERVICE = generation_service
    return _GENERATION_SERVICE


def create_webui(config_file: str = "config.json") -> gr.Blocks:
    config = load_config(config_file)
    other = config.get("other_params", {})
    choices = config.get("choose_configs", {})
    llm_names = config_service.config_names(config)
    embedding_name = config_service.selected_embedding_name(config)
    embedding_config = config_service.get_embedding_config(config, embedding_name)
    current_llm_name = llm_names[0]
    current_llm = config_service.get_llm_config(config, current_llm_name)
    proxy_config = config.get("proxy_setting", {})
    webdav_config = config.get("webdav_config", {})
    project_root = other.get("filepath") if os.path.isdir(other.get("filepath", "")) else "."
    try:
        initial_chapters = file_service.list_chapters(other.get("filepath", ""))
    except ValueError:
        initial_chapters = []
    initial_batch_start = max([int(chapter) for chapter in initial_chapters], default=0) + 1

    with gr.Blocks(title="AI Novel Generator WebUI", fill_width=True) as demo:
        config_state = gr.State(config)
        log_state = gr.State("")
        analyzed_roles_state = gr.State([])

        gr.Markdown("# AI Novel Generator WebUI")

        with gr.Tabs():
            with gr.Tab("主工作台"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chapter_editor = gr.Textbox(
                            label="本章内容（可编辑）",
                            value="",
                            lines=24,
                            elem_classes=["compact"],
                        )
                        with gr.Row():
                            btn_arch = gr.Button("Step1. 生成架构", variant="primary")
                            btn_blueprint = gr.Button("Step2. 生成目录", variant="primary")
                            btn_build_prompt = gr.Button("Step3a. 生成提示词")
                            btn_draft = gr.Button("Step3b. 生成草稿", variant="primary")
                            btn_finalize = gr.Button("Step4. 定稿章节", variant="primary")
                        draft_prompt = gr.Textbox(label="当前章节请求提示词（可编辑）", lines=12)
                        with gr.Row():
                            auto_enrich = gr.Checkbox(label="定稿时字数不足自动扩写", value=False)
                            btn_consistency = gr.Button("一致性审校")
                        log_box = gr.Textbox(label="输出日志", value="", lines=11, interactive=False)
                    with gr.Column(scale=1):
                        filepath = gr.Textbox(label="保存路径", value=other.get("filepath", ""))
                        topic = gr.Textbox(label="主题 Topic", value=other.get("topic", ""), lines=4)
                        genre = gr.Textbox(label="类型 Genre", value=other.get("genre", "玄幻"))
                        with gr.Row():
                            num_chapters = gr.Number(label="章节数", value=other.get("num_chapters", 10), precision=0)
                            word_number = gr.Number(label="每章字数", value=other.get("word_number", 3000), precision=0)
                            chapter_num = gr.Number(label="章节号", value=file_service.safe_int(other.get("chapter_num"), 1), precision=0)
                        user_guidance = gr.Textbox(label="内容指导", value=other.get("user_guidance", ""), lines=4)
                        characters_involved = gr.Textbox(label="核心人物", value=other.get("characters_involved", ""), lines=3)
                        key_items = gr.Textbox(label="关键道具", value=other.get("key_items", ""))
                        scene_location = gr.Textbox(label="空间坐标", value=other.get("scene_location", ""))
                        time_constraint = gr.Textbox(label="时间压力", value=other.get("time_constraint", ""))
                        btn_save_params = gr.Button("保存主参数")
                        with gr.Accordion("批量生成", open=False):
                            batch_start = gr.Number(label="起始章节", value=initial_batch_start, precision=0)
                            batch_end = gr.Number(label="结束章节", value=initial_batch_start, precision=0)
                            batch_word = gr.Number(label="期望字数", value=other.get("word_number", 3000), precision=0)
                            batch_min = gr.Number(label="最低字数", value=other.get("word_number", 3000), precision=0)
                            batch_auto_enrich = gr.Checkbox(label="低于最低字数时自动扩写", value=False)
                            btn_batch = gr.Button("开始批量生成")

            with gr.Tab("配置"):
                with gr.Row():
                    with gr.Column(scale=1):
                        llm_dropdown = gr.Dropdown(label="当前 LLM 配置", choices=llm_names, value=current_llm_name)
                        new_llm_name = gr.Textbox(label="新增/重命名配置名称")
                        with gr.Row():
                            btn_add_llm = gr.Button("新增")
                            btn_rename_llm = gr.Button("重命名")
                            btn_delete_llm = gr.Button("删除")
                        llm_api_key = gr.Textbox(label="API Key", value=current_llm.get("api_key", ""), type="password")
                        llm_base_url = gr.Textbox(label="Base URL", value=current_llm.get("base_url", ""))
                        llm_interface = gr.Dropdown(label="接口格式", choices=LLM_INTERFACE_OPTIONS, value=current_llm.get("interface_format", "OpenAI"))
                        llm_model = gr.Textbox(label="模型名称", value=current_llm.get("model_name", ""))
                        llm_temperature = gr.Slider(label="Temperature", minimum=0, maximum=2, step=0.01, value=current_llm.get("temperature", 0.7))
                        llm_max_tokens = gr.Number(label="Max Tokens", value=current_llm.get("max_tokens", 8192), precision=0)
                        llm_timeout = gr.Number(label="Timeout (sec)", value=current_llm.get("timeout", 600), precision=0)
                        with gr.Row():
                            btn_save_llm = gr.Button("保存 LLM 配置", variant="primary")
                            btn_test_llm = gr.Button("测试 LLM")
                    with gr.Column(scale=1):
                        embedding_interface = gr.Dropdown(
                            label="Embedding 接口格式",
                            choices=config_service.embedding_names(config),
                            value=embedding_name,
                        )
                        embedding_api_key = gr.Textbox(label="Embedding API Key", value=embedding_config.get("api_key", ""), type="password")
                        embedding_base_url = gr.Textbox(label="Embedding Base URL", value=embedding_config.get("base_url", ""))
                        embedding_model = gr.Textbox(label="Embedding Model Name", value=embedding_config.get("model_name", ""))
                        embedding_k = gr.Number(label="Retrieval Top-K", value=embedding_config.get("retrieval_k", 4), precision=0)
                        with gr.Row():
                            btn_save_embedding = gr.Button("保存 Embedding 配置", variant="primary")
                            btn_test_embedding = gr.Button("测试 Embedding")
                        gr.Markdown("### 模型用途选择")
                        architecture_llm = gr.Dropdown(label="生成架构", choices=llm_names, value=choices.get("architecture_llm", llm_names[0]))
                        chapter_outline_llm = gr.Dropdown(label="生成目录", choices=llm_names, value=choices.get("chapter_outline_llm", llm_names[0]))
                        prompt_draft_llm = gr.Dropdown(label="生成草稿", choices=llm_names, value=choices.get("prompt_draft_llm", llm_names[0]))
                        final_chapter_llm = gr.Dropdown(label="定稿章节", choices=llm_names, value=choices.get("final_chapter_llm", llm_names[0]))
                        consistency_review_llm = gr.Dropdown(label="一致性审校", choices=llm_names, value=choices.get("consistency_review_llm", llm_names[0]))
                        btn_save_choose = gr.Button("保存模型用途")
                with gr.Accordion("代理设置", open=False):
                    proxy_enabled = gr.Checkbox(label="启用代理", value=proxy_config.get("enabled", False))
                    proxy_address = gr.Textbox(label="地址", value=proxy_config.get("proxy_url", "127.0.0.1"))
                    proxy_port = gr.Textbox(label="端口", value=proxy_config.get("proxy_port", ""))
                    btn_save_proxy = gr.Button("保存代理设置")

            with gr.Tab("文件编辑"):
                with gr.Row():
                    project_files = gr.FileExplorer(
                        label="工程文件",
                        root_dir=project_root,
                        glob="**/*.txt",
                        file_count="single",
                        height=360,
                    )
                    with gr.Column(scale=2):
                        selected_file_content = gr.Textbox(
                            label="选中文件内容",
                            lines=14,
                            max_lines=14,
                            elem_classes=["scroll-preview"],
                        )
                        btn_refresh_project_files = gr.Button("刷新工程文件")
                with gr.Tabs():
                    with gr.Tab("架构"):
                        architecture_text = gr.Textbox(
                            label="Novel_architecture.txt",
                            lines=22,
                            max_lines=22,
                            elem_classes=["scroll-text"],
                        )
                        with gr.Row():
                            btn_load_arch_file = gr.Button("加载架构")
                            btn_save_arch_file = gr.Button("保存架构")
                    with gr.Tab("目录"):
                        directory_text = gr.Textbox(
                            label="Novel_directory.txt",
                            lines=22,
                            max_lines=22,
                            elem_classes=["scroll-text"],
                        )
                        with gr.Row():
                            btn_load_directory_file = gr.Button("加载目录")
                            btn_save_directory_file = gr.Button("保存目录")
                    with gr.Tab("角色状态"):
                        character_state_text = gr.Textbox(
                            label="character_state.txt",
                            lines=22,
                            max_lines=22,
                            elem_classes=["scroll-text"],
                        )
                        with gr.Row():
                            btn_load_character_file = gr.Button("加载角色状态")
                            btn_save_character_file = gr.Button("保存角色状态")
                    with gr.Tab("全局摘要"):
                        global_summary_text = gr.Textbox(
                            label="global_summary.txt",
                            lines=22,
                            max_lines=22,
                            elem_classes=["scroll-text"],
                        )
                        with gr.Row():
                            btn_load_summary_file = gr.Button("加载摘要")
                            btn_save_summary_file = gr.Button("保存摘要")
                    with gr.Tab("剧情要点"):
                        plot_arcs_text = gr.Textbox(
                            label="plot_arcs.txt",
                            lines=22,
                            max_lines=22,
                            elem_classes=["scroll-text"],
                        )
                        with gr.Row():
                            btn_load_plot_file = gr.Button("加载剧情要点")
                            btn_save_plot_file = gr.Button("保存剧情要点")
                    with gr.Tab("章节"):
                        with gr.Row():
                            chapter_select = gr.Dropdown(label="章节列表", choices=initial_chapters, value=None)
                            btn_refresh_chapters = gr.Button("刷新章节列表")
                            btn_prev_chapter = gr.Button("上一章")
                            btn_next_chapter = gr.Button("下一章")
                            btn_load_chapter = gr.Button("加载章节")
                            btn_save_chapter = gr.Button("保存章节")
                        chapter_file_text = gr.Textbox(
                            label="章节文件内容",
                            lines=24,
                            max_lines=24,
                            elem_classes=["scroll-text"],
                        )

            with gr.Tab("知识库"):
                knowledge_file = gr.File(label="上传知识库文件", file_types=[".txt"])
                btn_import_knowledge = gr.Button("导入知识库", variant="primary")
                clear_vector_confirm = gr.Checkbox(label="我确认要清空当前工程的向量库", value=False)
                btn_clear_vectorstore = gr.Button("清空向量库", variant="stop")

            with gr.Tab("角色库"):
                with gr.Row():
                    with gr.Column(scale=1):
                        role_category = gr.Dropdown(label="分类", choices=[], value=None)
                        role_select = gr.Dropdown(label="角色", choices=[], value=None)
                        with gr.Row():
                            btn_refresh_roles = gr.Button("刷新角色库")
                            btn_new_role = gr.Button("新增角色")
                            btn_delete_role = gr.Button("删除角色", variant="stop")
                        new_category_name = gr.Textbox(label="分类名称")
                        with gr.Row():
                            btn_add_category = gr.Button("新增分类")
                            btn_rename_category = gr.Button("重命名分类")
                            btn_delete_category = gr.Button("删除分类", variant="stop")
                        delete_category_mode = gr.Radio(label="删除分类时", choices=["移动角色", "全部删除"], value="移动角色")
                    with gr.Column(scale=2):
                        role_name = gr.Textbox(label="角色名称")
                        role_target_category = gr.Dropdown(
                            label="保存/移动到分类",
                            choices=[role_service.ALL_CATEGORY],
                            value=role_service.ALL_CATEGORY,
                        )
                        role_items = gr.Textbox(label="物品（每行一条）", lines=4)
                        role_abilities = gr.Textbox(label="能力（每行一条）", lines=4)
                        role_status = gr.Textbox(label="状态（每行一条）", lines=4)
                        role_relations = gr.Textbox(label="主要角色间关系网（每行一条）", lines=4)
                        role_events = gr.Textbox(label="触发或加深的事件（每行一条）", lines=4)
                        with gr.Row():
                            btn_save_role = gr.Button("保存角色", variant="primary")
                            btn_move_role = gr.Button("移动角色")
                        role_raw = gr.Textbox(label="角色原始文件预览", lines=10, interactive=False)
                with gr.Accordion("导入角色", open=False):
                    role_source_file = gr.File(label="上传 txt/docx")
                    role_source_text = gr.Textbox(label="待分析文本", lines=12)
                    with gr.Row():
                        btn_read_role_file = gr.Button("读取上传文件")
                        btn_load_character_state_source = gr.Button("加载 character_state.txt")
                        btn_analyze_roles = gr.Button("LLM 分析角色", variant="primary")
                    analyzed_roles_json = gr.JSON(label="分析结果")
                    analyzed_role_names = gr.CheckboxGroup(label="选择要导入的角色", choices=[])
                    import_role_category = gr.Dropdown(
                        label="导入到分类",
                        choices=[role_service.TEMP_CATEGORY],
                        value=role_service.TEMP_CATEGORY,
                        allow_custom_value=True,
                    )
                    btn_import_roles = gr.Button("导入选中角色")

            with gr.Tab("WebDAV"):
                webdav_url = gr.Textbox(label="WebDAV URL", value=webdav_config.get("webdav_url", ""))
                webdav_username = gr.Textbox(label="WebDAV 用户名", value=webdav_config.get("webdav_username", ""))
                webdav_password = gr.Textbox(label="WebDAV 密码", value=webdav_config.get("webdav_password", ""), type="password")
                with gr.Row():
                    btn_webdav_test = gr.Button("测试连接")
                    btn_webdav_backup = gr.Button("备份 config.json")
                    btn_webdav_restore = gr.Button("恢复 config.json")

        params_inputs = [
            filepath,
            topic,
            genre,
            num_chapters,
            word_number,
            chapter_num,
            user_guidance,
            characters_involved,
            key_items,
            scene_location,
            time_constraint,
        ]

        def collect_params(*values):
            return {
                "filepath": values[0],
                "topic": values[1],
                "genre": values[2],
                "num_chapters": values[3],
                "word_number": values[4],
                "chapter_num": values[5],
                "user_guidance": values[6],
                "characters_involved": values[7],
                "key_items": values[8],
                "scene_location": values[9],
                "time_constraint": values[10],
            }

        def append_log(log, message):
            log = (log or "").strip()
            return f"{log}\n{message}".strip() if log else message

        def dropdown_updates(cfg, selected=None):
            names = config_service.config_names(cfg)
            selected = selected if selected in names else names[0]
            return (
                gr.update(choices=names, value=selected),
                gr.update(choices=names),
                gr.update(choices=names),
                gr.update(choices=names),
                gr.update(choices=names),
                gr.update(choices=names),
            )

        def role_dropdown_updates(path, category=None, role=None):
            try:
                categories = role_service.list_categories(path)
            except ValueError:
                return (
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[role_service.ALL_CATEGORY], value=role_service.ALL_CATEGORY),
                    gr.update(choices=[role_service.TEMP_CATEGORY], value=role_service.TEMP_CATEGORY),
                )
            category = category if category in categories else categories[0]
            roles = role_service.list_roles(path, category)
            role = role if role in roles else (roles[0] if roles else None)
            return (
                gr.update(choices=categories, value=category),
                gr.update(choices=roles, value=role),
                gr.update(choices=categories, value=category),
                gr.update(choices=categories, value=role_service.TEMP_CATEGORY if role_service.TEMP_CATEGORY in categories else category),
            )

        def refresh_role_outputs(path, category=None, role=None):
            cat_update, role_update, target_update, import_update = role_dropdown_updates(path, category, role)
            return cat_update, role_update, target_update, import_update

        def save_params_handler(*args):
            cfg = config_service.save_project_params(config_file, collect_params(*args[:-1]))
            log = append_log(args[-1], "主参数已保存。")
            return cfg, log, log

        btn_save_params.click(
            save_params_handler,
            inputs=params_inputs + [log_state],
            outputs=[config_state, log_state, log_box],
        )

        def architecture_handler(cfg, *args):
            params = collect_params(*args[:-1])
            cfg = config_service.save_project_params(config_file, params)
            logs = []
            text = _generation_service().generate_architecture(cfg, params, logs.append)
            log = append_log(args[-1], "\n".join([*logs, "小说架构生成完成。"]))
            return cfg, text, log, log

        btn_arch.click(
            architecture_handler,
            inputs=[config_state] + params_inputs + [log_state],
            outputs=[config_state, architecture_text, log_state, log_box],
        )

        def blueprint_handler(cfg, *args):
            params = collect_params(*args[:-1])
            cfg = config_service.save_project_params(config_file, params)
            logs = []
            text = _generation_service().generate_blueprint(cfg, params, logs.append)
            log = append_log(args[-1], "\n".join([*logs, "章节蓝图生成完成。"]))
            return cfg, text, log, log

        btn_blueprint.click(
            blueprint_handler,
            inputs=[config_state] + params_inputs + [log_state],
            outputs=[config_state, directory_text, log_state, log_box],
        )

        def build_prompt_handler(cfg, *args):
            params = collect_params(*args[:-1])
            prompt = _generation_service().build_draft_prompt(cfg, params)
            log = append_log(args[-1], f"第 {int(float(params['chapter_num']))} 章提示词已生成，可编辑后继续生成草稿。")
            return prompt, log, log

        btn_build_prompt.click(
            build_prompt_handler,
            inputs=[config_state] + params_inputs + [log_state],
            outputs=[draft_prompt, log_state, log_box],
        )

        def draft_handler(cfg, *args):
            params = collect_params(*args[:-2])
            edited_prompt = args[-2] or _generation_service().build_draft_prompt(cfg, params)
            logs = []
            text = _generation_service().generate_draft(cfg, params, edited_prompt, logs.append)
            log = append_log(args[-1], "\n".join([*logs, "章节草稿生成完成。"]))
            return text, text, log, log

        btn_draft.click(
            draft_handler,
            inputs=[config_state] + params_inputs + [draft_prompt, log_state],
            outputs=[chapter_editor, chapter_file_text, log_state, log_box],
        )

        def finalize_handler(cfg, *args):
            params = collect_params(*args[:-3])
            text = args[-3]
            enrich = args[-2]
            logs = []
            final_text = _generation_service().finalize_current_chapter(cfg, params, text, enrich, logs.append)
            log = append_log(args[-1], "\n".join([*logs, "章节定稿完成。"]))
            return final_text, final_text, log, log

        btn_finalize.click(
            finalize_handler,
            inputs=[config_state] + params_inputs + [chapter_editor, auto_enrich, log_state],
            outputs=[chapter_editor, chapter_file_text, log_state, log_box],
        )

        def consistency_handler(cfg, *args):
            params = collect_params(*args[:-1])
            result = _generation_service().run_consistency_check(cfg, params)
            log = append_log(args[-1], f"审校结果：\n{result}")
            return log, log

        btn_consistency.click(
            consistency_handler,
            inputs=[config_state] + params_inputs + [log_state],
            outputs=[log_state, log_box],
        )

        def batch_handler(cfg, *args, progress=gr.Progress()):
            params = collect_params(*args[:11])
            start, end, batch_target, batch_min_words, enrich, log = args[11:]
            logs = []
            result = _generation_service().batch_generate(
                cfg,
                params,
                start,
                end,
                batch_target,
                batch_min_words,
                enrich,
                log=logs.append,
                progress=progress,
            )
            new_log = append_log(log, "\n".join([*logs, result]))
            chapters = file_service.list_chapters(params.get("filepath", ""))
            return gr.update(choices=chapters, value=chapters[-1] if chapters else None), new_log, new_log

        btn_batch.click(
            batch_handler,
            inputs=[config_state] + params_inputs + [batch_start, batch_end, batch_word, batch_min, batch_auto_enrich, log_state],
            outputs=[chapter_select, log_state, log_box],
        )

        def select_llm_handler(cfg, name):
            llm = config_service.get_llm_config(cfg, name)
            return (
                llm.get("api_key", ""),
                llm.get("base_url", ""),
                llm.get("interface_format", "OpenAI"),
                llm.get("model_name", ""),
                llm.get("temperature", 0.7),
                llm.get("max_tokens", 8192),
                llm.get("timeout", 600),
            )

        llm_dropdown.change(
            select_llm_handler,
            inputs=[config_state, llm_dropdown],
            outputs=[llm_api_key, llm_base_url, llm_interface, llm_model, llm_temperature, llm_max_tokens, llm_timeout],
        )

        def save_llm_handler(name, api_key, base_url, interface, model, temperature, max_tokens, timeout, log):
            cfg = config_service.save_llm_config(config_file, name, {
                "api_key": api_key,
                "base_url": base_url,
                "interface_format": interface,
                "model_name": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            })
            log = append_log(log, f"LLM 配置 '{name}' 已保存。")
            updates = dropdown_updates(cfg, name)
            return (cfg, *updates, log, log)

        btn_save_llm.click(
            save_llm_handler,
            inputs=[llm_dropdown, llm_api_key, llm_base_url, llm_interface, llm_model, llm_temperature, llm_max_tokens, llm_timeout, log_state],
            outputs=[config_state, llm_dropdown, architecture_llm, chapter_outline_llm, prompt_draft_llm, final_chapter_llm, consistency_review_llm, log_state, log_box],
        )

        def add_llm_handler(name, log):
            cfg = config_service.add_llm_config(config_file, name)
            updates = dropdown_updates(cfg, name)
            log = append_log(log, f"LLM 配置 '{name}' 已新增。")
            return (cfg, *updates, log, log)

        btn_add_llm.click(
            add_llm_handler,
            inputs=[new_llm_name, log_state],
            outputs=[config_state, llm_dropdown, architecture_llm, chapter_outline_llm, prompt_draft_llm, final_chapter_llm, consistency_review_llm, log_state, log_box],
        )

        def rename_llm_handler(old_name, new_name, log):
            cfg = config_service.rename_llm_config(config_file, old_name, new_name)
            updates = dropdown_updates(cfg, new_name)
            log = append_log(log, f"LLM 配置已重命名为 '{new_name}'。")
            return (cfg, *updates, log, log)

        btn_rename_llm.click(
            rename_llm_handler,
            inputs=[llm_dropdown, new_llm_name, log_state],
            outputs=[config_state, llm_dropdown, architecture_llm, chapter_outline_llm, prompt_draft_llm, final_chapter_llm, consistency_review_llm, log_state, log_box],
        )

        def delete_llm_handler(name, log):
            cfg = config_service.delete_llm_config(config_file, name)
            selected = config_service.config_names(cfg)[0]
            updates = dropdown_updates(cfg, selected)
            log = append_log(log, f"LLM 配置 '{name}' 已删除。")
            return (cfg, *updates, log, log)

        btn_delete_llm.click(
            delete_llm_handler,
            inputs=[llm_dropdown, log_state],
            outputs=[config_state, llm_dropdown, architecture_llm, chapter_outline_llm, prompt_draft_llm, final_chapter_llm, consistency_review_llm, log_state, log_box],
        )

        def test_llm_handler(api_key, base_url, interface, model, temperature, max_tokens, timeout, log):
            result = config_service.test_llm_config_sync({
                "api_key": api_key,
                "base_url": base_url,
                "interface_format": interface,
                "model_name": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            })
            log = append_log(log, result)
            return log, log

        btn_test_llm.click(
            test_llm_handler,
            inputs=[llm_api_key, llm_base_url, llm_interface, llm_model, llm_temperature, llm_max_tokens, llm_timeout, log_state],
            outputs=[log_state, log_box],
        )

        def embedding_select_handler(cfg, interface):
            emb = config_service.get_embedding_config(cfg, interface)
            return emb.get("api_key", ""), emb.get("base_url", ""), emb.get("model_name", ""), emb.get("retrieval_k", 4)

        embedding_interface.change(
            embedding_select_handler,
            inputs=[config_state, embedding_interface],
            outputs=[embedding_api_key, embedding_base_url, embedding_model, embedding_k],
        )

        def save_embedding_handler(interface, api_key, base_url, model, k, log):
            cfg = config_service.save_embedding_config(config_file, {
                "interface_format": interface,
                "api_key": api_key,
                "base_url": base_url,
                "model_name": model,
                "retrieval_k": k,
            })
            names = config_service.embedding_names(cfg)
            log = append_log(log, f"Embedding 配置 '{interface}' 已保存。")
            return cfg, gr.update(choices=names, value=config_service.selected_embedding_name(cfg)), log, log

        btn_save_embedding.click(
            save_embedding_handler,
            inputs=[embedding_interface, embedding_api_key, embedding_base_url, embedding_model, embedding_k, log_state],
            outputs=[config_state, embedding_interface, log_state, log_box],
        )

        def test_embedding_handler(interface, api_key, base_url, model, log):
            result = config_service.test_embedding_config_sync({
                "interface_format": interface,
                "api_key": api_key,
                "base_url": base_url,
                "model_name": model,
            })
            log = append_log(log, result)
            return log, log

        btn_test_embedding.click(
            test_embedding_handler,
            inputs=[embedding_interface, embedding_api_key, embedding_base_url, embedding_model, log_state],
            outputs=[log_state, log_box],
        )

        def save_choose_handler(*args):
            values = {
                "architecture_llm": args[0],
                "chapter_outline_llm": args[1],
                "prompt_draft_llm": args[2],
                "final_chapter_llm": args[3],
                "consistency_review_llm": args[4],
            }
            cfg = config_service.save_choose_configs(config_file, values)
            log = append_log(args[5], "模型用途配置已保存。")
            return cfg, log, log

        btn_save_choose.click(
            save_choose_handler,
            inputs=[architecture_llm, chapter_outline_llm, prompt_draft_llm, final_chapter_llm, consistency_review_llm, log_state],
            outputs=[config_state, log_state, log_box],
        )

        def save_proxy_handler(enabled, address, port, log):
            cfg = config_service.save_proxy_config(config_file, enabled, address, port)
            log = append_log(log, "代理设置已保存。")
            return cfg, log, log

        btn_save_proxy.click(
            save_proxy_handler,
            inputs=[proxy_enabled, proxy_address, proxy_port, log_state],
            outputs=[config_state, log_state, log_box],
        )

        def refresh_files_handler(path):
            root = path if os.path.isdir(path or "") else "."
            return gr.update(root_dir=root, value=None)

        btn_refresh_project_files.click(refresh_files_handler, inputs=[filepath], outputs=[project_files])

        def show_selected_file(path, selected):
            if not selected:
                return ""
            selected_path = selected if os.path.isabs(selected) else os.path.join(path or ".", selected)
            return file_service.read_uploaded_text(selected_path)

        project_files.change(show_selected_file, inputs=[filepath, project_files], outputs=[selected_file_content])

        def load_project_file_handler(path, key, log):
            text = file_service.read_project_file(path, key)
            log = append_log(log, f"{file_service.PROJECT_TEXT_FILES[key]} 已加载。")
            return text, log, log

        def save_project_file_handler(path, key, content, log):
            filename = file_service.save_project_file(path, key, content)
            log = append_log(log, f"已保存 {filename}。")
            return log, log

        btn_load_arch_file.click(lambda p, l: load_project_file_handler(p, "architecture", l), [filepath, log_state], [architecture_text, log_state, log_box])
        btn_save_arch_file.click(lambda p, c, l: save_project_file_handler(p, "architecture", c, l), [filepath, architecture_text, log_state], [log_state, log_box])
        btn_load_directory_file.click(lambda p, l: load_project_file_handler(p, "directory", l), [filepath, log_state], [directory_text, log_state, log_box])
        btn_save_directory_file.click(lambda p, c, l: save_project_file_handler(p, "directory", c, l), [filepath, directory_text, log_state], [log_state, log_box])
        btn_load_character_file.click(lambda p, l: load_project_file_handler(p, "character_state", l), [filepath, log_state], [character_state_text, log_state, log_box])
        btn_save_character_file.click(lambda p, c, l: save_project_file_handler(p, "character_state", c, l), [filepath, character_state_text, log_state], [log_state, log_box])
        btn_load_summary_file.click(lambda p, l: load_project_file_handler(p, "global_summary", l), [filepath, log_state], [global_summary_text, log_state, log_box])
        btn_save_summary_file.click(lambda p, c, l: save_project_file_handler(p, "global_summary", c, l), [filepath, global_summary_text, log_state], [log_state, log_box])
        btn_load_plot_file.click(lambda p, l: load_project_file_handler(p, "plot_arcs", l), [filepath, log_state], [plot_arcs_text, log_state, log_box])
        btn_save_plot_file.click(lambda p, c, l: save_project_file_handler(p, "plot_arcs", c, l), [filepath, plot_arcs_text, log_state], [log_state, log_box])

        def refresh_chapters_handler(path):
            chapters = file_service.list_chapters(path)
            return gr.update(choices=chapters, value=chapters[0] if chapters else None)

        btn_refresh_chapters.click(refresh_chapters_handler, inputs=[filepath], outputs=[chapter_select])

        def load_chapter_handler(path, chapter, log):
            text = file_service.load_chapter(path, chapter)
            log = append_log(log, f"第 {chapter} 章已加载。")
            return text, text, log, log

        btn_load_chapter.click(load_chapter_handler, inputs=[filepath, chapter_select, log_state], outputs=[chapter_file_text, chapter_editor, log_state, log_box])

        def save_chapter_handler(path, chapter, content, log):
            filename = file_service.save_chapter(path, chapter, content)
            log = append_log(log, f"已保存 {filename}。")
            return log, log

        btn_save_chapter.click(save_chapter_handler, inputs=[filepath, chapter_select, chapter_file_text, log_state], outputs=[log_state, log_box])

        def step_chapter_handler(path, current, delta):
            chapters = file_service.list_chapters(path)
            selected = file_service.next_chapter_number(chapters, current, delta)
            return gr.update(choices=chapters, value=selected), file_service.load_chapter(path, selected) if selected else ""

        btn_prev_chapter.click(lambda p, c: step_chapter_handler(p, c, -1), [filepath, chapter_select], [chapter_select, chapter_file_text])
        btn_next_chapter.click(lambda p, c: step_chapter_handler(p, c, 1), [filepath, chapter_select], [chapter_select, chapter_file_text])

        def import_knowledge_handler(cfg, *args):
            params = collect_params(*args[:11])
            result = _generation_service().import_knowledge(cfg, params, args[11])
            log = append_log(args[12], result)
            return log, log

        btn_import_knowledge.click(
            import_knowledge_handler,
            inputs=[config_state] + params_inputs + [knowledge_file, log_state],
            outputs=[log_state, log_box],
        )

        def clear_vector_handler(*args):
            params = collect_params(*args[:11])
            if not args[11]:
                raise gr.Error("请先勾选确认框。")
            result = _generation_service().clear_knowledge_vectorstore(params)
            log = append_log(args[12], result)
            return log, log

        btn_clear_vectorstore.click(
            clear_vector_handler,
            inputs=params_inputs + [clear_vector_confirm, log_state],
            outputs=[log_state, log_box],
        )

        def refresh_roles_handler(path):
            return refresh_role_outputs(path)

        btn_refresh_roles.click(
            refresh_roles_handler,
            inputs=[filepath],
            outputs=[role_category, role_select, role_target_category, import_role_category],
        )

        def category_change_handler(path, category):
            return role_dropdown_updates(path, category)[1]

        role_category.change(category_change_handler, inputs=[filepath, role_category], outputs=[role_select])

        def load_role_handler(path, category, role):
            if not role:
                return "", "", category, "", "", "", "", "", ""
            data = role_service.read_role(path, role, category)
            attrs = role_service.attrs_to_texts(data["attributes"])
            return data["raw"], data["name"], data["category"], *attrs

        role_select.change(
            load_role_handler,
            inputs=[filepath, role_category, role_select],
            outputs=[role_raw, role_name, role_target_category, role_items, role_abilities, role_status, role_relations, role_events],
        )

        def new_role_handler(path, category, log):
            data = role_service.create_role(path, category or role_service.ALL_CATEGORY)
            cat_update, role_update, target_update, import_update = role_dropdown_updates(path, category, data["name"])
            attrs = role_service.attrs_to_texts(data["attributes"])
            log = append_log(log, f"角色 '{data['name']}' 已创建。")
            return cat_update, role_update, target_update, import_update, data["raw"], data["name"], *attrs, log, log

        btn_new_role.click(
            new_role_handler,
            inputs=[filepath, role_category, log_state],
            outputs=[role_category, role_select, role_target_category, import_role_category, role_raw, role_name, role_items, role_abilities, role_status, role_relations, role_events, log_state, log_box],
        )

        def save_role_handler(path, category, current_role, name, target_category, items, abilities, status, relations, events, log):
            attrs = role_service.attrs_from_texts(items, abilities, status, relations, events)
            data = role_service.save_role(path, target_category or category, current_role, name, attrs)
            cat_update, role_update, target_update, import_update = role_dropdown_updates(path, target_category or category, data["name"])
            attr_texts = role_service.attrs_to_texts(data["attributes"])
            log = append_log(log, f"角色 '{data['name']}' 已保存。")
            return cat_update, role_update, target_update, import_update, data["raw"], data["name"], *attr_texts, log, log

        btn_save_role.click(
            save_role_handler,
            inputs=[filepath, role_category, role_select, role_name, role_target_category, role_items, role_abilities, role_status, role_relations, role_events, log_state],
            outputs=[role_category, role_select, role_target_category, import_role_category, role_raw, role_name, role_items, role_abilities, role_status, role_relations, role_events, log_state, log_box],
        )

        def move_role_handler(path, role, from_category, to_category, log):
            data = role_service.move_role(path, role, from_category, to_category)
            cat_update, role_update, target_update, import_update = role_dropdown_updates(path, to_category, data["name"])
            attrs = role_service.attrs_to_texts(data["attributes"])
            log = append_log(log, f"角色 '{data['name']}' 已移动到 {to_category}。")
            return cat_update, role_update, target_update, import_update, data["raw"], data["name"], *attrs, log, log

        btn_move_role.click(
            move_role_handler,
            inputs=[filepath, role_select, role_category, role_target_category, log_state],
            outputs=[role_category, role_select, role_target_category, import_role_category, role_raw, role_name, role_items, role_abilities, role_status, role_relations, role_events, log_state, log_box],
        )

        def delete_role_handler(path, category, role, log):
            roles = role_service.delete_role(path, category, role)
            log = append_log(log, f"角色 '{role}' 已删除。")
            return gr.update(choices=roles, value=roles[0] if roles else None), "", "", "", "", "", "", "", log, log

        btn_delete_role.click(
            delete_role_handler,
            inputs=[filepath, role_category, role_select, log_state],
            outputs=[role_select, role_raw, role_name, role_items, role_abilities, role_status, role_relations, role_events, log_state, log_box],
        )

        def add_category_handler(path, name, log):
            categories = role_service.add_category(path, name)
            log = append_log(log, f"分类 '{name}' 已新增。")
            return gr.update(choices=categories, value=name), gr.update(choices=categories, value=name), gr.update(choices=categories), log, log

        btn_add_category.click(
            add_category_handler,
            inputs=[filepath, new_category_name, log_state],
            outputs=[role_category, role_target_category, import_role_category, log_state, log_box],
        )

        def rename_category_handler(path, old_name, new_name, log):
            categories = role_service.rename_category(path, old_name, new_name)
            log = append_log(log, f"分类 '{old_name}' 已重命名为 '{new_name}'。")
            return gr.update(choices=categories, value=new_name), gr.update(choices=categories, value=new_name), gr.update(choices=categories), log, log

        btn_rename_category.click(
            rename_category_handler,
            inputs=[filepath, role_category, new_category_name, log_state],
            outputs=[role_category, role_target_category, import_role_category, log_state, log_box],
        )

        def delete_category_handler(path, category, mode_label, log):
            mode = "move" if mode_label == "移动角色" else "all"
            categories = role_service.delete_category(path, category, mode)
            selected = categories[0]
            log = append_log(log, f"分类 '{category}' 已删除。")
            return gr.update(choices=categories, value=selected), gr.update(choices=categories, value=selected), gr.update(choices=categories), log, log

        btn_delete_category.click(
            delete_category_handler,
            inputs=[filepath, role_category, delete_category_mode, log_state],
            outputs=[role_category, role_target_category, import_role_category, log_state, log_box],
        )

        btn_read_role_file.click(role_service.uploaded_role_source_text, inputs=[role_source_file], outputs=[role_source_text])
        btn_load_character_state_source.click(lambda p: file_service.read_project_file(p, "character_state"), inputs=[filepath], outputs=[role_source_text])

        def analyze_roles_handler(cfg, llm_name, text, log):
            roles = role_service.analyze_roles_from_text(cfg, llm_name, text)
            names = [role["name"] for role in roles]
            log = append_log(log, f"已分析出 {len(names)} 个角色。")
            return roles, roles, gr.update(choices=names, value=names), log, log

        btn_analyze_roles.click(
            analyze_roles_handler,
            inputs=[config_state, prompt_draft_llm, role_source_text, log_state],
            outputs=[analyzed_roles_state, analyzed_roles_json, analyzed_role_names, log_state, log_box],
        )

        def import_roles_handler(path, roles, selected, category, log):
            imported_roles = role_service.import_roles(path, roles, selected, category)
            categories = role_service.list_categories(path)
            log = append_log(log, f"已导入 {len(selected or [])} 个角色到 {category}。")
            return gr.update(choices=categories, value=category), gr.update(choices=imported_roles, value=imported_roles[0] if imported_roles else None), log, log

        btn_import_roles.click(
            import_roles_handler,
            inputs=[filepath, analyzed_roles_state, analyzed_role_names, import_role_category, log_state],
            outputs=[role_category, role_select, log_state, log_box],
        )

        def webdav_result_handler(action, url, username, password, log):
            if action == "test":
                result = webdav_service.test_connection(config_file, url, username, password)
            elif action == "backup":
                result = webdav_service.backup_config(config_file, url, username, password)
            else:
                result = webdav_service.restore_config(config_file, url, username, password)
            log = append_log(log, result)
            return load_config(config_file), log, log

        btn_webdav_test.click(lambda u, n, p, l: webdav_result_handler("test", u, n, p, l), [webdav_url, webdav_username, webdav_password, log_state], [config_state, log_state, log_box])
        btn_webdav_backup.click(lambda u, n, p, l: webdav_result_handler("backup", u, n, p, l), [webdav_url, webdav_username, webdav_password, log_state], [config_state, log_state, log_box])
        btn_webdav_restore.click(lambda u, n, p, l: webdav_result_handler("restore", u, n, p, l), [webdav_url, webdav_username, webdav_password, log_state], [config_state, log_state, log_box])

        demo.load(
            lambda p: refresh_role_outputs(p),
            inputs=[filepath],
            outputs=[role_category, role_select, role_target_category, import_role_category],
        )

    demo.queue(default_concurrency_limit=1)
    return demo
