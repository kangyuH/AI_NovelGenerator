# -*- coding: utf-8 -*-
import argparse
import os

from webui.app import create_webui


def resolve_port(port_arg):
    if port_arg is not None:
        return port_arg

    env_port = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    if env_port:
        try:
            return int(env_port)
        except ValueError as exc:
            raise SystemExit(f"GRADIO_SERVER_PORT must be an integer, got: {env_port}") from exc

    return 7860


def main():
    parser = argparse.ArgumentParser(description="Launch AI Novel Generator Gradio WebUI.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON file.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host. Defaults to local-only.")
    parser.add_argument("--port", default=None, type=int, help="Server port. Overrides GRADIO_SERVER_PORT.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public sharing link.")
    parser.add_argument("--no-browser", action="store_true", help="Do not try to open a browser automatically.")
    args = parser.parse_args()

    app = create_webui(config_file=args.config)
    app.launch(
        server_name=args.host,
        server_port=resolve_port(args.port),
        share=args.share,
        show_error=True,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
