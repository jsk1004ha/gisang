from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from weather_korea_forecast.dashboard.status import load_dashboard_state, render_dashboard_html
from weather_korea_forecast.utils.paths import resolve_path


def make_handler(artifact_root: str | Path):
    root = resolve_path(artifact_root)

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "WeatherKoreaDashboard/0.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                state = load_dashboard_state(root)
                self._send_text(render_dashboard_html(state), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                state = load_dashboard_state(root)
                self._send_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), "application/json; charset=utf-8")
                return
            if parsed.path.startswith("/artifact/"):
                self._serve_artifact(parsed.path.removeprefix("/artifact/"))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _send_text(self, text: str, content_type: str) -> None:
            data = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_artifact(self, relative_url: str) -> None:
            relative = Path(unquote(relative_url))
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN, "Artifact path escapes root")
                return
            if not candidate.exists() or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
                return
            content_type = _content_type(candidate)
            data = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DashboardHandler


def _content_type(path: Path) -> str:
    if path.suffix == ".html":
        return "text/html; charset=utf-8"
    if path.suffix == ".json":
        return "application/json; charset=utf-8"
    if path.suffix == ".csv":
        return "text/csv; charset=utf-8"
    if path.suffix == ".md":
        return "text/markdown; charset=utf-8"
    if path.suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local weather prediction status dashboard.")
    parser.add_argument("--artifact-root", default="data/artifacts/v3_experiments")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--write-html", help="Write a static HTML snapshot instead of starting the server.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_html:
        state = load_dashboard_state(args.artifact_root)
        path = Path(args.write_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_dashboard_html(state), encoding="utf-8")
        print(path)
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.artifact_root))
    print(f"Serving weather dashboard at http://{args.host}:{args.port}/")
    print(f"Artifact root: {resolve_path(args.artifact_root)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
