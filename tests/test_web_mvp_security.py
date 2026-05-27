from pathlib import Path


def test_web_mvp_avoids_dynamic_inner_html_rendering() -> None:
    app_js = Path("web/app.js").read_text(encoding="utf-8")

    assert "innerHTML" not in app_js
    assert "textContent" in app_js
    assert "replaceChildren" in app_js
