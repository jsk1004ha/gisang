from __future__ import annotations

import base64
import html
from pathlib import Path


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def embed_image_tag(path: Path, alt: str, max_width: str = "100%", mode: str = "full") -> str:
    if not path or not path.exists():
        return f'<div class="plot-placeholder">{html.escape(alt)} 사용할 수 없음 (not available)</div>'
    if mode == "external-assets":
        return (
            f'<img alt="{html.escape(alt)}" style="max-width:{html.escape(max_width)}" '
            f'src="{html.escape(str(path), quote=True)}">'
        )
    warning = ""
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 5:
            warning = f'<div class="small warn">큰 이미지: {size_mb:.1f}MB</div>'
        encoded = image_to_base64(path)
    except Exception as exc:
        return f'<div class="plot-placeholder">{html.escape(alt)} 이미지를 포함할 수 없음: {html.escape(str(exc))}</div>'
    if mode == "thumbnail":
        return (
            f'{warning}<a href="{html.escape(str(path), quote=True)}" title="원본 이미지 열기">'
            f'<img class="thumb" alt="{html.escape(alt)}" style="max-width:{html.escape(max_width)}" '
            f'src="data:image/png;base64,{encoded}"></a>'
        )
    return f'{warning}<img alt="{html.escape(alt)}" style="max-width:{html.escape(max_width)}" src="data:image/png;base64,{encoded}">'
