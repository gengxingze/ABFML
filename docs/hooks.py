"""Reuse repository documentation and map source links without importing ABFML or PyTorch."""

import posixpath
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parent.parent
PAGES = {
    "tools.md": "abfml/tools/README.md",
    "lammps.md": "lmp/ABFML/README.md",
    "examples.md": "examples/README.md",
    "project.md": "README.md",
    "license.md": "LICENSE",
}
CHAPTERS = {
    "install.md": ("manual.md", "2. 安装与环境", "3. 第一个完整流程"),
    "quickstart.md": ("manual.md", "3. 第一个完整流程", "4. 准备结构与标签"),
    "training.md": ("manual.md", "5. 配置和训练模型", "6. 模型评估与误差解释"),
    "evaluation.md": ("manual.md", "6. 模型评估与误差解释", "7. 物理一致性检查"),
    "checking.md": ("manual.md", "7. 物理一致性检查", "8. 绘图与数据分析"),
    "custom.md": ("manual.md", "9. 构建自定义模型", "10. 通过 ASE 使用模型"),
    "deepse.md": ("theory.md", "3. DeepPot-SE：可学习嵌入与矩阵收缩", "4. NEP：可训练径向函数与角向矩收缩"),
    "nep.md": ("theory.md", "4. NEP：可训练径向函数与角向矩收缩", "5. 三类模型的本质区别"),
    "mechanics.md": ("theory.md", "6. 从能量到力和 virial", "7. 当前代码索引"),
}


def on_files(files, config):
    for destination, source in PAGES.items():
        content = (ROOT / source).read_text(encoding="utf-8-sig")
        if destination == "license.md":
            content = "# 许可证\n\n```text\n" + content + "\n```\n"
        files.append(File.generated(config, destination, content=content))
    # Extract topic pages from one source while retaining complete pages and existing anchors.
    for destination, (source, start, end) in CHAPTERS.items():
        text = (ROOT / "docs" / source).read_text(encoding="utf-8-sig")
        chapter = text.split("## " + start + "\n", 1)[1].split("## " + end + "\n", 1)[0]
        title = start.split(". ", 1)[1]
        content = "# " + title + "\n\n" + chapter.strip() + "\n\n---\n\n[阅读完整文档](" + source + ")\n"
        files.append(File.generated(config, destination, content=content))
    return files


def on_page_markdown(markdown, page, config, files):
    """Map repository documentation to site pages and source/configuration links to GitHub."""
    uri = page.file.src_uri
    source = ROOT / PAGES.get(uri, "docs/" + uri)
    targets = {(ROOT / value).resolve(): key for key, value in PAGES.items()}
    targets.update({(ROOT / "docs" / file.src_uri).resolve(): file.src_uri
                    for file in files if file.is_documentation_page() and file.src_uri not in PAGES})

    def rewrite(match):
        target = match.group(1)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path or target.startswith("/"):
            return match.group(0)
        resolved = (source.parent / unquote(parsed.path)).resolve()
        suffix = ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
        if resolved in targets:
            path = posixpath.relpath(targets[resolved], posixpath.dirname(uri) or ".")
            return "](" + path + suffix + ")"
        if resolved.is_file() and resolved.is_relative_to(ROOT / "docs"):
            path = posixpath.relpath(resolved.relative_to(ROOT / "docs").as_posix(),
                                     posixpath.dirname(uri) or ".")
            return "](" + path + suffix + ")"
        if resolved.is_file() and resolved.is_relative_to(ROOT):
            path = quote(resolved.relative_to(ROOT).as_posix(), safe="/")
            return "](" + config.repo_url.rstrip("/") + "/blob/master/" + path + suffix + ")"
        return match.group(0)

    # Leave Markdown strings inside code examples unchanged.
    parts = re.split(r"(^```.*?^```[^\n]*$)", markdown, flags=re.M | re.S)
    return "".join(part if index % 2 else re.sub(r"\]\(([^\s)]+)\)", rewrite, part)
                   for index, part in enumerate(parts))
