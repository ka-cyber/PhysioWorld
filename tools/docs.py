"""Dependency-free public Markdown link checking and static HTML documentation.

Checks local links and heading fragments by default. --external opts into
network checks; offline builds never need a documentation framework.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {
    ".git",
    ".venv",
    "build",
    "dist",
    "site",
    "results",
    ".physioworld",
    "outputs",
    "run-artifacts",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
LINK = re.compile(r"!?\[([^\]\n]+)\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def slug(text: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[^\w\s-]", "", text.lower()).replace(" ", "-")


def prose(text: str) -> str:
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.MULTILINE | re.DOTALL)
    return re.sub(r"`[^`\n]+`", "", text)


def anchors(text: str) -> set[str]:
    values = set()
    seen: dict[str, int] = {}
    headings = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.MULTILINE | re.DOTALL)
    for line in headings.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)(?:\s+#+)?$", line)
        if match:
            base = slug(match[1].replace("`", ""))
            number = seen.get(base, 0)
            values.add(f"{base}-{number}" if number else base)
            seen[base] = number + 1
    values.update(re.findall(r'<(?:a|h[1-6])\s+[^>]*id=[\'"]([^\'"]+)', text))
    return values


def documents(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(
            part in SKIP or part.endswith(".egg-info") for part in path.relative_to(root).parts
        )
    )


def check(root: Path, paths: list[Path], external: bool) -> list[str]:
    failures = []
    urls = set()
    for path in paths:
        text = prose(path.read_text(encoding="utf-8"))
        destinations = [match[2] for match in LINK.finditer(text)]
        destinations += re.findall(r"^\s*\[[^]]+\]:\s+(\S+)", text, flags=re.MULTILINE)
        destinations += re.findall(r"<(https?://[^>\s]+)>", text)
        for destination in destinations:
            url = urllib.parse.urlsplit(destination.strip("<>"))
            if url.scheme in {"http", "https"}:
                urls.add(urllib.parse.urlunsplit(url._replace(fragment="")))
                continue
            if url.scheme == "mailto":
                continue
            if url.scheme or url.netloc:
                failures.append(f"{path.relative_to(root)}: unsupported link {destination}")
                continue
            target = (path.parent / urllib.parse.unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(root):
                failures.append(f"{path.relative_to(root)}: link escapes documentation root")
            elif not target.exists():
                failures.append(f"{path.relative_to(root)}: missing {destination}")
            elif url.fragment and target.suffix == ".md":
                if urllib.parse.unquote(url.fragment) not in anchors(
                    target.read_text(encoding="utf-8")
                ):
                    failures.append(f"{path.relative_to(root)}: missing heading {destination}")
    if external:
        opener = urllib.request.build_opener(WebRedirect())
        for external_url in sorted(urls):
            try:
                request = urllib.request.Request(
                    external_url, headers={"User-Agent": "PhysioWorld-docs-link-check"}
                )
                with opener.open(request, timeout=20) as response:
                    if response.status >= 400:
                        failures.append(f"external link HTTP {response.status}: {external_url}")
            except (OSError, urllib.error.URLError) as exc:
                failures.append(f"external link {external_url}: {exc}")
    return failures


class WebRedirect(urllib.request.HTTPRedirectHandler):
    """External docs checks can redirect only to another HTTP(S) URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme not in {"http", "https"}:
            raise ValueError("external link redirected to a non-web scheme")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def inline(text: str) -> str:
    links = []

    def protect(match: re.Match[str]) -> str:
        links.append(f'<a href="{html.escape(match[2], quote=True)}">{html.escape(match[1])}</a>')
        return f"\x00{len(links) - 1}\x00"

    value = html.escape(LINK.sub(protect, text))
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    for index, link in enumerate(links):
        value = value.replace(f"\x00{index}\x00", link)
    return value


def render(text: str) -> str:
    blocks = []
    in_code = False
    in_list = False
    in_table = False
    seen: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith(("```", "~~~")):
            if in_list:
                blocks.append("</ul>")
                in_list = False
            if in_table:
                blocks.append("</table>")
                in_table = False
            blocks.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            blocks.append(html.escape(line) + "\n")
            continue
        list_item = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.+)", line)
        if in_list and not list_item:
            blocks.append("</ul>")
            in_list = False
        if in_table and not line.startswith("|"):
            blocks.append("</table>")
            in_table = False
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            level = len(heading[1])
            base = slug(heading[2])
            number = seen.get(base, 0)
            anchor = f"{base}-{number}" if number else base
            seen[base] = number + 1
            blocks.append(f'<h{level} id="{html.escape(anchor)}">{inline(heading[2])}</h{level}>')
        elif list_item:
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{inline(list_item[1])}</li>")
        elif line.startswith("|"):
            if re.fullmatch(r"[|:\s-]+", line):
                continue
            if not in_table:
                blocks.append("<table>")
                in_table = True
            cells = line.strip("|").split("|")
            blocks.append(
                "<tr>" + "".join(f"<td>{inline(cell.strip())}</td>" for cell in cells) + "</tr>"
            )
        elif line.strip():
            blocks.append(f"<p>{inline(line)}</p>")
    if in_code:
        raise ValueError("unclosed Markdown code fence")
    if in_list:
        blocks.append("</ul>")
    if in_table:
        blocks.append("</table>")
    return "\n".join(blocks)


def build(root: Path, paths: list[Path], output: Path) -> None:
    if output.exists() or not output.parent.is_dir():
        raise ValueError("build output must be a new directory with an existing parent")
    output.mkdir()
    for path in paths:
        relative = path.relative_to(root)
        target = output / relative.with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8")

        def rewrite(match: re.Match[str]) -> str:
            url = urllib.parse.urlsplit(match[2])
            if url.scheme or url.netloc or not url.path:
                return match[0]
            source = (path.parent / urllib.parse.unquote(url.path)).resolve()
            if source.is_file() and source.suffix != ".md":
                copied = output / source.relative_to(root)
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, copied)
            new_path = url.path[:-3] + ".html" if url.path.endswith(".md") else url.path
            return f"[{match[1]}]({urllib.parse.urlunsplit(url._replace(path=new_path))})"

        text = LINK.sub(rewrite, text)
        home = os.path.relpath(output / "docs" / "index.html", target.parent)
        page = (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(path.stem)} — PhysioWorld</title>"
            "<style>body{max-width:72rem;margin:2rem auto;padding:0 1rem;font:18px/1.6 system-ui}"
            "pre{overflow:auto;background:#f4f5f7;padding:1rem}code{font-size:.9em}"
            "td{padding:.4rem;border:1px solid #ddd}table{border-collapse:collapse}</style>"
            f"<nav><a href='{html.escape(home)}'>PhysioWorld documentation</a></nav><main>"
            + render(text)
            + "</main></html>\n"
        )
        target.write_text(page, encoding="utf-8")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><a href="docs/index.html">PhysioWorld documentation</a>\n',
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = documents(root)
    failures = check(root, paths, args.external)
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    if args.command == "build":
        if args.output is None:
            parser.error("build requires --output")
        try:
            build(root, paths, args.output.resolve())
        except (OSError, ValueError) as exc:
            parser.exit(1, f"docs build: {exc}\n")
    print(
        f"Documentation {args.command}: {len(paths)} Markdown files, local links and anchors passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
