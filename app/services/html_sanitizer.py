"""
Reduces model-written HTML to the tags the Docs editor accepts.

Tags outside the allow-list are dropped but their text is kept, except for
tags whose content is never text (script, style, ...), which are dropped with
their content. Only a[href] keeps an attribute, and only with a safe scheme.
Unclosed tags are closed so the result is well formed.
"""

from html import escape
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import urlparse

ALLOWED_TAGS = {"p", "h1", "h2", "h3", "ul", "ol", "li", "strong", "em", "u", "s", "a", "blockquote", "code", "pre"}
DROPPED_WITH_CONTENT = {"script", "style", "iframe", "object", "embed", "template", "noscript"}
SAFE_SCHEMES = {"http", "https", "mailto"}


def _safe_href(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    scheme = urlparse(value.strip()).scheme.lower()
    return value.strip() if scheme in SAFE_SCHEMES else None


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.open_tags: List[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in DROPPED_WITH_CONTENT:
            self.skip_depth += 1
            return
        if self.skip_depth or tag not in ALLOWED_TAGS:
            return
        if tag == "a":
            href = _safe_href(dict(attrs).get("href"))
            self.out.append(f'<a href="{escape(href)}">' if href else "<a>")
        else:
            self.out.append(f"<{tag}>")
        self.open_tags.append(tag)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        # Self-closing tags (br, img, ...) are not in the allow-list.
        return

    def handle_endtag(self, tag: str) -> None:
        if tag in DROPPED_WITH_CONTENT:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth or tag not in self.open_tags:
            return
        while self.open_tags:
            open_tag = self.open_tags.pop()
            self.out.append(f"</{open_tag}>")
            if open_tag == tag:
                break

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.out.append(escape(data, quote=False))

    def result(self) -> str:
        self.close()
        return "".join(self.out) + "".join(f"</{tag}>" for tag in reversed(self.open_tags))


def sanitize_html(html: str) -> str:
    sanitizer = _Sanitizer()
    sanitizer.feed(html)
    return sanitizer.result()
