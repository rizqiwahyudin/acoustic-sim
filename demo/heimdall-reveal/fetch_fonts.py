"""Download the OFL Google Fonts used by the reveal, subset to the text it draws.

Scans src/*.js for every character in use, asks the Google Fonts CSS API for
exactly those glyphs and writes fonts/*.woff2 plus fonts/fonts.css. Re-run it
after adding new Japanese text.
"""

from __future__ import annotations

import re
import string
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONTS = HERE / "fonts"
FAMILIES = [
    ("Shippori Mincho B1", "800"),
    ("Noto Sans JP", "500;700"),
    ("Barlow Condensed", "500;600;700"),
    ("Share Tech Mono", "400"),
]
LICENSE_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/{}/OFL.txt"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def used_characters() -> str:
    chars = set(string.printable.strip()) | {" "}
    for path in (HERE / "src").glob("*.js"):
        chars |= set(path.read_text(encoding="utf-8"))
    chars |= set("°·×−–—…%")
    return "".join(sorted(ch for ch in chars if ch.isprintable()))


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    FONTS.mkdir(exist_ok=True)
    text = used_characters()
    css_out = []
    for family, weights in FAMILIES:
        query = urllib.parse.urlencode({"family": f"{family}:wght@{weights}", "text": text, "display": "block"})
        css = fetch(f"https://fonts.googleapis.com/css2?{query}").decode("utf-8")
        for block in re.findall(r"@font-face\s*{[^}]*}", css):
            weight = re.search(r"font-weight:\s*(\d+)", block).group(1)
            url = re.search(r"url\((https://[^)]+)\)", block).group(1)
            name = f"{family.replace(' ', '')}-{weight}.woff2"
            (FONTS / name).write_bytes(fetch(url))
            css_out.append(
                "@font-face {\n"
                f"  font-family: '{family}';\n  font-style: normal;\n  font-weight: {weight};\n"
                f"  font-display: block;\n  src: url('{name}') format('woff2');\n}}\n"
            )
            print(f"{name}: {(FONTS / name).stat().st_size // 1024} KiB")
    (FONTS / "fonts.css").write_text("".join(css_out), encoding="utf-8")
    # The SIL Open Font License requires the licence text to accompany the fonts.
    licences = []
    for family, _ in FAMILIES:
        text_ = fetch(LICENSE_URL.format(family.replace(" ", "").lower())).decode("utf-8")
        licences.append(f"==== {family} ====\n\n{text_.strip()}\n")
    (FONTS / "OFL-LICENSES.txt").write_text("\n".join(licences), encoding="utf-8")


if __name__ == "__main__":
    main()
