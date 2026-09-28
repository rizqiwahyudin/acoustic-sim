"""Download the OFL Google Fonts a demo scene uses, subset to the text it draws.

    python demo/fetch_fonts.py demo/heimdall-reveal
    python demo/fetch_fonts.py demo/heimdall-teaser

Reads the families from <scene>/fonts.json, scans <scene>/src/*.js for every
character in use and writes <scene>/fonts/*.woff2, fonts.css and the licence
texts. Re-run after adding new Japanese text.
"""

from __future__ import annotations

import json
import re
import string
import sys
import urllib.parse
import urllib.request
from pathlib import Path

LICENSE_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/{}/OFL.txt"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def used_characters(scene: Path) -> str:
    chars = set(string.printable.strip()) | {" "}
    for path in (scene / "src").glob("*.js"):
        chars |= set(path.read_text(encoding="utf-8"))
    chars |= set("°·×−–—…%")
    return "".join(sorted(ch for ch in chars if ch.isprintable()))


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main(scene: Path) -> None:
    families = json.loads((scene / "fonts.json").read_text(encoding="utf-8"))
    fonts = scene / "fonts"
    fonts.mkdir(exist_ok=True)
    text = used_characters(scene)
    css_out = []
    for family, weights in families:
        query = urllib.parse.urlencode({"family": f"{family}:wght@{weights}", "text": text, "display": "block"})
        css = fetch(f"https://fonts.googleapis.com/css2?{query}").decode("utf-8")
        for block in re.findall(r"@font-face\s*{[^}]*}", css):
            weight = re.search(r"font-weight:\s*(\d+)", block).group(1)
            url = re.search(r"url\((https://[^)]+)\)", block).group(1)
            name = f"{family.replace(' ', '')}-{weight}.woff2"
            (fonts / name).write_bytes(fetch(url))
            css_out.append(
                "@font-face {\n"
                f"  font-family: '{family}';\n  font-style: normal;\n  font-weight: {weight};\n"
                f"  font-display: block;\n  src: url('{name}') format('woff2');\n}}\n"
            )
            print(f"{name}: {(fonts / name).stat().st_size // 1024} KiB")
    (fonts / "fonts.css").write_text("".join(css_out), encoding="utf-8")
    # The SIL Open Font License requires the licence text to accompany the fonts.
    licences = []
    for family, _ in families:
        licence = fetch(LICENSE_URL.format(family.replace(" ", "").lower())).decode("utf-8")
        licences.append(f"==== {family} ====\n\n{licence.strip()}\n")
    (fonts / "OFL-LICENSES.txt").write_text("\n".join(licences), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
