"""Build index.html for the modular app from the original monolith."""
import os

src = os.path.join(os.path.dirname(__file__), '..', 'results', 'array_explorer.html')
dst = os.path.join(os.path.dirname(__file__), '..', 'app', 'index.html')

with open(src, 'r', encoding='utf-8') as f:
    lines = f.readlines()

head = (
    '<!DOCTYPE html>\n'
    '<html lang="en">\n'
    '<head>\n'
    '<meta charset="UTF-8">\n'
    '<title>Array Beam Pattern Explorer</title>\n'
    '<link rel="stylesheet" href="style.css">\n'
    '</head>\n'
)

# Body: lines 196-745 (1-indexed) = index 195-744
body_content = ''.join(lines[195:745])

# Importmap: lines 747-749 (1-indexed) = index 746-748
importmap = ''.join(lines[746:749])

script_tag = '<script type="module" src="js/main.js"></script>\n'

closing = '</body>\n</html>\n'

full = head + body_content + importmap + script_tag + closing

with open(dst, 'w', encoding='utf-8') as out:
    out.write(full)

print(f'Created index.html: {len(full)} chars, {full.count(chr(10))} lines')
