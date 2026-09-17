"""
extract_modules.py — Extracts remaining JS modules from array_explorer.html.
Run from acoustic-sim/ directory.
"""
import os

SRC = os.path.join(os.path.dirname(__file__), '..', 'results', 'array_explorer.html')
APP_JS = os.path.join(os.path.dirname(__file__), '..', 'app', 'js')

with open(SRC, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Get JS content (lines 750-3892, 0-indexed 749-3891)
# The script tag starts at line 750, content starts at 751
js_start = 750  # 0-indexed: line 751 is `import * as THREE from 'three';`
js_end = 3892   # 0-indexed: line before </script>

# Join all JS lines into one string for extraction
js_content = ''.join(lines[js_start:js_end])

# Write the full JS block to a temp file for reference
with open(os.path.join(APP_JS, '_full_original.js'), 'w', encoding='utf-8') as f:
    f.write(js_content)

print(f"Extracted full JS: {len(js_content)} chars ({js_end - js_start} lines)")
print(f"Written to {APP_JS}/_full_original.js")
print("Now manually creating modular files from this reference.")
