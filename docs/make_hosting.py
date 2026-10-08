"""
Draws docs/hosting.png (and .svg): how the app gets from your computer to a host, from start to end.
Edit the boxes and arrows below when the hosting setup changes, then run:

    python docs/make_hosting.py

Needs only the Python standard library, plus Chrome or Chromium for the PNG.
"""
import html
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1500, 1060
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#374151"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<text x="750" y="42" text-anchor="middle" font-size="28" font-weight="700" fill="#111827">Helping Eyes: from your computer to a live website</text>']


def panel(x, y, w, h, title, fill):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#9ca3af"/>')
    o.append(f'<text x="{x + w / 2}" y="{y + 28}" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{html.escape(title)}</text>')


def box(x, y, w, h, lines, fill="#ffffff", stroke="#4b5563"):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    top = y + h / 2 - (len(lines) - 1) * 10 + 5
    for i, text in enumerate(lines):
        o.append(f'<text x="{x + w / 2}" y="{top + i * 20}" text-anchor="middle" font-size="{15 if i == 0 else 13}" '
                 f'font-weight="{700 if i == 0 else 400}" fill="#111827">{html.escape(text)}</text>')


def arrow(points, label=None, at=None):
    d = "M" + " L".join(f"{a},{b}" for a, b in points)
    o.append(f'<path d="{d}" fill="none" stroke="#374151" stroke-width="1.8" marker-end="url(#a)"/>')
    if label:
        o.append(f'<text x="{at[0]}" y="{at[1]}" text-anchor="middle" font-size="12" fill="#374151" stroke="#fff" '
                 f'stroke-width="4" paint-order="stroke">{html.escape(label)}</text>')


BLUE, ORANGE, GREEN, GREY = "#1d4ed8", "#c2410c", "#15803d", "#6b7280"

# 1. start
box(250, 75, 1000, 70, ["START: your computer", "the code, plus a free Gemini API key from Google AI Studio (the key is never put in the code or in git)"],
    "#dbeafe", BLUE)

# 2. two ways to send the code to a host
panel(30, 190, 700, 290, "Option A: Render (free web service)", "#eff6ff")
panel(770, 190, 700, 290, "Option B: Hugging Face Spaces (Docker needs PRO now)", "#fff7ed")
box(60, 230, 640, 55, ["Push the repository to GitHub"], "#fff", BLUE)
box(60, 310, 640, 65, ["On Render: New Web Service, Docker runtime", "Dockerfile Path: helping_eyes/cloud/Dockerfile (Root Directory stays empty)"], "#fff", BLUE)
box(60, 400, 640, 60, ["Add the secret LLM_API_KEY", "in Render's Environment settings"], "#fff", BLUE)
box(800, 230, 640, 55, ["Run: HF_TOKEN=... ./helping_eyes/cloud/deploy_space.sh <username>"], "#fff", ORANGE)
box(800, 310, 640, 65, ["The script assembles a clean folder", "requirements.txt, the 4 app files, web/, start.sh, Dockerfile, Space card"], "#fff", ORANGE)
box(800, 400, 640, 60, ["It pushes the folder to your Space with git", "then add the secret LLM_API_KEY in the Space settings"], "#fff", ORANGE)
arrow([(600, 145), (380, 190)]); arrow([(900, 145), (1120, 190)])
arrow([(380, 285), (380, 310)]); arrow([(380, 375), (380, 400)])
arrow([(1120, 285), (1120, 310)]); arrow([(1120, 375), (1120, 400)])

# 3. the host builds the container
panel(30, 520, 1440, 165, "The host builds the container from the Dockerfile (once, when you deploy)", "#f0fdf4")
steps = [["Start from Python 3.11", "a small base image"], ["Install system libraries", "libgl and libglib for OpenCV"],
         ["pip install requirements.txt", "the one requirements file"], ["Apple Vision OCR", "built into macOS: nothing to download"],
         ["Copy the app", "the helping_eyes/ folder"]]
for i, text in enumerate(steps):
    box(55 + i * 280, 575, 250, 80, text, "#fff", GREEN)
    if i:
        arrow([(55 + i * 280 - 30, 615), (55 + i * 280, 615)])
arrow([(380, 460), (380, 520)]); arrow([(1120, 460), (1120, 520)])

# 4. the container starts
box(250, 725, 1000, 80, ["The container starts: bash cloud/start.sh",
                         "runs uvicorn server:app on the host's port ($PORT, or 7860); the OCR loads in the background so the first visit is quick"],
    "#fff", GREEN)
arrow([(750, 685), (750, 725)])

# 5. people use it
panel(30, 845, 1440, 175, "Live: people use it", "#f3f4f6")
box(55, 885, 340, 80, ["A visitor's browser", "opens the https address: page, camera, mic"], "#fff", GREY)
box(515, 885, 360, 80, ["The server", "/ws/live, /api/capture, /api/book/page, /api/ask"], "#fff", GREEN)
box(915, 885, 250, 80, ["Gemini API", "writes the answers"], "#fefce8", "#a16207")
box(1195, 885, 250, 80, ["DuckDuckGo", "web lookup, only if you agree"], "#fefce8", "#a16207")
arrow([(750, 805), (750, 845)])
arrow([(395, 912), (515, 912)], "frames, voice", (455, 903)); arrow([(515, 942), (395, 942)], "hints, answers", (455, 960))
arrow([(875, 925), (915, 925)]); arrow([(1165, 925), (1195, 925)])
o.append('<text x="750" y="1008" text-anchor="middle" font-size="13" fill="#374151">A free host goes to sleep when nobody visits; the next visit wakes it (about a minute). '
         'Sessions are kept in memory, so run one copy only.</text>')
o.append("</svg>")

svg_path = os.path.join(HERE, "hosting.svg")
open(svg_path, "w").write("\n".join(o))
print("wrote", svg_path)

png_path = os.path.join(HERE, "hosting.png")
browsers = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chromium-browser"]
chrome = next((b for b in browsers if shutil.which(b)), None)
if chrome is None:
    print("No Chrome / Chromium found: open hosting.svg in a browser and export it as hosting.png")
else:
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                    "--force-device-scale-factor=1.5", f"--screenshot={png_path}", "file://" + svg_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("wrote", png_path)
