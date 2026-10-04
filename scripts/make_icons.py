from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BG_TOP = (11, 16, 32)
BG_BOTTOM = (8, 11, 20)
BRAND_A = (59, 130, 246)
BRAND_B = (139, 92, 246)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render(size: int, rounded: bool) -> Image.Image:
    s = 1024
    img = Image.new("RGBA", (s, s))
    px = img.load()
    for y in range(s):
        row = _lerp(BG_TOP, BG_BOTTOM, y / (s - 1))
        for x in range(s):
            px[x, y] = row + (255,)

    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((212, 212, 812, 812), fill=BRAND_A + (110,))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(90)))

    ring = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    grad = Image.new("RGBA", (s, s))
    gpx = grad.load()
    for y in range(s):
        for x in range(s):
            gpx[x, y] = _lerp(BRAND_A, BRAND_B, (x + y) / (2 * s - 2)) + (255,)
    mask = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((262, 262, 762, 762), fill=255)
    d.ellipse((372, 372, 652, 652), fill=0)
    d.rectangle((300, 488, 724, 536), fill=255)
    ring.paste(grad, (0, 0), mask)
    img = Image.alpha_composite(img, ring)

    if rounded:
        corner = Image.new("L", (s, s), 0)
        ImageDraw.Draw(corner).rounded_rectangle((0, 0, s - 1, s - 1), radius=230, fill=255)
        out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        out.paste(img, (0, 0), corner)
        img = out
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate app icons for web/PWA, macOS and iOS.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root = Path(args.root)

    web = root / "web" / "static" / "icons"
    web.mkdir(parents=True, exist_ok=True)
    full = render(1024, rounded=False)
    for n in (192, 512):
        full.resize((n, n), Image.LANCZOS).save(web / f"icon-{n}.png")
    full.resize((180, 180), Image.LANCZOS).convert("RGB").save(web / "apple-touch-icon.png")

    mac = root / "packaging" / "macos"
    mac.mkdir(parents=True, exist_ok=True)
    render(1024, rounded=True).save(mac / "occlubio.icns",
                                    sizes=[(16, 16), (32, 32), (64, 64), (128, 128),
                                           (256, 256), (512, 512), (1024, 1024)])
    full.convert("RGB").save(mac / "icon-1024.png")

    ios = root / "ios-client" / "resources"
    ios.mkdir(parents=True, exist_ok=True)
    full.convert("RGB").save(ios / "icon-1024.png")
    print("icons written:", web, mac, ios)


if __name__ == "__main__":
    main()
