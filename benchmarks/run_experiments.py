from __future__ import annotations

import csv
import gc
import resource
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
import preview  # noqa: E402


INPUT = Path("/workspace/scratch/6ada4874c60d/upload/image(2).jpeg")
OUT_DIR = ROOT / "benchmarks" / "baseline_outputs"


CASES = {
    "01_default": dict(),
    "02_calm_edge": dict(strength=18, radius=24, turbulence=0.0,
                          turbulence_detail=1, noise_scale=400, dispersion=0.15,
                          dispersion_steps=3, iridescence=0.30),
    "03_wild": dict(strength=70, radius=64, turbulence=0.90,
                     turbulence_detail=6, noise_scale=120, dispersion=0.50,
                     dispersion_steps=48, iridescence=0.80),
    "04_large_soft": dict(detection_scale=4.0, threshold=20, contrast=1.4,
                           strength=140, radius=192, turbulence=0.75,
                           turbulence_detail=4, noise_scale=800, dispersion=0.35,
                           dispersion_steps=16, iridescence=0.55),
    "05_fine_turbulent": dict(detection_scale=0.5, threshold=60, contrast=0.8,
                               strength=60, radius=32, turbulence=1.0,
                               turbulence_detail=6, noise_scale=32, dispersion=0.70,
                               dispersion_steps=32, iridescence=0.70),
    "06_zero_dispersion_128": dict(strength=46, radius=48, turbulence=0.60,
                                    turbulence_detail=4, noise_scale=220,
                                    dispersion=0.0, dispersion_steps=128,
                                    iridescence=0.55),
    "07_classic_rgb": dict(strength=46, radius=48, turbulence=0.60,
                            turbulence_detail=4, noise_scale=220, dispersion=1.0,
                            dispersion_steps=3, iridescence=0.55),
    "08_max_ui_stress": dict(detection_scale=4.0, threshold=0, contrast=4.0,
                              strength=200, radius=256, turbulence=1.0,
                              turbulence_detail=6, noise_scale=800,
                              flow_speed=4.0, dispersion=1.0,
                              dispersion_steps=128, iridescence=1.0,
                              light_angle_deg=-180, seed=10000, phase_offset=10000),
}


def rss_mib() -> float:
    # Linux reports KiB; this benchmark runs in the Linux reference environment.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = preview.load_image(INPUT)
    rows = []
    outputs: list[tuple[str, Image.Image]] = []

    for name, overrides in CASES.items():
        params = preview.Params(**overrides)
        t0 = time.perf_counter()
        out = preview.render(src, params)
        elapsed = time.perf_counter() - t0
        output_path = OUT_DIR / f"{name}.png"
        preview.save_image(out, output_path)
        difference = float(abs(out - src).mean())
        rows.append({
            "case": name,
            "elapsed_s": f"{elapsed:.6f}",
            "proxy_fps": f"{1.0 / elapsed:.6f}",
            "mean_abs_rgb_delta": f"{difference:.6f}",
            "max_rss_mib": f"{rss_mib():.1f}",
            "params": repr(overrides),
        })
        preview_img = Image.open(output_path).convert("RGB")
        preview_img.thumbnail((384, 288), Image.Resampling.LANCZOS)
        outputs.append((name, preview_img.copy()))
        del out, preview_img
        gc.collect()
        print(rows[-1])

    with (OUT_DIR / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cell_w, cell_h = 420, 330
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(outputs):
        x = (i % 2) * cell_w
        y = (i // 2) * cell_h
        sheet.paste(image, (x + 18, y + 30))
        draw.text((x + 18, y + 8), name, fill="black")
    sheet.save(OUT_DIR / "contact_sheet.png")
    print(f"wrote {OUT_DIR / 'benchmark.csv'}")
    print(f"wrote {OUT_DIR / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
