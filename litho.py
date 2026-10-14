"""Gemeinsame Bausteine für die Plakatvarianten im Stil der Farblithografie.

Farbsteine (feste Druckfarben), Kreidekorn in den Übergängen, versetzte
Zeichnungsplatte, Papierstruktur, Schrift und Freistellen von Personen.
"""
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFont
from scipy import ndimage as ndi
from scipy.special import ndtr

ORDNER = Path(__file__).parent
FONTS = ORDNER / "fonts"
MODELL = ORDNER / "modelle" / "selfie_multiclass_256x256.tflite"
MODELL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
              "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")

B, H = 2480, 3508  # A4 bei 300 dpi
HELL = np.array([0.3, 0.59, 0.11], dtype=np.float32)


def font(name, size):
    return ImageFont.truetype(str(FONTS / f"{name}.ttf"), size)


def kreidekorn(form, seed, skala=1.3):
    """Gleichverteiltes, leicht klumpiges Korn wie Lithokreide auf dem Stein."""
    rng = np.random.default_rng(seed)
    n = ndi.gaussian_filter(rng.normal(size=form), skala)
    return ndtr(n / n.std())


def lithografie(img, palette, tinte, seed=1, median=5, linien=True,
                kontrast=1.15, saettigung=1.1):
    """Bild (bereits in Zielgrösse) als Farblithografie umsetzen."""
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(kontrast)
    img = ImageEnhance.Color(img).enhance(saettigung)
    for _ in range(2):
        img = img.filter(ImageFilter.MedianFilter(median))
    px = np.asarray(img, dtype=np.float32)

    # Abstand zu jeder Druckfarbe; nächste und zweitnächste bestimmen
    w = np.array([0.35, 0.45, 0.2], dtype=np.float32)
    d = np.stack([(((px - p) ** 2) * w).sum(-1) for p in palette], -1)
    i1 = d.argmin(-1)
    d1 = np.take_along_axis(d, i1[..., None], -1)[..., 0]
    np.put_along_axis(d, i1[..., None], np.inf, -1)
    i2 = d.argmin(-1)
    d2 = np.take_along_axis(d, i2[..., None], -1)[..., 0]

    # Krümel entfernen, damit klare Farbflächen bleiben
    i1 = np.asarray(Image.fromarray(i1.astype(np.uint8)).filter(ImageFilter.ModeFilter(7)))
    i2 = np.asarray(Image.fromarray(i2.astype(np.uint8)).filter(ImageFilter.ModeFilter(7)))

    # Übergänge nicht weich, sondern als Kreidekorn zwischen zwei Steinen
    t = np.sqrt(d1) / (np.sqrt(d1) + np.sqrt(d2) + 1e-6)
    t = ndi.gaussian_filter(t, 2.5)
    anteil = np.clip((t - 0.34) / 0.16, 0, 1) * 0.5
    # Dunkle Partien druckt der Tintenstein satt, ohne Korn
    hell = palette @ HELL
    anteil[(hell[i1] < 80) & (hell[i2] < 110)] = 0
    idx = np.where(kreidekorn(t.shape, seed, 1.8) < anteil, i2, i1)
    out = palette[idx]

    if linien:
        # Zeichnungsplatte: Konturen in Tinte, körnig und leicht versetzt gedruckt
        grau = ndi.gaussian_filter(px @ HELL, 2)
        kante = np.hypot(ndi.sobel(grau, 0), ndi.sobel(grau, 1))
        linie = ndi.median_filter((kante > 70).astype(np.float32), 5)
        linie = ndi.gaussian_filter(linie, 1.0)
        linie *= kreidekorn(linie.shape, seed + 1, 0.9) > 0.18
        linie = np.roll(linie, (3, -2), axis=(0, 1))
        a = np.clip(linie, 0, 1)[..., None] * 0.8
        out = out * (1 - a) + np.array(tinte, dtype=np.float32) * a
    return Image.fromarray(out.astype(np.uint8), "RGB")


def papier(img, papierfarbe, seed=7, wolkig=0.03):
    """Druck auf Papier: Wolkigkeit, Faser, Stellen ohne Farbannahme."""
    a = np.asarray(img, dtype=np.float32)
    form = a.shape[:2]
    rng = np.random.default_rng(seed)
    wolke = ndi.gaussian_filter(rng.normal(size=form), 60)
    wolke = 1 + wolkig * wolke / wolke.std()
    faser = 1 + 0.01 * ndi.gaussian_filter(rng.normal(size=form), 0.7) / 0.3
    a = a * (wolke * faser)[..., None]
    # Winzige Fehlstellen, an denen das Papier durchscheint
    fehl = (kreidekorn(form, seed + 5, 1.0) > 0.996)[..., None]
    a = np.where(fehl, a * 0.4 + np.array(papierfarbe, dtype=np.float32) * 0.6, a)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def textbreite(draw, text, f, sperr=0):
    return sum(draw.textlength(c, font=f) for c in text) + sperr * (len(text) - 1)


def text_mitte(draw, y, text, f, fill, sperr=0, kontur=0, konturfarbe=None):
    """Zentrierter Text mit optionaler Sperrung und Kontur."""
    x = (B - textbreite(draw, text, f, sperr)) / 2
    for c in text:
        draw.text((x, y), c, font=f, fill=fill,
                  stroke_width=kontur, stroke_fill=konturfarbe)
        x += draw.textlength(c, font=f) + sperr


def passende_schrift(draw, text, name, start, max_breite, sperr=0):
    """Grösste Schriftgrösse, bei der der Text in max_breite passt."""
    groesse = start
    while textbreite(draw, text, font(name, groesse), sperr) > max_breite:
        groesse -= 10
    return font(name, groesse)


def personen_maske(foto):
    """Wahrscheinlichkeit 0..1 je Pixel, dass dort eine Person ist."""
    from ai_edge_litert.interpreter import Interpreter

    if not MODELL.exists():
        MODELL.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODELL_URL, MODELL)
    it = Interpreter(model_path=str(MODELL))
    it.allocate_tensors()
    ein, aus = it.get_input_details()[0], it.get_output_details()[0]
    h, w = ein["shape"][1:3]
    x = np.asarray(foto.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32) / 255
    it.set_tensor(ein["index"], x[None])
    it.invoke()
    o = it.get_tensor(aus["index"])[0]
    e = np.exp(o - o.max(-1, keepdims=True))
    p = 1 - e[..., 0] / e.sum(-1)  # alles ausser Hintergrund
    return Image.fromarray((p * 255).astype(np.uint8)).resize(foto.size, Image.BILINEAR)
