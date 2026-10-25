"""Gemeinsame Bausteine für die Plakatvarianten im Stil der Farblithografie.

Farbsteine (feste Druckfarben), Kreidekorn in den Übergängen, versetzte
Zeichnungsplatte, Papierstruktur, Schrift und Freistellen von Personen.
"""
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
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
                kontrast=1.15, saettigung=1.1, korn=1.0):
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
    anteil = np.clip((t - 0.34) / 0.16, 0, 1) * 0.5 * korn
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


def freistellen(foto, schwelle=0.6, weich=2):
    """Person als RGBA-Ausschnitt: grösste zusammenhängende Fläche der Maske."""
    m = np.asarray(personen_maske(foto), dtype=np.float32) / 255
    m = ndi.gaussian_filter(m, weich * foto.width / 640) > schwelle
    teile, n = ndi.label(m)
    if n > 1:
        groessen = ndi.sum(m, teile, range(1, n + 1))
        m = teile == (np.argmax(groessen) + 1)
    m = ndi.binary_fill_holes(m)
    ys, xs = np.nonzero(m)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    rgba = foto.convert("RGBA")
    rgba.putalpha(Image.fromarray(m.astype(np.uint8) * 255))
    return rgba.crop(box)


# --- Handlettering nach Emil Cardinaux (Zermatt, 1908) -------------------------
# Jeder Buchstabe als gleichmässig breite Pinselstriche mit runden Enden,
# Koordinaten in Einheiten der Versalhöhe (0 = oben, 100 = unten).
def _bogen(cx, cy, r, von, bis, schritte=14):
    w = np.linspace(np.radians(von), np.radians(bis), schritte)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in w]


CARDINAUX_GLYPHEN = {
    "Z": (74, [[(2, 2), (72, 2), (2, 98), (74, 98)]]),
    "E": (60, [[(60, 2), (2, 2), (2, 98), (62, 98)], [(2, 50), (48, 50)]]),
    "R": (66, [[(2, 98), (2, 2), (36, 2)] + _bogen(36, 27, 25, -90, 90) + [(2, 52)],
               [(30, 52), (66, 98)]]),
    "M": (84, [[(2, 98), (8, 2), (42, 72), (76, 2), (82, 98)]]),
    "A": (76, [[(2, 98), (38, 2), (74, 98)], [(16, 64), (60, 64)]]),
    "T": (74, [[(2, 2), (72, 2)], [(37, 2), (37, 98)]]),
}


def cardinaux_schrift(img, text, y, hoehe, farbe, konturfarbe, sperr=0.30, breit=1.12):
    """Text im Handlettering des Cardinaux-Plakats zentriert auf img zeichnen."""
    s = hoehe / 100
    strich = round(hoehe * 0.17)
    kontur = max(2, round(strich * 0.12))
    breiten = [CARDINAUX_GLYPHEN[c][0] * s * breit for c in text]
    x = (img.width - sum(breiten) - sperr * hoehe * (len(text) - 1)) / 2
    d = ImageDraw.Draw(img)
    zuege = []
    for c, b in zip(text, breiten):
        for zug in CARDINAUX_GLYPHEN[c][1]:
            zuege.append([(x + px * s * breit, y + py * s) for px, py in zug])
        x += b + sperr * hoehe
    # erst die dunkle Kontur, dann die Farbe darüber
    for breite, f in ((strich + 2 * kontur, konturfarbe), (strich, farbe)):
        for zug in zuege:
            d.line(zug, fill=f, width=breite, joint="curve")
            for px, py in (zug[0], zug[-1]):
                r = breite / 2
                d.ellipse([px - r, py - r, px + r, py + r], fill=f)
