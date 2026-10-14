"""Geburtstags-Flyer im Stil alter Schweizer Tourismusplakate.

Nimmt das Familienfoto, setzt es als Farblithografie um (Farbsteine,
Kreidekorn, versetzte Zeichnungsplatte)
und setzt es in ein Plakatlayout mit Titel- und Infotext.

Aufruf: python3 plakat.py <foto.jpg> <ausgabe.png>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
from scipy import ndimage as ndi
from scipy.special import ndtr

FONTS = Path(__file__).parent / "fonts"

# --- Texte (Platzhalter in eckigen Klammern) -------------------------------
OBERZEILE = "ZERMATT · SCHWEIZ"
TITEL = "DANIEL [ALTER]"
UNTERTITEL = "EINLADUNG ZUM FAMILIENWOCHENENDE"
INFO = "SA 31. JULI – SO 1. AUGUST 2027  ·  [ORT]"
ZUSATZ = "MIT DER GANZEN FAMILIE · KINDER, PARTNER & ENKEL"

# --- Farbpalette: gedämpfte Steindruckfarben ---------------------------------
PAPIER = (241, 230, 205)
TINTE = (38, 44, 58)
ROT = (186, 48, 40)
PALETTE = np.array([
    (241, 230, 205),  # Papier / Wolken
    (206, 214, 214),  # helle Wolkenschatten
    (58, 110, 168),   # Himmelblau
    (31, 72, 128),    # tiefes Blau
    (38, 44, 58),     # Tinte
    (92, 88, 96),     # Felsgrau
    (150, 140, 132),  # helles Gestein
    (120, 128, 64),   # Olivgrün
    (176, 160, 86),   # Alpwiese / Ocker
    (214, 190, 120),  # Sandgelb
    (186, 48, 40),    # Jackenrot
    (128, 40, 36),    # Dunkelrot
    (236, 196, 168),  # Haut Licht
    (214, 160, 128),  # Haut hell
    (176, 120, 92),   # Haut Schatten
    (140, 92, 72),    # Haut tief
    (70, 58, 50),     # Dunkelbraun
], dtype=np.float32)

B, H = 2480, 3508  # A4 bei 300 dpi
RAND = 70


def font(name, size):
    return ImageFont.truetype(str(FONTS / f"{name}.ttf"), size)


def kreidekorn(form, seed, skala=1.3):
    """Gleichverteiltes, leicht klumpiges Korn wie Lithokreide auf dem Stein."""
    rng = np.random.default_rng(seed)
    n = ndi.gaussian_filter(rng.normal(size=form), skala)
    return ndtr(n / n.std())


def lithografie(foto, groesse, seed=1):
    """Foto als Farblithografie: Farbsteine, Kreidekorn, Zeichnungsplatte."""
    img = foto.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = img.resize((groesse, groesse), Image.LANCZOS)
    for _ in range(2):
        img = img.filter(ImageFilter.MedianFilter(5))
    px = np.asarray(img, dtype=np.float32)

    # Abstand zu jeder Druckfarbe; nächste und zweitnächste bestimmen
    w = np.array([0.35, 0.45, 0.2], dtype=np.float32)
    d = np.stack([(((px - p) ** 2) * w).sum(-1) for p in PALETTE], -1)
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
    hell = PALETTE @ np.array([0.3, 0.59, 0.11], dtype=np.float32)
    anteil[(hell[i1] < 80) & (hell[i2] < 110)] = 0
    idx = np.where(kreidekorn(t.shape, seed, 1.8) < anteil, i2, i1)
    out = PALETTE[idx]

    # Zeichnungsplatte: Konturen in Tinte, körnig und leicht versetzt gedruckt
    grau = ndi.gaussian_filter(px @ np.array([0.3, 0.59, 0.11], dtype=np.float32), 2)
    kante = np.hypot(ndi.sobel(grau, 0), ndi.sobel(grau, 1))
    linie = ndi.median_filter((kante > 70).astype(np.float32), 5)
    linie = ndi.gaussian_filter(linie, 1.0)
    linie *= kreidekorn(linie.shape, seed + 1, 0.9) > 0.18
    linie = np.roll(linie, (3, -2), axis=(0, 1))
    a = np.clip(linie, 0, 1)[..., None] * 0.8
    out = out * (1 - a) + np.array(TINTE, dtype=np.float32) * a
    return Image.fromarray(out.astype(np.uint8), "RGB")


def papier(img, seed=7):
    """Druck auf Papier: Wolkigkeit, Faser, Stellen ohne Farbannahme."""
    a = np.asarray(img, dtype=np.float32)
    form = a.shape[:2]
    rng = np.random.default_rng(seed)
    wolke = ndi.gaussian_filter(rng.normal(size=form), 60)
    wolke = 1 + 0.03 * wolke / wolke.std()
    faser = 1 + 0.01 * ndi.gaussian_filter(rng.normal(size=form), 0.7) / 0.3
    a = a * (wolke * faser)[..., None]
    # Winzige Fehlstellen, an denen das Papier durchscheint
    papierfarbe = np.array(PAPIER, dtype=np.float32)
    fehl = (kreidekorn(form, seed + 5, 1.0) > 0.996)[..., None]
    a = np.where(fehl, a * 0.4 + papierfarbe * 0.6, a)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def text_mitte(draw, y, text, f, fill, sperr=0):
    """Zentrierter Text mit optionaler Sperrung (Buchstabenabstand)."""
    breiten = [draw.textlength(c, font=f) for c in text]
    gesamt = sum(breiten) + sperr * (len(text) - 1)
    x = (B - gesamt) / 2
    for c, bw in zip(text, breiten):
        draw.text((x, y), c, font=f, fill=fill)
        x += bw + sperr
    return gesamt


def main(src, ziel):
    plakat = Image.new("RGB", (B, H), PAPIER)
    d = ImageDraw.Draw(plakat)

    # Oberzeile
    text_mitte(d, RAND + 30, OBERZEILE, font("josefin700", 78), TINTE, sperr=22)

    # Bildfeld
    bild_y = RAND + 160
    bild_g = B - 2 * RAND
    motiv = lithografie(Image.open(src), bild_g)
    plakat.paste(motiv, (RAND, bild_y))
    d.rectangle([RAND, bild_y, RAND + bild_g - 1, bild_y + bild_g - 1],
                outline=TINTE, width=8)

    # Titelband
    band_y = bild_y + bild_g
    band_h = 470
    d.rectangle([RAND, band_y, B - RAND, band_y + band_h], fill=ROT)
    groesse = 440
    while True:  # Titel so gross wie möglich, aber innerhalb des Bandes
        f_titel = font("bebas", groesse)
        breite = sum(d.textlength(c, font=f_titel) for c in TITEL) + 24 * (len(TITEL) - 1)
        if breite <= B - 2 * RAND - 160:
            break
        groesse -= 10
    tb = d.textbbox((0, 0), TITEL, font=f_titel)
    ty = band_y + (band_h - (tb[3] - tb[1])) / 2 - tb[1]
    # Versatz-Schatten wie beim mehrfarbigen Steindruck
    text_mitte(d, ty + 14, TITEL, f_titel, TINTE, sperr=24)
    text_mitte(d, ty, TITEL, f_titel, PAPIER, sperr=24)

    # Untertitel und Infozeile
    y = band_y + band_h + 55
    text_mitte(d, y, UNTERTITEL, font("oswald700", 104), TINTE, sperr=10)
    text_mitte(d, y + 165, INFO, font("oswald400", 66), TINTE, sperr=4)
    text_mitte(d, y + 265, ZUSATZ, font("josefin600", 50), ROT, sperr=8)

    # Äusserer Rahmen
    d.rectangle([RAND // 2, RAND // 2, B - RAND // 2, H - RAND // 2],
                outline=TINTE, width=6)

    plakat = papier(plakat)
    plakat.save(ziel, dpi=(300, 300))
    plakat.resize((B // 4, H // 4), Image.LANCZOS).save(
        str(Path(ziel).with_suffix("")) + "_vorschau.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
