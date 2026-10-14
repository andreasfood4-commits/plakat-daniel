"""Variante A: Familienfoto als Farblithografie mit rotem Titelband.

Nimmt das Familienfoto, setzt es als Farblithografie um (Farbsteine,
Kreidekorn, versetzte Zeichnungsplatte)
und setzt es in ein Plakatlayout mit Titel- und Infotext.

Aufruf: python3 variante_a_foto.py <foto.jpg> <ausgabe.png>
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from litho import (B, H, font, freistellen, lithografie, papier, passende_schrift,
                   text_mitte)

# --- Texte (Platzhalter in eckigen Klammern) -------------------------------
OBERZEILE = "ZERMATT · SCHWEIZ"
TITEL = "DANIEL 70"
UNTERTITEL = "EINLADUNG ZUM FAMILIENWOCHENENDE"
INFO = "SA 31. JULI – SO 1. AUGUST 2027  ·  PENTHOUSE HEINZ JULEN"
ZUSATZ = "MIT DER GANZEN FAMILIE · KINDER, PARTNER & ENKEL"

# --- Zusätzliche Personen im Vordergrund ------------------------------------
# (Foto, horizontale Mitte, Höhe, Unterkante) – jeweils als Anteil der Bildgrösse;
# eine Unterkante über 1 schneidet die Person unten an.
VORDERGRUND = [
    ("enkel_1.jpg", 0.53, 0.52, 1.15),
]

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
    (96, 56, 36),     # Fleecebraun dunkel
    (140, 86, 52),    # Fleecebraun
    (140, 162, 192),  # Jeansblau
    (150, 176, 166),  # Salbeigrün
], dtype=np.float32)

RAND = 70


def mit_vordergrund(foto, groesse):
    """Familienfoto in Zielgrösse, davor die zusätzlichen Personen."""
    bild = foto.convert("RGB").resize((groesse, groesse), Image.LANCZOS)
    for datei, mitte, hoehe, unten in VORDERGRUND:
        person = freistellen(ImageOps.exif_transpose(Image.open(Path(__file__).parent / datei)))
        # Innenaufnahme ans Tageslicht der Familie angleichen:
        # Weissabgleich je Kanal (nur über die Person), dann etwas aufhellen
        rgb = ImageOps.autocontrast(person.convert("RGB"), cutoff=1,
                                    mask=person.getchannel("A"), preserve_tone=False)
        rgb = rgb.point(lambda v: round(255 * (v / 255) ** 0.85))
        person = Image.merge("RGBA", (*rgb.split(), person.getchannel("A")))
        h = round(groesse * hoehe)
        person = person.resize((round(person.width * h / person.height), h), Image.LANCZOS)
        x = round(groesse * mitte - person.width / 2)
        y = round(groesse * unten) - h
        # Tuschekontur um die Person, damit sie sich von der Familie abhebt
        alpha = np.asarray(person.getchannel("A")) > 127
        ring = ndi.binary_dilation(alpha, iterations=6) & ~alpha
        kontur = Image.new("RGBA", person.size, (*TINTE, 0))
        kontur.putalpha(Image.fromarray(ring.astype(np.uint8) * 255))
        bild.paste(kontur, (x, y), kontur)
        bild.paste(person, (x, y), person)
    return bild


def main(src, ziel):
    plakat = Image.new("RGB", (B, H), PAPIER)
    d = ImageDraw.Draw(plakat)

    # Oberzeile
    text_mitte(d, RAND + 30, OBERZEILE, font("josefin700", 78), TINTE, sperr=22)

    # Bildfeld
    bild_y = RAND + 160
    bild_g = B - 2 * RAND
    motiv = lithografie(mit_vordergrund(Image.open(src), bild_g), PALETTE, TINTE)
    plakat.paste(motiv, (RAND, bild_y))
    d.rectangle([RAND, bild_y, RAND + bild_g - 1, bild_y + bild_g - 1],
                outline=TINTE, width=8)

    # Titelband
    band_y = bild_y + bild_g
    band_h = 470
    d.rectangle([RAND, band_y, B - RAND, band_y + band_h], fill=ROT)
    f_titel = passende_schrift(d, TITEL, "bebas", 440, B - 2 * RAND - 160, sperr=24)
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

    plakat = papier(plakat, PAPIER)
    plakat.save(ziel, dpi=(300, 300))
    plakat.resize((B // 4, H // 4), Image.LANCZOS).save(
        str(Path(ziel).with_suffix("")) + "_vorschau.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
