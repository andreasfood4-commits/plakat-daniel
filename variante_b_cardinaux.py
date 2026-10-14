"""Variante B: Familie vor dem Matterhorn von Emil Cardinaux (Zermatt, 1908).

Das gemeinfreie Plakatmotiv wird in Zielgrösse neu als Lithografie gedruckt,
die Familie aus dem Foto freigestellt und in den dunklen Vordergrund gestellt.
Die Schrift folgt dem Original: breite, gerundete Versalien mit Kontur.

Aufruf: python3 variante_b_cardinaux.py <foto.jpg> <vorlage.jpg> <ausgabe.png>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

from litho import (B, H, lithografie, papier, passende_schrift, personen_maske,
                   text_mitte, font)

# --- Texte -------------------------------------------------------------------
TITEL = "DANIEL 70"
ZEILE_1 = "FAMILIENWOCHENENDE  31. JULI – 1. AUGUST 2027"
ZEILE_2 = "PENTHOUSE HEINZ JULEN  ·  ZERMATT"
ZEILE_3 = "MIT DER GANZEN FAMILIE · KINDER, PARTNER & ENKEL"

# --- Farben (aus dem Original abgenommen) --------------------------------------
PAPIER = (240, 232, 214)
TINTE = (30, 27, 28)
ORANGE = (196, 96, 44)

# Motivausschnitt in der Vorlage (ohne Rahmenlinie), Pixel des Screenshots
MOTIV_BOX = (122, 62, 491, 508)

# Druckfarben für die Familie: Abendstimmung passend zum Original
FAMILIE_PALETTE = np.array([
    (30, 27, 28),     # Tinte
    (62, 58, 50),     # dunkles Oliv
    (40, 50, 82),     # Marine
    (70, 98, 146),    # Mittelblau
    (150, 168, 196),  # Hellblau
    (186, 58, 38),    # Jackenrot
    (122, 40, 30),    # Dunkelrot
    (238, 196, 158),  # Haut Licht
    (212, 152, 112),  # Haut
    (160, 102, 74),   # Haut Schatten
    (240, 228, 196),  # Creme
    (172, 152, 112),  # Khaki
    (112, 106, 100),  # Grau
    (200, 104, 48),   # Abendorange
    (236, 212, 140),  # Gelblicht
], dtype=np.float32)

RAND = 190
RADIUS = 36


# Druckfarben des Motivs, am Original abgenommen und leicht aufgefrischt
MOTIV_PALETTE = np.array([
    (24, 20, 20),     # Tinte
    (52, 48, 32),     # dunkles Oliv (Vordergrund)
    (98, 88, 54),     # Oliv hell
    (182, 132, 156),  # Abendhimmel violett
    (196, 152, 174),  # Abendhimmel hell
    (212, 110, 56),   # Alpenglühen Orange
    (148, 62, 36),    # Fels Rotbraun
    (196, 160, 72),   # Ocker
    (252, 226, 120),  # Glühgelb
    (250, 242, 206),  # Schnee
], dtype=np.float32)


def familie_freistellen(foto, breite):
    """Familie als Lithografie mit Tuschekontur, plus Deckmaske."""
    maske = np.asarray(personen_maske(foto), dtype=np.float32) / 255
    ys, xs = np.nonzero(maske > 0.5)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    massstab = breite / (box[2] - box[0])
    gross = (round(foto.width * massstab), round(foto.height * massstab))

    bild = lithografie(foto.resize(gross, Image.LANCZOS), FAMILIE_PALETTE, TINTE, seed=3)
    m = Image.fromarray((maske * 255).astype(np.uint8)).resize(gross, Image.BILINEAR)
    m = np.asarray(m.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255 > 0.5
    m = ndi.binary_opening(m, iterations=4)

    # Kräftige Umrisslinie wie bei Cardinaux
    rand = ndi.binary_dilation(m, iterations=9) & ~m
    a = np.asarray(bild, dtype=np.float32)
    a[rand] = TINTE
    deck = m | rand

    k = [round(v * massstab) for v in box]
    k[1] -= 12  # Platz für die Kontur über den Köpfen
    bild = Image.fromarray(a.astype(np.uint8)).crop(k)
    deck = Image.fromarray(deck.astype(np.uint8) * 255).crop(k)
    return bild, deck


def main(src, vorlage, ziel):
    plakat = Image.new("RGB", (B, H), PAPIER)
    d = ImageDraw.Draw(plakat)

    # Motiv: Original neu in Zielgrösse drucken
    original = Image.open(vorlage).convert("RGB").crop(MOTIV_BOX)
    mb = B - 2 * RAND
    mh = round(mb * original.height / original.width)
    motiv = lithografie(original.resize((mb, mh), Image.LANCZOS), MOTIV_PALETTE,
                        TINTE, seed=5, median=7, linien=False, kontrast=1.05, saettigung=1.3)

    # Familie in den dunklen Vordergrund stellen, unten angeschnitten
    familie, deck = familie_freistellen(Image.open(src), round(mb * 0.74))
    fx = (mb - familie.width) // 2
    fy = mh - familie.height + round(familie.height * 0.06)
    motiv.paste(familie, (fx, fy), deck)

    # Motiv mit abgerundeter Tuschekante aufs Papier
    ecken = Image.new("L", (mb, mh), 0)
    ImageDraw.Draw(ecken).rounded_rectangle([0, 0, mb - 1, mh - 1], RADIUS, fill=255)
    oben = 150
    plakat.paste(motiv, (RAND, oben), ecken)
    d.rounded_rectangle([RAND, oben, RAND + mb - 1, oben + mh - 1], RADIUS,
                        outline=TINTE, width=10)

    # Schrift wie im Original: Orange mit dunkler Kontur, darunter schwarz
    y = oben + mh + 70
    f_titel = passende_schrift(d, TITEL, "daysone", 330, mb - 200, sperr=20)
    text_mitte(d, y, TITEL, f_titel, ORANGE, sperr=20, kontur=7, konturfarbe=TINTE)
    y += f_titel.size + 60
    for text, groesse, farbe, abstand in [(ZEILE_1, 78, TINTE, 110),
                                          (ZEILE_2, 66, TINTE, 100),
                                          (ZEILE_3, 46, ORANGE, 0)]:
        f = passende_schrift(d, text, "daysone", groesse, mb, sperr=4)
        text_mitte(d, y, text, f, farbe, sperr=4)
        y += abstand

    plakat = papier(plakat, PAPIER, wolkig=0.012)
    plakat.save(ziel, dpi=(300, 300))
    plakat.resize((B // 4, H // 4), Image.LANCZOS).save(
        str(Path(ziel).with_suffix("")) + "_vorschau.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
