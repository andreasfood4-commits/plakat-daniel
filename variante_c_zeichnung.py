"""Variante C: gezeichnetes Werbeplakat – Familie rastet vor dem Matterhorn.

Das Foto wird stärker vereinfacht als in Variante A: grössere Farbflächen,
weniger Druckfarben, kräftige Konturen. Der grosse Titel steht im Himmel,
die Angaben auf einem Papierfeld darunter.

Aufruf: python3 variante_c_zeichnung.py [--plakat] [--format=75x100] <foto.jpg> <ausgabe.png>
                                         [links oben breite]
  --plakat: knapper Plakattext statt des vollständigen Flyertexts
  --format: a4 (Standard, 300 dpi) oder 75x100 (75 x 100 cm, 150 dpi)
  Bildausschnitt: linke und obere Kante sowie Breite als Anteil der Fotobreite
  bzw. -höhe; die Höhe ergibt sich aus dem Bildfeld.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageOps

from litho import (B, H, cardinaux_schrift, font, lithografie, papier, passende_schrift,
                   text_mitte)

# --- Texte -------------------------------------------------------------------
TITEL = "DANIEL 70"
OBERTITEL = "ZERMATT · SCHWEIZ"
UNTERTITEL = "EINLADUNG ZUM FAMILIENWOCHENENDE"
INFO = "DO 29. JULI – DI 3. AUGUST 2027  ·  PENTHOUSE HEINZ JULEN"
ZUSATZ = "MIT DER GANZEN FAMILIE · KINDER, PARTNER & ENKEL"
# Plakat-Version: knapper Text im Papierfeld
PLAKAT_ORT = "ZERMATT"
PLAKAT_INFO = "FAMILIENWOCHENENDE  ·  29. JULI – 3. AUGUST 2027  ·  PENTHOUSE HEINZ JULEN"

# --- Farben ------------------------------------------------------------------
PAPIER = (242, 234, 214)
TINTE = (28, 32, 48)
ROT = (186, 52, 44)
CARDINAUX_ORANGE = (196, 96, 44)  # Schriftfarbe des Zermatt-Plakats von Emil Cardinaux
PALETTE = np.array([
    (28, 32, 48),     # Tinte
    (34, 78, 160),    # Himmel tief
    (58, 112, 196),   # Himmel
    (104, 156, 222),  # Himmel hell
    (168, 198, 234),  # Himmel Dunst
    (242, 242, 236),  # Schnee
    (172, 186, 206),  # Schneeschatten
    (62, 70, 92),     # Fels dunkel
    (104, 112, 132),  # Fels
    (214, 198, 140),  # Gras Stroh
    (178, 156, 88),   # Gras Ocker
    (120, 116, 60),   # Gras Oliv
    (70, 68, 40),     # Gras tief
    (132, 104, 44),   # Shirt Olivbraun
    (168, 170, 176),  # Shirt Grau
    (190, 58, 56),    # Shirt Rot
    (40, 50, 104),    # Marine
    (222, 172, 138),  # Haut
    (168, 116, 88),   # Haut Schatten
    (222, 194, 176),  # Hose Rosé
    (62, 46, 36),     # Haar dunkel
    (200, 168, 108),  # Haar blond
], dtype=np.float32)

RAND = 100
FELD = 690  # Höhe des Papierfelds für die Angaben


def ruhige_flaechen_glaetten(img, sigma=10, schwelle=6):
    """Wo das Foto kaum Struktur hat (Himmel), stark weichzeichnen.

    So entstehen beim Drucken klare Farbstufen statt Flimmern; Berge,
    Menschen und Gras behalten ihre Zeichnung.
    """
    a = np.asarray(img, dtype=np.float32)
    grau = a @ np.array([0.3, 0.59, 0.11], dtype=np.float32)
    mittel = ndi.uniform_filter(grau, 15)
    streuung = np.sqrt(np.maximum(ndi.uniform_filter(grau ** 2, 15) - mittel ** 2, 0))
    ruhig = ndi.gaussian_filter((streuung < schwelle).astype(np.float32), 6)[..., None]
    weich = np.stack([ndi.gaussian_filter(a[..., k], sigma) for k in range(3)], -1)
    return Image.fromarray((a * (1 - ruhig) + weich * ruhig).astype(np.uint8))


# Ausgabeformate: Breite und Höhe in Pixeln, Auflösung in dpi
FORMATE = {
    "a4": (B, H, 300),                 # 21 x 29,7 cm
    "75x100": (4429, 5906, 150),       # 75 x 100 cm
}


def flyer_text(d, y, bb, k):
    """Vollständiger Einladungstext für die Flyer-Version."""
    text_mitte(d, y, UNTERTITEL,
               passende_schrift(d, UNTERTITEL, "oswald700", round(110 * k), bb, 10 * k),
               TINTE, sperr=10 * k)
    text_mitte(d, y + 180 * k, INFO, passende_schrift(d, INFO, "oswald400", round(70 * k), bb, 4 * k),
               TINTE, sperr=4 * k)
    text_mitte(d, y + 290 * k, ZUSATZ, font("josefin600", round(52 * k)), ROT, sperr=8 * k)


def main(src, ziel, links=0.0, oben=0.0, breite=1.0, plakat_text=False, format="a4"):
    W, Hh, dpi = FORMATE[format]
    k = W / B  # Massstab gegenüber dem A4-Layout
    rand, feld = round(RAND * k), round(FELD * k)
    plakat = Image.new("RGB", (W, Hh), PAPIER)
    d = ImageDraw.Draw(plakat)

    # Bildfeld bis zum Papierfeld; Foto passend zuschneiden
    bb, bh = W - 2 * rand, Hh - rand - feld
    foto = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    x0, y0, w = round(foto.width * links), round(foto.height * oben), round(foto.width * breite)
    hoehe = round(w * bh / bb)
    y0 = min(y0, foto.height - hoehe)
    # Das Motiv im A4-Massstab drucken, damit Korn und Flächen gleich wirken,
    # dann auf die Zielgrösse bringen
    mb, mh = round(bb / k), round(bh / k)
    foto = foto.crop((x0, y0, x0 + w, y0 + hoehe)).resize((mb, mh), Image.LANCZOS)
    foto = ruhige_flaechen_glaetten(foto)
    # Gezeichnet: grosse Flächen (starker Median), kräftige Konturen
    motiv = lithografie(foto, PALETTE, TINTE, seed=11, median=9,
                        kontrast=1.1, saettigung=1.15, korn=0.35)
    if k != 1:
        motiv = motiv.resize((bb, bh), Image.LANCZOS)
    plakat.paste(motiv, (rand, rand))
    d.rectangle([rand, rand, rand + bb - 1, rand + bh - 1], outline=TINTE, width=round(8 * k))

    # Titel im Himmel: Papierfarbe mit versetztem Tintenschatten
    f_ober = font("josefin700", round(70 * k))
    text_mitte(d, rand + 110 * k, OBERTITEL, f_ober, PAPIER, sperr=24 * k)
    f_titel = passende_schrift(d, TITEL, "bebas", round(620 * k), bb - 260 * k, sperr=26 * k)
    y = rand + 220 * k
    text_mitte(d, y + 16 * k, TITEL, f_titel, TINTE, sperr=26 * k)
    text_mitte(d, y, TITEL, f_titel, PAPIER, sperr=26 * k)

    # Angaben im Papierfeld
    y = rand + bh + 80 * k
    if plakat_text:
        # "ZERMATT" im Handlettering des Cardinaux-Plakats
        cardinaux_schrift(plakat, PLAKAT_ORT, y, 210 * k, CARDINAUX_ORANGE, TINTE)
        text_mitte(d, y + 285 * k, PLAKAT_INFO,
                   passende_schrift(d, PLAKAT_INFO, "oswald400", round(70 * k), bb, 4 * k),
                   TINTE, sperr=4 * k)
    else:
        flyer_text(d, y, bb, k)

    plakat = papier(plakat, PAPIER, wolkig=0.015)
    plakat.save(ziel, dpi=(dpi, dpi))
    plakat.resize((W // 4, Hh // 4), Image.LANCZOS).save(
        str(Path(ziel).with_suffix("")) + "_vorschau.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    plakat_text = "--plakat" in args
    format = next((a.split("=", 1)[1] for a in args if a.startswith("--format=")), "a4")
    args = [a for a in args if not a.startswith("--")]
    main(args[0], args[1], *map(float, args[2:5]), plakat_text=plakat_text, format=format)
