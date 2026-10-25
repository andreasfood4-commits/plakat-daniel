"""Variante C: gezeichnetes Werbeplakat – Familie rastet vor dem Matterhorn.

Das Foto wird stärker vereinfacht als in Variante A: grössere Farbflächen,
weniger Druckfarben, kräftige Konturen. Der grosse Titel steht im Himmel,
die Angaben auf einem Papierfeld darunter.

Aufruf: python3 variante_c_zeichnung.py [--plakat] <foto.jpg> <ausgabe.png> [links oben breite]
  --plakat: knapper Plakattext statt des vollständigen Flyertexts
  Bildausschnitt: linke und obere Kante sowie Breite als Anteil der Fotobreite
  bzw. -höhe; die Höhe ergibt sich aus dem Bildfeld.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageOps

from litho import B, H, font, lithografie, papier, passende_schrift, text_mitte

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


def flyer_text(d, y, bb):
    """Vollständiger Einladungstext für die Flyer-Version."""
    text_mitte(d, y, UNTERTITEL, passende_schrift(d, UNTERTITEL, "oswald700", 110, bb, 10),
               TINTE, sperr=10)
    text_mitte(d, y + 180, INFO, passende_schrift(d, INFO, "oswald400", 70, bb, 4), TINTE, sperr=4)
    text_mitte(d, y + 290, ZUSATZ, font("josefin600", 52), ROT, sperr=8)


def main(src, ziel, links=0.0, oben=0.0, breite=1.0, plakat_text=False):
    plakat = Image.new("RGB", (B, H), PAPIER)
    d = ImageDraw.Draw(plakat)

    # Bildfeld bis zum Papierfeld; Foto passend zuschneiden
    bb, bh = B - 2 * RAND, H - RAND - FELD
    foto = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    x0, y0, w = round(foto.width * links), round(foto.height * oben), round(foto.width * breite)
    hoehe = round(w * bh / bb)
    y0 = min(y0, foto.height - hoehe)
    foto = foto.crop((x0, y0, x0 + w, y0 + hoehe)).resize((bb, bh), Image.LANCZOS)
    foto = ruhige_flaechen_glaetten(foto)
    # Gezeichnet: grosse Flächen (starker Median), kräftige Konturen
    motiv = lithografie(foto, PALETTE, TINTE, seed=11, median=9,
                        kontrast=1.1, saettigung=1.15, korn=0.35)
    plakat.paste(motiv, (RAND, RAND))
    d.rectangle([RAND, RAND, RAND + bb - 1, RAND + bh - 1], outline=TINTE, width=8)

    # Titel im Himmel: Papierfarbe mit versetztem Tintenschatten
    f_ober = font("josefin700", 70)
    text_mitte(d, RAND + 110, OBERTITEL, f_ober, PAPIER, sperr=24)
    f_titel = passende_schrift(d, TITEL, "bebas", 620, bb - 260, sperr=26)
    y = RAND + 220
    text_mitte(d, y + 16, TITEL, f_titel, TINTE, sperr=26)
    text_mitte(d, y, TITEL, f_titel, PAPIER, sperr=26)

    # Angaben im Papierfeld
    y = RAND + bh + 80
    if plakat_text:
        text_mitte(d, y, PLAKAT_ORT, font("oswald700", 170), ROT, sperr=40)
        text_mitte(d, y + 260, PLAKAT_INFO,
                   passende_schrift(d, PLAKAT_INFO, "oswald400", 70, bb, 4), TINTE, sperr=4)
    else:
        flyer_text(d, y, bb)

    plakat = papier(plakat, PAPIER, wolkig=0.015)
    plakat.save(ziel, dpi=(300, 300))
    plakat.resize((B // 4, H // 4), Image.LANCZOS).save(
        str(Path(ziel).with_suffix("")) + "_vorschau.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    plakat_text = "--plakat" in args
    args = [a for a in args if a != "--plakat"]
    main(args[0], args[1], *map(float, args[2:5]), plakat_text=plakat_text)
