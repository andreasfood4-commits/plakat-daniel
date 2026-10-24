# Plakat Daniel – Familienwochenende 1. August 2027

Einladung im Stil Schweizer Tourismusplakate (Farblithografie), in zwei Varianten.

| Variante | Skript | Aktueller Entwurf |
|---|---|---|
| A – Familienfoto mit rotem Titelband | `variante_a_foto.py` | `entwurf_a9.png` |
| B – Familie vor dem Matterhorn von Emil Cardinaux (1908, gemeinfrei) | `variante_b_cardinaux.py` | `entwurf_b3.png` |
| C – gezeichnetes Werbeplakat, Familie rastet vor dem Matterhorn | `variante_c_zeichnung.py` | `entwurf_c1.png`, `entwurf_c2.png` |

    pip install pillow numpy scipy ai-edge-litert
    python3 variante_a_foto.py foto_familie_hd.jpg entwurf_a.png
    python3 variante_c_zeichnung.py foto_rast_1.jpg entwurf_c1.png 0.13 0.30 0.73
    python3 variante_c_zeichnung.py foto_rast_2.jpg entwurf_c2.png 0.07 0.15 0.76
    python3 variante_b_cardinaux.py foto_familie_hd.jpg vorlage_cardinaux_zermatt.jpg entwurf_b.png

Weitere Personen (z. B. Enkel) stehen in Variante A in der Liste `VORDERGRUND`
oben im Skript: Foto, Position und Grösse.

Gemeinsame Bausteine (Farbsteine, Kreidekorn, Papier, Freistellen) liegen in
`litho.py`. Variante B lädt beim ersten Lauf das MediaPipe-Modell zum
Freistellen der Personen nach `modelle/` (nicht im Repo).

Schriften: Bebas Neue, Oswald, Josefin Sans, Abril Fatface, Days One
(SIL Open Font License).
