# Plakat Daniel – Familienwochenende 1. August 2027

Einladung im Stil Schweizer Tourismusplakate (Farblithografie), in zwei Varianten.

| Variante | Skript | Aktueller Entwurf |
|---|---|---|
| A – Familienfoto mit rotem Titelband | `variante_a_foto.py` | `entwurf_6.png` |
| B – Familie vor dem Matterhorn von Emil Cardinaux (1908, gemeinfrei) | `variante_b_cardinaux.py` | `entwurf_b2.png` |

    pip install pillow numpy scipy ai-edge-litert
    python3 variante_a_foto.py foto_original.jpg entwurf_a.png
    python3 variante_b_cardinaux.py foto_original.jpg vorlage_cardinaux_zermatt.jpg entwurf_b.png

Gemeinsame Bausteine (Farbsteine, Kreidekorn, Papier, Freistellen) liegen in
`litho.py`. Variante B lädt beim ersten Lauf das MediaPipe-Modell zum
Freistellen der Personen nach `modelle/` (nicht im Repo).

Schriften: Bebas Neue, Oswald, Josefin Sans, Abril Fatface, Days One
(SIL Open Font License).
