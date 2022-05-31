#
Überblick
Template für die MatVec Aufgabe
Einfaches Beispiel für die Einführung in CUDA. 

# Files
Makefile
matvec.cu

# Aufgabe
1. Setzen Sie die Matrix-Vektor-Multiplikation
  $ Ax=y $ mit dem Ihnen zur Verfügung stehenden Wissen mittels CUDA um und messen
 Sie die Zeiten auf der GPU für eine Matrix $A\in \mathbb{R}^{n\times n}$ mit $n=1024 \cdot i$ für $i\in\{1,4,8,16\}$.\\
 Jeder Thread soll dabei eine komplette Zeile bearbeiten (1~dimensionale Grids, Blocks und Arrays).\\

2. Nutzen Sie die gleiche Matrix $A$ zu Berechnung von
  $A^Tx=y $, indem Sie die Indizes der Matrix $A$ vertauschen.\\
 Welchen Einfluss hat dies auf die Berechnungszeit?
 
3. Vergleichen Sie die Zeiten mit einer reinen CPU Implementierung.

Nutzen Sie die Templates in \texttt{02\_matvec} und geben Sie Ihren Code zusammen mit der Zeitmessung bis nächste Woche Montag in Moodle ab.


# TODO
Zusammen mit 01_beispiel in Moodle laden

