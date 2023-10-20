
![Titelbild](https://github.com/patschue/DLBS_landslides/assets/84413011/e5cdb28d-6702-46e7-9588-e7b53e43527f)

# Deep Learning for Landslide Detection (DLBS)

Dieses Repository enthält Materialien und den Code für das "Deep Learning for Landslide Detection" (DLBS) Modul, in dem Methoden der semantischen Bildsegmentierung zur Klassifizierung von Erdrutschen verwendet werden.  
Das Hauptziel dieses Projekts ist es, Erdrutsche auf Bildern mithilfe von Deep Learning-Techniken effektiv zu identifizieren und zu klassifizieren.  
Die Daten stammen ursprünglich von der [Challenge LandSlide4Sense](https://www.iarai.ac.at/landslide4sense/challenge/), ich verwende eine aufbereitete Version von [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/landslide-divided).  
Am Schluss zusätzlich verwendete Bilder stammen von [Planet Labs](https://www.planet.com/gallery/).

## Berichte und weitere Ressourcen

Kurzbericht zur Mini-Challenge: [DLBS Mini-Challenge Report](https://wandb.ai/patschue/DLBS%20Landslides%20FCN/reports/DLBS-Mini-Challenge-Report--Vmlldzo1NzE5OTc0)  
Ausführlicher Bericht: [DLBS Mini-Challenge Report lang](https://wandb.ai/patschue/DLBS%20Landslides%20FCN/reports/DLBS-Mini-Challenge-Report-lang--Vmlldzo1NzE3Njg0)

## Voraussetzungen und Installation

Der Code wurde auf der CSCS-Plattform ausgeführt.  
Als Kernel wurde der vom Deep Dive für Objekterkennung verwendet. Eine Kopie der verwendeten Libraries befindet sich im requirements.txt

> pip install -r requirements.txt

## Verzeichnisstruktur

- DLBS_MC_PatrickS.ipynb: Das Jupyter-Notebook, das den Hauptcode für das Projekt enthält.
- data/: Verzeichnis, das die Bilder für das Training, Testen und Validation enthält.
- helpers/: Enthält Hilfscode-Blöcke, die aus Gründen der Übersichtlichkeit aus dem Hauptnotebook ausgelagert wurden.
- weights/: Dieser Ordner ist anfänglich leer und ist vorgesehen für die Speicherung der trainierten Modellgewichte.
- Cropping: Beinhaltet den Code, der für das Zuschneiden der Trainingsbilder verwendet wurde. Dieser Schritt ist nicht wiederholbar, da er bereits abgeschlossen wurde.
