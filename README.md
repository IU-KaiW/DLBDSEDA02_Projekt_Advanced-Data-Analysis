# Projekt: Advanced Data Analysis (DLBDSEDA02)
Ziel des Projekts ist es NLP-Techniken auf einem organisch entstandenen Datensatzgemäß der Aufgabenstellung der Hochschule auszuführen.
Aufgabe Nr. 1



## Datenaquisition (engl. data acquisition)
Trichter

### Datenrecherche (engl. )
Zunächst wird eine Onlinerecherche auf verschiedenen Plattformen (Kaggle, GitHub, ect. ) durchgeführt und nach geeigneten englischen und deutschen Datensätzen gesucht. Offensichtlich synthetisch erzeugte Varianten () werden ignoriert.

### Datensammlung (engl. )
Datenformat: csv

#### Download (engl. )
Datenquellen mit organisch vermutetem Ursprung werden lokal heruntergeladen und anschließend anhand eines KI-Detectors genauer geprüft.

#### Datenprüfung (engl. )
Die Instanzen (engl. samples) der Datensätze werden auf synthetisch erzeugte Varianten hin geprüft und im Anschluss getaggt.

anhand von Kriterien wie dem vermuteten Anteil realer oder synthetischer Samples sowie den nicht durch das Modell verarbeitbaren Samples (Real/Fake/Error) wird der Datensatz mit dem geringsten Real/Fake-Quotienten gewählt, da hier die Wahrscheinlichkeit eines organischen Ursprungs am höchsten erscheint. Der so identifizierte Datensatz geht in die NLP-Pipeline, welche mit der Textvorverarbeitung (engl. text pre-processing) des Datensatzes beginnt. Übersteigt die Anzahl der Samples eine Schwelle von 2000 wird der Datensatz zunächst aus Performancegründen gesplittet.
#### Datenauswahl (engl. ) 



## NLP-Pipeline

### Pre-Processing

### Processing

### Post-Processing

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#environment-variables">Environment Variables</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#configuration">Configuration</a></li>
  </ol>
</details>
