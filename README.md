# Projekt: Advanced Data Analysis (DLBDSEDA02)
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Inhaltsverzeichnis</summary>
  <ol>
    <li><a href="##aufgabe-11-nlp-techniken-anwenden-um-eine-textsammlng-zu-analyieren">Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlng zu analyieren</a></li>
    <li><a href="#phase1">Phase 1 - Konzeptionsphase</a>
      <ul>
        <li><a href="#prerequisites">Datenaquisition (engl. data acquisition)</a></li>
        <li><a href="#nlp-pipeline">NLP-Pipeline</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#nlp-pipeline">NLP-Pipeline</a></li>
      </ul>
    <li><a href="#phase2">Phase 2 - Erarbeitungsphase</a>
    <li><a href="#phase3">Phase 3 - Finalisierungsphase</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#nlp-pipeline">NLP-Pipeline</a></li>
      </ul>
    
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#configuration">Configuration</a></li>x
  </ol>
</details>

## Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlng zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlihen Beschwerden anzuwenden und so, die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden.


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
1. Schriftliches Konzept
2. Datenverarbeitung
2.1. Vektorisierung 2 Techniken
2.2. Extraktion von Themen 2 Ansätze.

### Pre-Processing

### Processing

### Post-Processing