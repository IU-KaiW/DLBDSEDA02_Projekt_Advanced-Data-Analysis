# Projekt: Advanced Data Analysis (DLBDSEDA02)
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Inhaltsverzeichnis</summary>
  <ol>
    <li><a href="##aufgabe-11-nlp-techniken-anwenden-um-eine-textsammlung-zu-analyieren">Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlng zu analyieren</a></li>
    <li><a href="#phase1">Phase 1 - Konzeptionsphase</a>
      <ul>
      <li><a href="#datenaquisition-engl-data-acquisition">Datenaquisition (engl. data acquisition)</a></li>
        Datenrecherche
        Datensammlung
        Download
        Datenprüfung
        Datenauswahl
        <li><a href="#nlp-pipeline">NLP-Pipeline</a></li>
        Pre-Processing
        Processing
        Post-Processing
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

## Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlihen Beschwerden anzuwenden und so, die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden. 

1. Schriftliches Konzept
2. Datenverarbeitung
2.1. Vektorisierung 2 Techniken
2.2. Extraktion von Themen 2 Ansätze.

## Datenaquisition (engl. data acquisition)
Trichter

<!-- Datenaquisition -->
<details>
  <summary>Datenaquisition (engl. data acquisition)</summary>
  <ol>
    <li><a href="#datenrecherche">Datenrecherche (engl. data research)</a></li>
    Zunächst wird eine Onlinerecherche auf verschiedenen Plattformen (Kaggle, GitHub, ect. ) durchgeführt und nach geeigneten englischen oder deutschen Datensätzen gesucht. Offensichtlich synthetisch erzeugte Varianten () werden ignoriert.
    <li><a href="#datensammlung">Datensammlung (engl. data collection)</a></li>
    Datenformat: csv
      <li><a href="#download">Download (engl. data collection)</a></li>
      Datenquellen mit organisch vermutetem Ursprung werden lokal heruntergeladen und anschließend anhand eines KI-Detectors genauer geprüft.
      <li><a href="#datensatzprüfung">Datenprüfung (engl. data check)</a></li>
      Die Instanzen (engl. samples) der Datensätze werden auf synthetisch erzeugte Varianten hin geprüft und im Anschluss getaggt.anhand von Kriterien wie dem vermuteten Anteil realer oder synthetischer Samples sowie den nicht durch das Modell verarbeitbaren Samples (Real/Fake/Error) wird der Datensatz mit dem geringsten Real/Fake-Quotienten gewählt, da hier die Wahrscheinlichkeit eines organischen Ursprungs am höchsten erscheint. Der so identifizierte Datensatz geht in die NLP-Pipeline, welche mit der Textvorverarbeitung (engl. text pre-processing) des Datensatzes beginnt. Übersteigt die Anzahl der Samples eine Schwelle von 2000 wird der Datensatz zunächst aus Performancegründen gesplittet.
      <li><a href="#datensatzauswahl">Datensatzauswahl (engl. data selection)</a></li>
      <li><a href="#datenaufnahme">Datenaufnahme (engl. data ingestion)</a></li>


      <ul>
      <li><a href="#datenaquisition-engl-data-acquisition">Datenaquisition (engl. data acquisition)</a></li>
        Datenrecherche
        Datensammlung
        Download
        Datenprüfung
        Datenauswahl
        <li><a href="#nlp-pipeline">NLP-Pipeline</a></li>
        Pre-Processing
        Processing
        Post-Processing
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


## NLP-Verarbeitungsschritte (engl. Pipeline)
anhand von Kriterien wie
<!-- Pre-Processing -->
<details>
  <summary>Pre-Processing</summary>
  <ol>
      <ul>
<!-- Processing -->
<details>
  <summary>Processing</summary>
  <ol>


<!-- Post-Processing -->
<details>
  <summary>Post-Processing</summary>
  <ol>




### Pre-Processing
### Processing
### Post-Processing
