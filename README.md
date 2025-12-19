# Projekt: Advanced Data Analysis (DLBDSEDA02_D)

## Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlihen Beschwerden anzuwenden und so, die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden. 

1. Schriftliches Konzept<br>
2. Datenverarbeitung<br>
2.1. Vektorisierung 2 Techniken<br>
2.2. Extraktion von Themen 2 Ansätze.<br>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Inhaltsverzeichnis</summary>
  <ol>
    <li><a href="##aufgabe-11-nlp-techniken-anwenden-um-eine-textsammlung-zu-analyieren">Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlng zu analyieren</a></li>
    <li><a href="#phase1">Phase 1 - Konzeptionsphase</a>
      <ul>
      <li><a href="#datenaquisition-engl-data-acquisition">Datenaquisition (engl. data acquisition)</a></li>
        Datenrecherche<br>
        Datensammlung<br>
        Download<br>
        Datenprüfung<br>
        Datenauswahl<br>
        <li><a href="#nlp-pipeline">NLP-Pipeline</a></li>
        Pre-Processing<br>
        Processing<br>
        Post-Processing<br>
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

## Datenaquisition (engl. data acquisition)
*In der Phase der Datenaquisition werden Datensätze für den Input der NLP-Pipeline gesucht, bewertet und ausgewählt. Hierzu wird ein trichterförmiger, vier stufiger Prozess Datensatz-Recherche, –Sammlung, –Prüfung sowie –Auswahl durchlaufen, an dessen Ende die Pipeline-Eingabe steht.*

<details>
  <summary>Datenaquisition (engl. data acquisition)</summary>
  <ol>
    <li><a href="#datenrecherche">Datenrecherche (engl. data research)</a></li>
    Zunächst wird eine Onlinerecherche auf verschiedenen Plattformen und Datenportalen (Kaggle, GitHub, GovData, ect.) durchgeführt und nach geeigneten  deutschen oder englischen Datensätzen gesucht. Offensichtlich synthetisch erzeugte Varianten wurden hierbei ignoriert.
    <li><a href="#datensammlung">Datensammlung (engl. data collection)</a></li>
    Datenformat: csv
    Datenquellen mit organisch vermutetem Ursprung werden lokal heruntergeladen und anschließend anhand eines KI-Detectors genauer geprüft.
      <li><a href="#datensatzprüfung">Datenprüfung (engl. data check)</a></li>
      Die Instanzen (engl. samples) der Datensätze werden auf synthetisch erzeugte Varianten hin geprüft und im Anschluss getaggt.anhand von Kriterien wie dem vermuteten Anteil realer oder synthetischer Samples sowie den nicht durch das Modell verarbeitbaren Samples (Real/Fake/Error) wird der Datensatz mit dem geringsten Real/Fake-Quotienten gewählt, da hier die Wahrscheinlichkeit eines organischen Ursprungs am höchsten erscheint. Der so identifizierte Datensatz geht in die NLP-Pipeline, welche mit der Textvorverarbeitung (engl. text pre-processing) des Datensatzes beginnt. Übersteigt die Anzahl der Samples eine Schwelle von 2000 wird der Datensatz zunächst aus Performancegründen gesplittet.
      <li><a href="#datensatzauswahl">Datensatzauswahl (engl. data selection)</a></li>
  </ol>
</details>

Pipeline-Input

## NLP-Verarbeitungsschritte (engl. NLP-Pipeline)
anhand von Kriterien wie

### Pre-Processing
*Beginn der der Merkmalsvorbereitung (engl. feature preparation).*
<ol>
    <details>
      <summary>Textbereinigung (engl. text cleaning)</summary>
        <ol>
        <li> Standardisierung (engl. standardisation)<br></li>
        In diesem Schritt wird
        <li> Rauschentfernung (engl. noise reduction)<br></li>
        In Diesem Schritt werden
        </ol>
    </details>
    <details>
      <summary>Merkmalsextraktion (engl. feature extraction)</summary>
        <ol>
        <li> Tokenisierung (engl. tokenization)</li>
         In diesem Schritt
        <li> Vokabularerstellung/Wortschatzaufbau (engl. Vocabulary Construction)</li>
         In diesem Schritt
        </ol>
    </details>
</ol>


### Processing
*Beginn der Merkmalsaufbereitung (engl. feature engineering)*
<ol>
    <details>
      <summary>Merkmalsextraktion (engl. feature extraction)</summary>
        <ol>
        <li>Vektorisierung</li>
        In diesem Schritt
       <ul>
        <li><a prerequisites>Frequency Based Embedding</a></li>
        TF-IDF
        <li><a installation>Prediction Based Word Embedding</a></li>
        GloVE; Word2Vec
        <li><a installation>Contextualized Based Word Embedding</a></li>
        GloVE; Word2Vec
        </ul>
        </ol>
    </details>
    <details>
      <summary>Merkmals ... (engl. feature transformation)</summary>
        <ol>
        <li> 1<br></li>
        In diesem Schritt wird
        <li> 2<br></li>
        In Diesem Schritt werden
        </ol>
    </details>
    <details>
      <summary>Merkmals ... (engl. feature selection)</summary>
        <ol>
        <li> 1<br></li>
        In diesem Schritt wird
        <li> 2<br></li>
        In Diesem Schritt werden
        </ol>
    </details>    
</ol>
### Post-Processing
*Beginn der Evaluation*
<ol>
  <details>
      <summary>Post-Processing</summary>
      <ol>
        <li>Aggregation</li>
        <li>Visualisierung</li>
      </ol>
  </details>
</ol>