# Projekt: Advanced Data Analysis (DLBDSEDA02_D)

## Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlihen Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden. 

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

Features of Texts are: 

### Pre-Processing

<b>🟢 Merkmalsvorbereitung (engl. feature preparation)</b></br>
*Beginn der der Merkmalsvorbereitung (engl. feature preparation).*
<ol>
    <details>
      <summary> Textbereinigung (engl. text cleaning)</b></summary>
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

<b>🟢 Merkmalsaufbereitung (engl. feature engineering)</b></br>
*Beginn der Merkmalsaufbereitung (engl. feature engineering)*
<ol>
    <details>
      <summary>Merkmalsextraktion (engl. feature extraction)</summary>
        <ol>
        <li>Vektorisierung</li>
        <i>In diesem Schritt</i>
       <ul>
          <li><a prerequisites>Frequency Based Embedding</a></li>
           TF-IDF
          <li><a installation>Prediction Based Word Embedding</a></li>
           GloVE; Word2Vec
          <li><a installation>Contextualized Based Word Embedding</a></li>
    </details>
</ol>


#### Text Analyse (engl. Text Analytics)
*Beginn der Textanalyse (engl. Text Analytics), bei der die Themenmodellierung beginnt, durch welche Themen aus dem Text extrahiert werden.*
<ol>
    <details>
      <summary><b>🟢 Merkmalsextraktion (engl. feature extraction)</b></summary>
      <i>Beginn der Merkmalsextraktion zum Zweck der Themenmodellierung</i>
      <summary>Themenmodellierung (engl. topic modeling)</summary>
      <i>Beginn der Themenmodellierung</i>
           NMF, LDA
    </details>
    <details>
      <summary><b>🟢 Merkmalsumwandlung (engl. feature transformation)</b></summary>
      <i>Beginn der Merkmalsumwandlung in der Merkmale in eine geeignete Form gebracht werden.</i>
          .<br>
      <summary>Themenmodellierung (engl. topic modeling)</summary>
      <i>Beginn der Themenmodellierung</i>
           LSA
    </details>
    <details>
      <summary><b>🟢 Merkmalsauswahl (engl. feature selection)</b></summary>
      <i>Zum Ende werden die besten Merkmale ausgewählt.</i>
    </details>
</ol>    
<b>🟢 Merkmalslernen (engl. feature learning / representation learning)</b></br>
<i>Beginn der Modellbildung für Aufgabe<i>

### Pipeline Ausgabe (engl. Pipeline Output)
<b>🟢 Merkmalsanalyse (engl. feature-analysis)</b></br>
      <i>xxxxxxxxxxxx</i>
<ol>
    <details>
      <summary><b>🟢 Visualisierung</b></summary>
      <i>grafische Darstellung</i>
    </details>
    <details>
      <summary><b>🟢 Aggregation</b></summary>
      <i>numerische Darstellun</i>
    </details>
</ol>

### Post-Processing
*Beginn der Evaluation*
<ol>
  <details>
      <summary>Evaluation</summary>
      <ol>
        <li>Aggregation</li>
        <i>blub blub, blub</i>
        <li>Visualisierung</li>
        <i>blub blub, blub</i>
      </ol>
  </details>
</ol>

### Referenzen
`asdasdasdasdasd`

<li>1</li>
````python
12535
````

<li>2</li>

### Codeblocks
````python
33333
````