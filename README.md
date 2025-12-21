# Projekt: Advanced Data Analysis (DLBDSEDA02_D)

## Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlihen Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden. 

1. Schriftliches Konzept<br>
2. Datenverarbeitung<br>
2.1. Vektorisierung 2 Techniken<br>
2.2. Extraktion von Themen 2 Ansätze.<br>

### Projektstruktur
```markdown
├── datasets/  # Roh- und vorverarbeitete Texte
├── src/       # Python-Module
├── docs/      # Abbildungen, PDFs
└── README.md
```

<img src="https://github.com/IU-KaiW/DLBDSEDA02_Projekt_Advanced-Data-Analysis/blob/main/Projekt%20ADA%20v.3-Pipeline.jpg" width="500">

## Pipeline-Input
<b>⚪ Datenaquisition (engl. dataset acquisition)</b></br>
In der Phase der Datenaquisition werden Datensätze für den Input der NLP-Pipeline gesucht, bewertet und ausgewählt. Hierzu wird ein trichterförmiger, vier stufiger Prozess Datensatzrecherche, –sammlung, –prüfung sowie –auswahl durchlaufen, an dessen Ende die Eingabe (engl. input) in die Pipeline steht.</br>
<ol>
    <details>
      <summary>⚪ Datensatzrecherche (engl. dataset research)</b></summary>
      <i>Es wird eine Onlinerecherche auf verschiedenen Datenportalen (Kaggle, GitHub, GovData, ect.) durchgeführt und nach geeigneten deutschen und englischen Datensätzen gesucht.</i></br>
    </details>
    <details>
      <summary>⚪ Datensatzsammlung (engl. dataset collection)</summary>
      <i>Offensichtlich synthetisch erzeugte Datensätze werden ignoriert. Datenquellen mit vermutetem organischen Ursprung werden im CSV-Datenformat heruntergeladen und lokal gespeichert.</i></br>
    </details>
    <details>
      <summary>⚪ Datensatzprüfung (engl. dataset check)</summary>
      <i>Die gesammelten Datensätze werden anhand eines vorhandenen KI-Detectors auf synthetisch erzeugte Instanzen (engl. samples) geprüft und mit Labels (Real / Fake / Error) getaggt.</i></br>
    </details>
    <details>
      <summary>⚪ Datensatzauswahl (engl. dataset selection)</summary>
      <i>Anhand der Label wird der Datensatz mit dem höchsten Score an organisch identifizierten Instanzen gewählt, da hier die Wahrscheinlichkeit eines organischen Ursprungs am höchsten scheint.</i></br>
      <br>$$\% \text{organisch} = \frac{REAL}{REAL + FAKE + ERROR} \cdot 100$$</br>
      <br><i>Der so identifizierte Datensatz wird für weitere Verarbeitungsschritte genutzt. Übersteigt die Anzahl der Samples eine Schwelle von 2000 wird der Datensatz zunächst aus Performancegründen gesplittet.</i></br>
</ol>
    </details>
    
$$
(dataset_{organisch})=\left(\frac{REAL}{REAL + FAKE + ERROR}\right)\cdot100
$$
## NLP-Verarbeitungsschritte (engl. NLP-Pipeline)
anhand von Kriterien wie

Features of Texts are: 

### Pre-Processing
*Das pre-processing

<b>🔴 Merkmalsvorbereitung (engl. feature preparation)</b></br>
*Beginn der der Merkmalsvorbereitung (engl. feature preparation).*
<ol>
    <details>
      <summary>🔴 Textbereinigung (engl. text cleaning)</b></summary>
      <i>xxxxx</i>
        <ol>
        <li> Standardisierung (engl. standardisation)<br></li>
        <i>xxxxx</i>
        <li> Rauschentfernung (engl. noise reduction)<br></li>
        <i>xxxxx</i>
        </ol>
    </details>
    <details>
      <summary>🔴 Merkmalsextraktion (engl. feature extraction)</summary>
      <i>xxxxx</i>
        <ol>
        <li>Tokenisierung (engl. tokenization)</li>
        <i>In diesem Schritt</i>
        <li>Vokabularerstellung/Wortschatzaufbau (engl. Vocabulary Construction)</li>
        <i>In diesem Schritt</i>
        </ol>
    </details>
</ol>


### Processing

<b>🟠 Merkmalsaufbereitung (engl. feature engineering)</b></br>
*Beginn der Merkmalsaufbereitung (engl. feature engineering)*
<ol>
    <details>
      <summary>🟠 Merkmalsextraktion (engl. feature extraction)</summary>
      <i>xxxxx</i>
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

<b>🟡 Text Analyse (engl. Text Analytics)</b></br>
*Beginn der Textanalyse (engl. Text Analytics), bei der die Themenmodellierung beginnt, durch welche Themen aus dem Text extrahiert werden.*
<ol>
    <details>
      <summary><b>🟡 Merkmalsextraktion (engl. feature extraction)</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Beginn der Merkmalsextraktion zum Zweck der Themenmodellierung</i>
      <summary>Themenmodellierung (engl. topic modeling)</summary>
      <i>Beginn der Themenmodellierung</i>
           NMF, LDA
    </details>
    <details>
      <summary><b>🟡 Merkmalsumwandlung (engl. feature transformation)</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Beginn der Merkmalsumwandlung in der Merkmale in eine geeignete Form gebracht werden.</i>
          .<br>
      <summary>Themenmodellierung (engl. topic modeling)</summary>
      <i>Beginn der Themenmodellierung</i>
           LSA
    </details>
    <details>
      <summary><b>🟡 Merkmalsauswahl (engl. feature selection)</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Zum Ende werden die besten Merkmale ausgewählt.</i>
    </details>
</ol>    
<b>🔵 Merkmalslernen (engl. feature learning / representation learning)</b></br>
<i>Beginn der Modellbildung für Aufgabe<i>

### Pipeline Ausgabe (engl. Pipeline Output)
<b>🟢 Merkmalsanalyse (engl. feature-analysis)</b></br>
      <i>xxxxxxxxxxxx</i>
<ol>
    <details>
      <summary><b>🟢 Visualisierung</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grafische Darstellung</i>
    </details>
    <details>
      <summary><b>🟢 Aggregation</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numerische Darstellun</i>
    </details>
</ol>

### Post-Processing
*Beginn der Evaluation*
<ol>
  <details>
      <summary>🟣 Evaluation</summary>
      <ol>
        <li> Aggregation</li>
        <i>blub blub, blub</i>
        <li> Visualisierung</li>
        <i>blub blub, blub</i>
      </ol>
  </details>
</ol>

### Referenzen
Detector
`KI-Detektor`

Bibliotheken
`Gensim`

<li>1</li>
````python
12535
````

<li>2</li>

### Codeblocks
````python
33333
````


### Installation

Prerequisites
Installation


### 
📊 Ergebnisse
Themenverteilungen
Top-Wörter pro Thema
Modellvergleich