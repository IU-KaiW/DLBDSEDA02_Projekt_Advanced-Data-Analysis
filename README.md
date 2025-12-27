# Projekt: Advanced Data Analysis (DLBDSEDA02_D)

## Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlihen Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden. 

1. Schriftliches Konzept<br>
2. Datenverarbeitung<br>
2.1. Vektorisierung 2 Techniken<br>
2.2. Extraktion von Themen 2 Ansätze.<br>

### Projektstruktur
#### Ordnerstruktur
```markdown
├── datasets/  # Roh- und vorverarbeitete Texte
├── src/       # Python-Module
├── docs/      # Abbildungen, PDFs
└── README.md
```
### Installation
##### Einrichtung (engl. initial setup)
###### KI-Text-Detector
Für die Prüfung der recherchierten Datensätze kommt ein extern entwickelter AI-Text-Detector zum Einsatz. Zur Nutzbarmachung wurde die Installationsroutine des Entwicklers befolgt. 

###### Virtuelle Umgebung (engl. virtual environment)
Als virtuelle Umgebungen stehen in Python "conda" und "venv" zur Verfügung. Aufgrund der Bibiliothek SpaCy wurde sich für conda entschieden.
```console
`pip install conda`
```

###### Pipeline Eingabe
`pandas`       https://pandas.pydata.org<br>
`Gensim`       https://pypi.org/project/gensim/<br>
`transformers` https://pypi.org/project/transformers/<br>
`torch`        https://pytorch.org<br>

###### Pipeline Verarbeitung
`spacy`        https://spacy.io<br>
`nltk`         https://www.nltk.org<br>

###### Pipeline Ausgabe
`matplotlib`   https://matplotlib.org<br>
`seaborn`      https://seaborn.pydata.org<br>

```python
import seaborn as sns
import matplotlib as mpl
import pandas as pd“ 
import numpy as np“ 
```

`Cartopy`      https://cartopy.readthedocs.io/stable/getting_started/index.html<br>
`re`         

<img src="https://github.com/IU-KaiW/DLBDSEDA02_Projekt_Advanced-Data-Analysis/blob/main/Visualisierung.jpg" width="1200">

## ⚪ Pipeline Eingabe (engl. Pipeline-Input)
<b> Datenaquisition (engl. dataset acquisition)</b></br>
In der Phase der Datenaquisition werden Datensätze für den Input der NLP-Pipeline gesucht, bewertet und ausgewählt. Hierzu wird ein trichterförmiger, vier stufiger Prozess Datensatzrecherche, –sammlung, –prüfung sowie –auswahl durchlaufen, an dessen Ende die Eingabe (engl. input) in die Pipeline steht.</br>
<ol>
    <details>
      <summary>⚪ <u> Datensatzrecherche (engl. dataset research) <u></b></summary>
      <i>Es wird eine Onlinerecherche auf verschiedenen Datenportalen (Kaggle, GitHub, GovData, ect.) durchgeführt und nach geeigneten deutschen und englischen Datensätzen gesucht.</i></br>
    </details>
    <details>
      <summary>⚪ Datensatzsammlung (engl. dataset collection)</summary>
      <i>Offensichtlich synthetisch erzeugte Datensätze werden ignoriert. Datenquellen mit vermutetem organischen Ursprung werden im CSV-Datenformat heruntergeladen und lokal gespeichert.</i></br>
    </details>
    <details>
      <summary>⚪ Datensatzprüfung (engl. dataset check)</summary>
      <i>Die gesammelten Datensätze werden anhand eines KI-Text-Detectors auf synthetisch erzeugte Instanzen (engl. samples) geprüft und mit Labels (REAL / FAKE / ERROR) getaggt. Dazu muss die Spaltenbeschriftung der textführende Spalte in "text" umgenannt werden. </i><br>
    </details>
    <details>
      <summary>⚪ Datensatzauswahl (engl. dataset selection)</summary>
      <i>Durch eine Häufigkeitsauswertung der Label wird der Datensatz mit dem prozentual höchsten Anteil an organischen (REAL-Label) Instanzen die die Formel:</i><br>
      <br>
      $$\%\text{ organisch} = \left(\frac{REAL}{REAL + FAKE + ERROR}\right) \cdot100$$
      <br>
      <br><i>ausgewertet. Die Wahrscheinlichkeit eines organischen Ursprungs erscheint höher, je höher der Prozentsatz organisch identifizierter Instanzen im Verhältnis zum Gesamtdatensatz ist. Kann ein Datensatz nicht in angemessener Zeit (30 min.) durch das Modell verarbeitet werden, wird die Prüfung abgebrochen und die Bewertung als n/a markiert. Der Datensatz fließt dann nicht in den Ergebnisvergleich ein.</i>
    </details>
</ol>

```markdown
| Nr. | Datensatz                          | Bewertung | Größe     | Quelle         | Link                                                                                 |
|-----|------------------------------------|-----------|-----------|----------------|--------------------------------------------------------------------------------------|
| 1   | Consumer_Complaints.csv            | n/a       | 59,40  MB | Kaggle         | https://www.kaggle.com/datasets/ashwinik/consumer-complaints-financial-products      |
| 2   | rows.csv                           | n/a       | 176    MB | Kaggle         | https://www.kaggle.com/datasets/selener/consumer-complaint-database                  |
| 3   | Consumer_Complaints.csv            | 13,5 %    | 107,0  MB | Kaggle/GovData | https://www.kaggle.com/code/saurabhsawhney/nlp-complaints-classification             |
| 4   | complaints_processed.csv           | 64,72 %   | 19,8   MB | Kaggle         | https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp       |
| 5   | Comcast.csv                        | 82 %      | 60,0   kB | Kaggle         | https://www.kaggle.com/datasets/yasserh/comcast-telecom-complaints                   |
| 7   | user_complaints                    | 0,69 %    | 229,0  kB | GitHub         | https://github.com/gurneetjuneja/NLP-Problem-Solving/blob/main/user_complaints.csv   |
| 8   | consumer_complaints.csv            | n/a       | 175,39 MB | Kaggle         | https://www.kaggle.com/code/mchirico/analyzing-text-in-consumer-complaints           |
| 9   | Complaints_Reports_Data.sql        | n/a       | 3,28   MB | MendeleyData   | https://data.mendeley.com/datasets/w2cp7h53s5/1                                      |
| 10  | chatgpt_reviews.csv                | 35,03 %   | 119,9  MB | GitHub         | https://github.com/Schossi2908/DLBDSEDA02_D                                          |
| 11  | dataset-tickets-multi-lang3-4k.csv | n/a       | 6,87   MB | Kaggle         | https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets    |
```
Der Datensatz mit der prozentualen höchsten Bewertung wird als Korpus für die nachfolgenden Aufgabe genutzt.

## NLP-Verarbeitungsschritte (engl. NLP-Pipeline)
Übersteigt die Anzahl der Instanzen die Schwelle von 2000, wird der Datensatz für die folgenden Verarbeitungschritte auf diese Anzahl begrenzt.
Merkmale (engl. features) eines Textes oder Dokuments sind Informationen wie Länge (engl. length), Quelle (engl. source) und Datum der Veröffentlichungsdatum (engl. date of publication). 

### 🔴 Vorverarbeitung (engl. pre-processing)
Das pre-processing

<b> Merkmalsvorbereitung (engl. feature preparation)</b></br>
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


### 🟠 Verarbeitung (engl. processing)

<b> Merkmalsaufbereitung (engl. feature engineering)</b></br>
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

<b>🟠 Text Analyse (engl. Text Analytics)</b></br>
*Beginn der Textanalyse (engl. Text Analytics), bei der die Themenmodellierung beginnt, durch welche Themen aus dem Text extrahiert werden.*
<ol>
    <details>
      <summary><b>🟠 Merkmalsextraktion (engl. feature extraction)</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Beginn der Merkmalsextraktion zum Zweck der Themenmodellierung</i>
      <summary>Themenmodellierung (engl. topic modeling)</summary>
      <i>Beginn der Themenmodellierung</i>
           NMF, LDA
    </details>
    <details>
      <summary><b>🟠 Merkmalsumwandlung (engl. feature transformation)</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Beginn der Merkmalsumwandlung in der Merkmale in eine geeignete Form gebracht werden.</i>
          .<br>
      <summary>Themenmodellierung (engl. topic modeling)</summary>
      <i>Beginn der Themenmodellierung</i>
           LSA
    </details>
    <details>
      <summary><b>🟠 Merkmalsauswahl (engl. feature selection)</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Zum Ende werden die besten Merkmale ausgewählt.</i>
    </details>
</ol>    
<b>🔵 Merkmalslernen (engl. feature learning / representation learning)</b></br>
<i>Beginn der Modellbildung für Aufgabe<i>

## 🟢 Pipeline Ausgabe (engl. Pipeline Output)
<b>🟢 Merkmalsanalyse (engl. feature-analysis)</b></br>
      <i>xxxxxxxxxxxx</i>
<ol>
    <details>
      <summary><b>🟢 Visualisierung</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grafische Darstellung</i>
    </details>
    Alpha - 
    Beta - Wortverteilung in Themen
    K^T (n × k)
    <details>
      <summary><b>🟢 Aggregation</b></summary>
      <i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numerische Darstellun</i>
    </details>
</ol>

### Nachverarbeitung (engl. post-processing)
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

### Codeblocks
````python
33333
````

Installation


### 
📊 Ergebnisse<br>
Themenverteilungen<br>
Top-Wörter pro Thema<br>
Modellvergleich<br>


### Referenzen
<ul>
<li>Software</li>

###### KI-Text-Detector 
Kishan Jai Soorya N. AI-Text-Detector-Python. (https://github.com/Kishanjaisoorya/AI-Text-Detector-python)

<li>Bibliotheken</li>


###### initial setup
`pandas` https://pandas.pydata.org<br>

###### Vorprüfung
`transformers`https://pypi.org/project/transformers/<br>


###### NLP-Pipeline
`Gensim` https://pypi.org/project/gensim/<br>
`SpaCy` https://spacy.io<br>
`NLTK`https://www.nltk.org<br>


###### Nachverarbeitung
`matplotlib` https://matplotlib.org<br>
`seaborn` https://seaborn.pydata.org<br>

<li>Literatur</li>

###### Skripte
IU Internationale Hochschule. (2024). Advanced Data Analysis (DLBDSEDA01_D) [Lernskript]. 001-2024-1210.<br>

###### Wissenschaftliche Artikel
Blei, David M. and Ng, Andrew Y. and Jordan, Michael I.: Latent dirichlet allocation. In: The Journal of Machine Learning Research. Nr. 3, 3. Januar 2003, S. 993–1022<br>

<li>Websites</li>

###### GeekforGeek.org
Sanchhaya Education Private Ltd., 2025 (https://www.geeksforgeeks.org/nlp/nlp-gensim-tutorial-complete-guide-for-beginners/)

###### Dokumentationen
https://spacy.io/usage/spacy-101

- <br>
- <br>
</ul>