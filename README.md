# Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlichen Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden.<br>

Das zu erstellende schriftliche Konzept soll die Schritte der NLP-Datenverarbeitung mit Python darlegen. Dabei sollen kurz zwei Techniken zur Vektorisierung sowie zwei Ansätze zur Extraktion von Themen aus dem Datensatz genannt und die zu verwendenten Python-Bibliotheken erwähnt werden.

## Konzeption
Der Grafik können die geplanten Phasen des Projekts sowie die zugeordneten Prozesse entnommen werden. 

<img src="https://github.com/IU-KaiW/DLBDSEDA02_Projekt_Advanced-Data-Analysis/blob/main/docs/Visualisierung.jpg" width="1200">

      
### ⚪ Datenakquisition (engl. data acquisition)
> In der Phase der Datenaquisition werden Datensätze für den Input der NLP-Pipeline gesucht, bewertet und ausgewählt. Hierzu wird ein trichterförmiger, vier stufiger Prozess Datensatzrecherche, –sammlung, –prüfung sowie –auswahl durchlaufen, an dessen Ende die Eingabe (engl. input) in die Pipeline steht.</br>
<ol>
    <details>
      <summary>⚪ <u> Datensatzrecherche (engl. dataset research) <u></b></summary>
      <i>Es wird eine Onlinerecherche auf verschiedenen Datenportalen (Kaggle, GitHub, GovData, MendeleyData, u.A.) durchgeführt und nach geeigneten deutschen und englischen Datensätzen gesucht.</i></br>
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
Der Datensatz mit der prozentualen höchsten Bewertung wird als Korpus bzw. Pipeline Eingabe (engl. Pipeline-Input) für die nachfolgenden Schritte genutzt. Übersteigt die Anzahl der Instanzen die Schwelle von 2000, wird der Datensatz für die folgenden Verarbeitungschritte die Anzahl begrenzt.<br>

```markdown
| Nr. | Datensatz                          | Bewertung | Größe     | Quelle         | Link                                                                                 |
|-----|------------------------------------|-----------|-----------|----------------|--------------------------------------------------------------------------------------|
| 01  | Consumer_Complaints.csv            | n/a       | 59,40  MB | Kaggle         | https://www.kaggle.com/datasets/ashwinik/consumer-complaints-financial-products      |
| 02  | rows.csv                           | n/a       | 176    MB | Kaggle         | https://www.kaggle.com/datasets/selener/consumer-complaint-database                  |
| 03  | Consumer_Complaints.csv            | 13,5 %    | 107,0  MB | Kaggle/GovData | https://www.kaggle.com/code/saurabhsawhney/nlp-complaints-classification             |
| 04  | complaints_processed.csv           | 64,72 %   | 19,8   MB | Kaggle         | https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp       |
| 05  | Comcast.csv                        | 82 %      | 60,0   kB | Kaggle         | https://www.kaggle.com/datasets/yasserh/comcast-telecom-complaints                   |
| 07  | user_complaints                    | 0,69 %    | 229,0  kB | GitHub         | https://github.com/gurneetjuneja/NLP-Problem-Solving/blob/main/user_complaints.csv   |
| 08  | consumer_complaints.csv            | n/a       | 175,39 MB | Kaggle         | https://www.kaggle.com/code/mchirico/analyzing-text-in-consumer-complaints           |
| 09  | Complaints_Reports_Data.sql        | n/a       | 3,28   MB | MendeleyData   | https://data.mendeley.com/datasets/w2cp7h53s5/1                                      |
| 10  | chatgpt_reviews.csv                | 35,03 %   | 119,9  MB | GitHub         | https://github.com/Schossi2908/DLBDSEDA02_D                                          |
| 11  | dataset-tickets-multi-lang3-4k.csv | n/a       | 6,87   MB | Kaggle         | https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets    |
```
[^01]. [D01] (https://www.kaggle.com/datasets/ashwinik/consumer-complaints-financial-products)<br>
[^02] [D02](https://www.kaggle.com/datasets/selener/consumer-complaint-database)<br>
[^03] [D03](https://www.kaggle.com/code/saurabhsawhney/nlp-complaints-classification)<br>
[^04]. [D04](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)<br>
[^05]. [D05](https://www.kaggle.com/datasets/yasserh/comcast-telecom-complaints)<br>
[^06]. [D06](https://github.com/gurneetjuneja/NLP-Problem-Solving/blob/main/user_complaints.csv)<br>
[^07]. [D07](https://github.com/gurneetjuneja/NLP-Problem-Solving/blob/main/user_complaints.csv)<br>
[^08]. [D08](https://www.kaggle.com/code/mchirico/analyzing-text-in-consumer-complaints)<br>
[^09]. [D09](https://data.mendeley.com/datasets/w2cp7h53s5/1)<br>
[^10]. [D10](https://github.com/Schossi2908/DLBDSEDA02_D)<br>
[^11]. [D11](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)<br>

[^01]: [Kaggle] https://www.kaggle.com/datasets/ashwinik/
[^02]: [Kaggle] https://www.kaggle.com/datasets/selener/
[^03]: [GitHub] www.github.com
[^04]: [GovData] GovData
[^05]: [MendeleyData] data.mendeley.com
[^06]: [Huggingface] Huggingface
[^07]: []
[^08]:
[^09]:
[^10]:      
[^11]:     
______________

###### Pipeline-Eingabe
Es wird Datensatz Nr. 5 "Comcast.csv" mit der Bewertung von 82 % gewählt und als Input für die NLP-Pipeline genutzt.

### ⚫ Sprachverarbeitung (engl. NLP-Pipeline)
Merkmale (engl. features) eines Textes oder Dokuments sind Informationen wie Länge (engl. length), Quelle (engl. source) und Datum der Veröffentlichungsdatum (engl. date of publication)

Haupt Freatures / Sekundärfeatuers?

#### 🔴 Datenvorverarbeiten (engl. data pre-processing)
> Durch die Datenvorverarbeiten erfolgt eine *Merkmalsvorbereitung (engl. feature preparation)* für nachfolgende Phasen in einem mehrstufigen Prozess, welcher sich grob die Prozesse Textbereinigung und Merkmalsextraktion einteilen lässt.
<ol type="1">
  <details>
    <summary>🔴 Textbereinigung (engl. text cleaning)</summary>
    <p><i>Im Rahmen der Textbereinigung werden Texte zunächst standardisiert und anschließend von Rauschen befreit.</i></p>
    <ol type="1">
      <li>
        <ins>Standardisierung (engl. standardisation)</ins><br>
        <i>Im Rahmen der Textbereinigung werden Texte zunächst standadisiert, um inhaltlich relevanten Tokens zu vereinheitlichen. Hierdurch wird vermieden, dass gleiche Inhalte nicht in mehreren, leicht unterschiedlichen Varianten auftreten. </i>
      <ol type="2">
            <li>**Normalisierung (engl. normalisation)**</li>
            Durch die Normalisierung wird Text .... Sie setzt sich zusammen aus: 
            <ul>
              <li>Kasusumwandlung (engl. case conversion)</li>
              In diesem Schritt erfolgt eine konsequente Kleinschreibung (engl. lowercasing) aller Wörter.
              <li>Grundformreduktion (engl. inflection reduction)</li>
              Durch diesen Schritt werden Wörter auf ihre Grundformen reduziert. Da Lemmatisierung (engl. lemmatization) genauer als Stammformreduktion (engl. stemming) ist, wird diese eingesetzt. (WordNet lemmatizer from NLTK)
              <li>Formatnormalisierungen (engl. format normalisations)</li>
            </ul>
            <li>Rechtschreibfehlerkorrektur (engl. spelling correction)</li>
            Durch Rechtschreibkorrekrur werden Tipp bzw. Schreibfehler korrigiert.
      </ol>
      </li>
      <li>
        Rauschentfernung (engl. noise reduction)<br>
        <i>Ziel der Rauschentfernung ist es, für die Verarbeitung irrelevante Tokens durch eine Zeichenentfernung zu löschen.</i>
            <ol type="2">
            <li>Stoppworte (engl. stopwords)</li>
            <li>Satzzeichen (engl. punctuation marks)</li>
            <li>Leerzeichen (engl. white space)</li>
            <li>Nummern (engl. removing numbers)</li>
            <li>Sonderzeichen (engl. special character)</li>
      </ol>
    </ol>
  </details>

  <details>
    <summary>🔴 Merkmalsextraktion (engl. feature extraction)</summary>
    <p><i>Durch Merkmalsextraktion wird Text im Rahmen der Merkmalsvorbereitung zur weiteren Verarbeitung präpariert.</i></p>
    <ol type="1">
      <li>
        Tokenisierung (engl. tokenization)<br>
        <i>Durch Tokenisierung wird der vorbereitet Text in Einzeltoken oder Phrasen (n-Gramme) zerlegt.</i>
        SpaCy
      </li>
      <li>
        Vokabularerstellung/Wortschatzaufbau (engl. Vocabulary Construction)<br>
        <i>Im Schritt des Wortschatzaufbaus (Vocabulary Construction) wird aus dem tokenisierten Textcorpus ein endliches Vokabular erstellt, das alle einzigartigen Tokens enthält und als Basis für nachfolgende Modelle dient.</i>
      </li>
    </ol>
  </details>
</ol>

#### 🟤 Datenverarbeitung (engl. data processing)
> Datenverarbeitung erfolgt durch Merkmalsaufbereitung (engl. feature engineering) und Modellbildung (engl. modeling). Bei Merkmalsaufbereitung werden Rohdaten in Merkmale (engl. features) umgewandelt werden, welche ein Modell im Anschluss nutzen kann. Bei Modellbildung werden

Merkmalsaufbereitung kann in Merkmalsextraktion, Merkmalsumwandlung, Merkmalskonstruktion und Merkmalsauswahl unterteilt werden. Im Merkmalslernen erfolgt die Bildung von Modellen ohne Feature Engineering, basierend auf ...
<ol>
  <details>
    <summary>🟤 Vektorisierung (engl. vectorization)</summary>
    <p><i>Vektorisierungstechniken wandeln Text in Embeddings, nummerische Repräsentationen um. Hierbei wird zwischen unsemantischen Embeddings (feature vektors) und semantischen Varianten (word vectors / sentence vectors) differenziert. Vektorisierungstechniken nutzen Merkmalsextraktion, um Text im Rahmen der Merkmalsaufbereitung für nachfolgende Schritte vorzubereiten.</i></p>
    <ol type="1">
       <li>unsemantische Embeddings</li>
       Hierunter werden 
          <ul>
            <li>Frequency Based Embedding</li>
           BoW, TF-IDF
          </ul>
       <li>semantische Embeddings</li>
          <ul>
            <li>Prediction Based Word Embedding</li>
            GloVE; Word2Vec, FastText
            <li>Contextualized Based Word Embedding</li>
            ELMO, BERT, GPT
          </ul>
  </details>
  <details>
    <summary>🟤 Text Analyse (engl. Text Analytics)</summary>
    <p><i>Beginn der Textanalyse (engl. Text Analytics), in welcher Merkmalsmodellierung (engl. feature modeling) und die Merkmalserkennung (engl. feature recognition) zu verorten sind.</i></p>
    <ol type="1">
       <li>🟤 Merkmalsmodellierung (engl. feature modeling)</li>
       Unter Merkmalsmodellierung versteht man die inhaltliche Strukturierung und Deutung des zuvor vektorisierten Texts. Sie legt fest, wie Merkmale thematisch oder semantisch für die Textanalyse genutzt werden können. Über Themenmodellierung (engl. topic modeling) können Themen unüberwacht (engl. unsuperviced) mittels Merkmalsextraktion oder Merkmalsumwandlung identifiziert werden.<br>
            <ol type="2">
            <li>🟤 Merkmalsextraktion (engl. feature extraction)</li><br>
            <i>Latent Dirichlet Allocation (LDA)</i>
            Identifiziert latente Themen in einer Sammlung von Dokumenten und stellt Dokumente basierend auf ihren Verteilungen über diese Themen dar.<br>
           <br><li>🟤 Merkmalsumwandlung (engl. feature transformation)<br></i></li>
            <i>Latent Semantic Analysis (LSA)</i><br>
            <li>🟤 Merkmalsauswahl (engl. feature selection)<br>
            <i>Zum Ende werden die besten Merkmale ausgewählt.</i>
          </ol>
       <li>🟤 Merkmalserkennung (engl. feature recognition)</li>
          <ol>
            <li>1</li>
            xxx
            <li>2</li>
            xxx
          </ol>
  </details>
  <details>
    <summary>🟤  Merkmalslernen (engl. feature learning / representation learning)</summary></br>
    In der Datenverarbeitung beginnt die Merkmalsaufbereitung (engl. feature engineering) und das Merkmalslernen (engl. feature learning / representation learning).
    <i>Beginn der Modellbildung für Aufgabe<i>
  </details>
</ol>

###### Pipeline Ausgabe (engl. Pipeline Output)
______________

### 🔵 Datenkonsolidierung (engl. data consolidation)
> Im Rahmen der Datenkonsolidierung erfolgt die Datennachverarbeitung (engl. data post-processing) in der die Merkmalsanalysen (engl. feature analysis) sowie die Modellevaluationen (engl. model evaluations) durchgeführt und letztlich als Datenpräsentation (engl. data presentation) aufbereitet werden.
<ol> 
  <details>
    <summary>⚪ Datenauswertung (engl. data analysis)</summary>
    <p><i>xxxxxxx</i></p>
    <ol type="1">
      <li>Merkmalsauswertungen (engl. feature Inspections)</li>
      <p><i>Themenverteilungen; Top-Wörter pro Thema</i></p>
        𝛼 (Alpha) - Themenmischung pro Dokument<br>
        β (Beta) - Wortverteilung in Themen<br>
        K<sup>T</sup> (n × k)
    <summary> Evaluation (engl. model evaluation)</summary>
    <i>Modellvergleich</i><br>
  </details>
  <details>
    <summary>⚪ Datenpräsentation (engl. data presentation)<br></summary>
    <p><i>xxxxxxx</i></p>
    <ol type="1">
      <li>Aggregation</li>
      Aggregation reduziert die Datenmenge durch mathematische Operationen wie Summe, Mittelwert, Zählung oder Maximum über Gruppierungen (z. B. nach Token-Typ, Dokument oder Zeitraum). In NLP könnte dies die Häufigkeitsverteilung von n-Grammen pro Domäne oder die durchschnittliche Embedding-Distanz pro Klasse bedeuten. Sie erfolgt vor der Visualisierung, um Überladung zu vermeiden, und ist rein datenverarbeitend ohne grafische Elemente.<br>
      <li>Visualisierung</li>
      <i>Aggregation fasst Rohdaten zu kompakteren Zusammenfassungen zusammen, während Visualisierung diese aggregierten Daten grafisch darstellt, um Muster erkennbar zu machen.</i>
      <i>grafische Darstellung</i><br>
      PyLDAvis
      <i>numerische Darstellung</i>
      <i>Modellperformance<i>
  </details>
</ol>



______________

## Projektstruktur
### Ordnerstruktur
```markdown
├── datasets/  # Roh- und vorverarbeitete Texte
├── src/       # Python-Module
├── docs/      # Abbildungen, PDFs
└── README.md
```

## Installation (engl. setup)
### Vorbereitende Installation (engl. preparatory setup)
#### Datenvalidierung (engl. data validation)
KI Detektor

#### Laufzeitumgebung (engl. runtime environment)
Als virtuelle Umgebungen stehen in Python "conda" und "venv" zur Verfügung. Aufgrund der Bibiliothek SpaCy wurde sich für conda entschieden.
```console
`pip install conda`
```
### Erstinstallation (engl. initial setup)

#### Voraussetzungen (engl. prerequisite setup)
##### Abhängigkeiten (engl. dependencies)

#### Bibliotheken (engl. librarys)
##### Sprachverarbeitung (engl. NLP-Pipeline)
##### Datenkonsolidierung (engl. data consolidation)
###### Visualisierung
###### Aggregation



```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
```

`Cartopy`      https://cartopy.readthedocs.io/stable/getting_started/index.html<br>
`re`   
`csv`   
`matplotlib`   https://matplotlib.org<br>
`seaborn`      https://seaborn.pydata.org<br>
`spacy`        https://spacy.io<br>
`nltk`         https://www.nltk.org<br>
`pandas`
`Gensim`
`transformers`
`torch`        

### Referenzen
<ul>
<li>Datenportale</li>

###### Kaggle
###### GovData
###### MendeleyData
###### Huggingface
###### Github

<li>Externe Software</li>

###### Detektor
Kishan Jai Soorya N. AI-Text-Detector-Python. (https://github.com/Kishanjaisoorya/AI-Text-Detector-python)

<li>Bibliotheken (engl. librarys)</li>


###### Einrichtung (engl. initial setup)
`pandas` https://pandas.pydata.org<br>

###### Vorprüfung
`transformers`https://pypi.org/project/transformers/<br>


###### NLP-Pipeline
`Gensim` https://pypi.org/project/gensim/<br>
Dokumentation: 

`SpaCy` https://spacy.io<br>
Dokumentation: 

`NLTK`https://www.nltk.org<br>
Dokumentation: 


###### Nachverarbeitung
Visualisierung
`matplotlib` https://matplotlib.org<br>
Dokumentation: 

`seaborn` https://seaborn.pydata.org<br>
Dokumentation: 

`PyLDAvis`     https://pypi.org/project/pyLDAvis/
Dokumentation: 

Evaluation


<li>Literatur</li>

###### Skripte
IU Internationale Hochschule. (2024). Advanced Data Analysis (DLBDSEDA01_D) [Lernskript]. 001-2024-1210.<br>

###### Wissenschaftliche Artikel
Blei, David M. and Ng, Andrew Y. and Jordan, Michael I.: Latent dirichlet allocation. In: The Journal of Machine Learning Research. Nr. 3, 3. Januar 2003, S. 993–1022<br>
Blum et al., 2020 ?

Abbott, D., Kommer, I., & Kommer, C. (2025). Datenvisualisierung im praktischen Einsatz: Ansprechende Diagramme und Dashboards gestalte (1. Auflage). dpunkt.verlag.


<li>Websites</li>

###### GeekforGeek.org
Sanchhaya Education Private Ltd., 2025 (https://www.geeksforgeeks.org/nlp/nlp-gensim-tutorial-complete-guide-for-beginners/)
StudySmarter GmbH, 2025 (https://www.studysmarter.de/schule/informatik/computerlinguistik-theorie/feature-selection/)
Radim Řehůřek, 10.08.2024 - (https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py)
Christopher Helm, 08.05.2025 - (https://konfuzio.com/de/spacy-vs-nltk/)



https://www.kaggle.com<br>
https://www.github.com<br>
https://data.mendeley.com<br>

###### Dokumentationen
SpaCy -  https://spacy.io/usage/spacy-101<br>
GitHub - https://docs.github.com/de/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax<br>

</ul>

🟣 🟢🟠🔴
🔵 🟡🟤⚫
https://emojiterra.com/de/gelber-kreis/




<ol>
  <details>
    <summary>🟤 Merkmalsextraktion (engl. feature extraction)</summary>
    <p><i>Durch Merkmalsextraktion wird Text im Rahmen der Merkmalsaufbereitung zur weiteren Verarbeitung vorbereitet.</i></p>
    <ol type="1">
      <li>
        Vektorisierung (engl. vectorization)<br>
        <i>Vektorisierungstechniken wandeln Text in nummerische Repräsentationen Embeddings um. Hierbei wird zwischen unsemantischen Embeddings, die Feature Vektoren erzeugen und semantischen Varianten welche Word Embeddings erzeugen differenziert.</i>
      <ol type="2">
       <li>unsemantische Embeddings</li>
       Hierunter werden 
          <ul>
            <li>Frequency Based Embedding</li>
           BoW, TF-IDF
          </ul>
       <li>semantische Embeddings</li>
          <ul>
            <li>Prediction Based Word Embedding</li>
            GloVE; Word2Vec, FastText
            <li>Contextualized Based Word Embedding</li>
            ELMO, BERT, GPT
          </ul>
    </ol>
  </details>
</ol>



<ol> 
  <details>
    <summary>⚪ Datenauswertung (engl. data analysis)</summary>
    <p><i>xxxxxxx</i></p>
    <ol type="1">
      <li><ins>Merkmalsauswertungen (engl. feature Inspections)</ins></li>
      <p><i>Themenverteilungen; Top-Wörter pro Thema</i></p>
        𝛼 (Alpha) - Themenmischung pro Dokument<br>
        β (Beta) - Wortverteilung in Themen<br>
        K<sup>T</sup> (n × k)
    <summary> Evaluation (engl. model evaluation)</summary>
    <i>Modellvergleich</i><br>
  </details>
<ol>
  <details>
    <summary>⚪ Datenpräsentation (engl. data presentation)<br></summary>
    <summary>⚪ Visualisierung</summary>
    <i>grafische Darstellung</i><br>
    <li>PyLDAvis</li>
    <summary>⚪ Aggregation</summary>
    <i>numerische Darstellung</i>
    Modellperformance

  </details>
</ol>