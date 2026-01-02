# Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einen unstrukturierten, organisch entstandenen Datensatz mit schriftlichen Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden.<br>

Das zu erstellende schriftliche Konzept soll die Schritte der NLP-Datenverarbeitung mit Python darlegen. Dabei sollen kurz zwei Techniken zur Vektorisierung sowie zwei Ansätze zur Extraktion von Themen aus dem Datensatz genannt und die verwendeten (externen/integrierten) Bibliotheken genannt werden.

## Konzeption
Der Grafik können die geplanten Phasen des Projekts sowie die zugeordneten Prozesse entnommen werden. 

<img src="https://github.com/IU-KaiW/DLBDSEDA02_Projekt_Advanced-Data-Analysis/blob/main/docs/Visualisierung.jpg" width="1200">


### ⚪ Datenakquisition (engl. data acquisition)
In der Phase der Datenaquisition werden Datensätze für den Input der NLP-Pipeline gesucht, anhand eines [KI-Detektors](https://github.com/Kishanjaisoorya/AI-Text-Detector-python) bewertet und anschließend bewertungsbasiert ausgewählt. Hierzu wird ein trichterförmiger, vier stufiger Prozess bestehend aus Datensatzrecherche, –sammlung, –prüfung sowie –auswahl durchlaufen, an dessen Ende die Eingabe (engl. input) in die Pipeline steht.</br>
<ol>
    <details>
      <summary>⚪ Datensatzrecherche (engl. dataset research)</b></summary>
      <i>Es wird eine Onlinerecherche auf verschiedenen Datenportalen (Kaggle, GitHub, GovData, MendeleyData, u.A.) durchgeführt und nach geeigneten deutschen und englischen Datensätzen gesucht.</i></br>
    </details>
</ol>
<ol>
    <details>
      <summary>⚪ Datensatzsammlung (engl. dataset collection)</summary>
      <i>Offensichtlich synthetisch erzeugte Datensätze werden ignoriert. Datenquellen mit vermutetem organischen Ursprung werden im CSV-Datenformat heruntergeladen und lokal gespeichert.</i></br>
    </details>
</ol>
<ol>    
    <details>
      <summary>⚪ Datensatzprüfung (engl. dataset check)</summary>
      <i>Die gesammelten Datensätze werden anhand eines extern entwickelten KI-Detektors auf synthetisch erzeugte Instanzen (engl. samples) geprüft und mit Labels (REAL / FAKE / ERROR) getaggt.</i><br><br>
      <div style="margin-left: 2em;">
       <code>transformers</code>&nbsp;<code>torch</code><br></br>
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

| Nr.| Bezeichnung                        | Bewertung | Größe     |Quelle                     |
|----|------------------------------------|-----------|-----------|---------------------------|
| 01 | Consumer_Complaints.csv            | n/a       | 59,40  MB |[^01] &nbsp; Kaggle        |
| 02 | rows.csv                           | n/a       | 176    MB |[^02] &nbsp; Kaggle        |
| 03 | Consumer_Complaints.csv            | 13,5 %    | 107,0  MB |[^03] &nbsp; Kaggle/GovData|
| 04 | complaints_processed.csv           | 64,72 %   | 19,8   MB |[^04] &nbsp; Kaggle        |
| 05 | Comcast.csv                        | 82 %      | 60,0   kB |[^05] &nbsp; Kaggle        |
| 06 | user_complaints                    | 0,69 %    | 229,0  kB |[^06] &nbsp; GitHub        |
| 07 | consumer_complaints.csv            | n/a       | 175,39 MB |[^07] &nbsp; Kaggle        |
| 08 | Complaints_Reports_Data.sql        | n/a       | 3,28   MB |[^08] &nbsp; MendeleyData  |
| 09 | chatgpt_reviews.csv                | 35,03 %   | 119,9  MB |[^09] &nbsp; GitHub        |
| 10 | dataset-tickets-multi-lang3-4k.csv | n/a       | 6,87   MB |[^10] Kaggle               |

Der Datensatz mit der prozentualen höchsten Bewertung wird als Korpus bzw. Pipeline Eingabe (engl. Pipeline-Input) für die nachfolgenden Schritte genutzt. Übersteigt die Anzahl der Instanzen die Schwelle von 2000, wird der Datensatz für die folgenden Verarbeitungschritte die Anzahl begrenzt.<br>
______________

###### Pipeline-Eingabe
Es wird Datensatz Nr. 05 "Comcast.csv"[^05]: mit der Bewertung von 82 % gewählt und als Input für die NLP-Pipeline genutzt.

### 🟠 Sprachverarbeitung (engl. NLP-Pipeline)
Merkmale (engl. features) eines Textes oder Dokuments sind Informationen wie Länge (engl. length), Quelle (engl. source) und Datum der Veröffentlichungsdatum (engl. date of publication)

Haupt Freatures / Sekundärfeatuers?



#### 🔴 Datenvorverarbeiten (engl. data pre-processing)
> Durch die Datenvorverarbeiten erfolgt eine *Merkmalsvorbereitung (engl. feature preparation)* für nachfolgende Phasen in einem mehrstufigen Prozess, welcher sich grob die Prozesse Textbereinigung und Merkmalsextraktion einteilen lässt. spaCy [^11]
<ol type="1">
  <details>
    <summary>🔴 Textbereinigung (engl. text cleaning)</summary>
    <p><i>Im Rahmen der Textbereinigung werden Texte zunächst standardisiert und anschließend von Rauschen befreit.</i></p>
    <ol type="1">
      <li>Standardisierung (engl. standardisation)<br></li>
        <i>Im Rahmen der Textbereinigung werden Texte zunächst standadisiert, um inhaltlich relevanten Tokens zu vereinheitlichen. Hierdurch wird vermieden, dass gleiche Inhalte nicht in mehreren, leicht unterschiedlichen Varianten auftreten.</i>
          <div style="margin-left: 2em;">
            <code>???</code>&nbsp;<code>????</code><br></br>
          </div>
      <ol type="2">
            <li><ins>Normalisierung (engl. normalisation)</ins></li>
            Durch die Normalisierung wird Text .... Sie setzt sich zusammen aus: 
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
            </div>
            <ul>
              <li>Kasusumwandlung (engl. case conversion)</li>
              In diesem Schritt erfolgt eine konsequente Kleinschreibung (engl. lowercasing) aller Wörter.
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
              <li>Grundformreduktion (engl. inflection reduction)</li>
              Durch diesen Schritt werden Wörter auf ihre Grundformen reduziert. Da Lemmatisierung (engl. lemmatization) genauer als Stammformreduktion (engl. stemming) ist, wird diese eingesetzt.
              <div style="margin-left: 2em;">
                <code>NLTK (WordNet lemmatizer)</code>&nbsp;<code>spaCy</code><br></br>
              </div>
              <li>Formatnormalisierungen (engl. format normalisations)</li>
              In der Formatnormalisierung erfolgt ...
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            </ul>
            <li><ins>Rechtschreibfehlerkorrektur (engl. spelling correction)</ins></li>
            Durch Rechtschreibkorrekrur werden Tipp bzw. Schreibfehler korrigiert.
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
      </ol>
      </li>
      <li>
      <ins>Rauschentfernung (engl. noise reduction)</ins><br>
        <i>Ziel der Rauschentfernung ist es irrelevante Token (Zeichen und Zeichenketten) für nachfolgende Prozesse zu identifizieren und anschließend zu entfernen.</i>
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
            </div>
            <ol type="2">
            <li>Stoppworte (engl. stopwords)</li>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            <li>Satzzeichen (engl. punctuation marks)</li>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            <li>Leerzeichen (engl. white space)</li>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            <li>Nummern (engl. removing numbers)</li>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            <li>Sonderzeichen (engl. special character)</li>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
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
      </li>
      <div style="margin-left: 2em;">
        <code>SpaCy</code>&nbsp;<code>????</code><br></br>
      </div>
      <li>
        Vokabularerstellung/Wortschatzaufbau (engl. Vocabulary Construction)<br>
        <i>Im Schritt des Wortschatzaufbaus (Vocabulary Construction) wird aus dem tokenisierten Textcorpus ein endliches Vokabular erstellt, das alle einzigartigen Tokens enthält und als Basis für nachfolgende Modelle dient.</i>
      </li>
      <div style="margin-left: 2em;">
        <code>????</code>&nbsp;<code>????</code><br></br>
      </div>
    </ol>
  </details>
</ol>

#### 🟡 Datenverarbeitung (engl. data processing)
> Datenverarbeitung kann mit oder ohne Merkmalsaufbereitung (engl. feature engineering) erfolgen. Bei Datenverarbeitung mit Merkmalsaufbereitung werden Merkmale (engl. features) durch mittels Merkmalsextraktion, Merkmalsumwandlung, Merkmalskonstruktion oder Merkmalsauswahl gewonnen. Bei Datenverarbeitung ohne Merkmalsaufbereitung werden Merkmale aus Rohtexten direkt durch ... (vortrainierte Modelle) gewonnen. 
<ol>
  <details>
    <summary>🟡 Vektorisierung (engl. vectorization)</summary>
    <p><i>Vektorisierungstechniken wandeln Text in Embeddings, nummerische Repräsentationen um. Hierbei wird zwischen unsemantischen Embeddings (feature vektors) und semantischen Varianten (word vectors / sentence vectors) differenziert. Vektorisierungstechniken nutzen Merkmalsextraktion, um Text im Rahmen der Merkmalsaufbereitung für nachfolgende Schritte vorzubereiten.</i></p>
    <ol type="1">
       <li>unsemantische Embeddings</li>
       Hierunter werden 
          <ul>
            <li>Frequency Based Embedding</li>
           BoW, TF-IDF
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
            </div>
          </ul>
       <li>semantische Embeddings</li>
          <ul>
            <li>Prediction Based Word Embedding</li>
            GloVE; Word2Vec, FastText
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
            </div>
            <li>Contextualized Based Word Embedding</li>
            ELMO, BERT, GPT
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
            </div>
          </ul>
  </details>
  <details>
    <summary>🟡 Text Analyse (engl. Text Analytics)</summary>
    <p><i>Textanalyse kann durch Merkmalsmodellierung (engl. feature modeling) oder Merkmalserkennung (engl. feature recognition) auf syntaktischer, lexikalischer oder semantischer Ebene erfolgen. Syntaktische Analyse befasst sich mit den Merkmalen der Sprache wie Kategorien, Wortgrenzen und grammatikalischen Funktionen wie Wortarten und ihre Zusammensetzung in Phrasen (Alpar et al., 2023, p. 45). Lexikalische Analysen befassen sich mit der Bedeutung einzelner Wörter. Semantische Analysen hingehen konzentrieren sich auf die Bedeutung größerer Textstücke. Dabei geht es um das Verstehen ganzer, in natürlicher Sprache verfasster Texte (Lane et al., 2019).</i></p><!--syntaktisch-Tokenization / Part-of-Speech(POS)-Tagging-->
    <ol type="1">
       <li>🟡 Merkmalsmodellierung (engl. feature modeling)</li>
       Unter Merkmalsmodellierung versteht man die inhaltliche Strukturierung und Deutung des zuvor vektorisierten Texts. Sie legt fest, wie Merkmale thematisch oder semantisch für die Textanalyse genutzt werden können. Über Themenmodellierung (engl. topic modeling) können Themen unüberwacht (engl. unsuperviced) mittels Merkmalsextraktion (engl. feature extraction) oder Merkmalsumwandlung (engl. feature transformation) identifiziert werden.<br><br>
          <ul>
            <li>Latent Dirichlet Allocation (LDA)<br></li>
            <i>Merkmalsextraktion</i>
            Identifiziert latente Themen in einer Sammlung von Dokumenten und stellt Dokumente basierend auf ihren Verteilungen über diese Themen dar.<br>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
           <li>Latent Semantic Analysis (LSA)<br></i></li>
            <i>Merkmalsumwandlung</i><br>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            <li>Merkmalsauswahl (engl. feature selection)<br>
            <i>Zum Ende werden die besten Merkmale ausgewählt.</i><br><br>
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
          </ul>
       <li>🟡 Merkmalserkennung (engl. feature recognition)</li>
          <ul>
            <li>1</li>
            xxx
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
          </ul>

  </details>  
  <details>
    <summary>🟡 Merkmalslernen (engl. feature learning / representation learning)</summary></br>
    In der Datenverarbeitung beginnt die Merkmalsaufbereitung (engl. feature engineering) und das Merkmalslernen (engl. feature learning / representation learning).
    <i>Beginn der Modellbildung für Aufgabe<i>
  </details>
</ol>

###### Pipeline Ausgabe (engl. Pipeline Output)
Die verarbeiteten Daten fließen in die Datenkonsolidierung ein.
______________

### 🔵 Datenkonsolidierung (engl. data consolidation)
Im Rahmen der Datenkonsolidierung erfolgt die Datennachverarbeitung (engl. data post-processing) in der die Merkmalsanalysen (engl. feature analysis) sowie die Modellevaluationen (engl. model evaluations) durchgeführt und letztlich als Datenpräsentation (engl. data presentation) aufbereitet werden.


Datenauswertung (engl. data analysis)
Datenpräsentation  (engl. data presentation)

#### 🔵 Merkmalsauswertungen (engl. feature Inspections)
>Themenverteilungen; Top-Wörter pro Thema; 𝛼 (Alpha) – Themenmischung pro Dokument; β (Beta) – Wortverteilung in Themen; K<sup>T</sup> (n × k)
<ol>
    <details>
      <summary>🔵 Aggregation (engl. aggregation)</summary>
      <p><i>alphanumerische Darstellungen - Aggregation reduziert die Datenmenge durch mathematische Operationen wie Summe, Mittelwert, Zählung oder Maximum über Gruppierungen (z. B. nach Token-Typ, Dokument oder Zeitraum). In NLP könnte dies die Häufigkeitsverteilung von n-Grammen pro Domäne oder die durchschnittliche Embedding-Distanz pro Klasse bedeuten. Sie erfolgt vor der Visualisierung, um Überladung zu vermeiden, und ist rein datenverarbeitend ohne grafische Elemente. Aggregation fasst Rohdaten zu kompakteren Zusammenfassungen zusammen.</i></p>
      <div style="margin-left: 2em;">
        <code>???</code>&nbsp;<code>????</code><br></br>
      </div>
    <ol>
    </details>
    <details>
      <summary>🔵 Visualisierung (engl. visualization)</summary>
      <p><i>grafische Darstellung - Visualisierung stellt die aggregierten Daten grafisch dar, um Muster erkennbar zu machen.</i></p>
      <div style="margin-left: 2em;">
        <code>???</code>&nbsp;<code>????</code><br></br>
      </div>
    </details>
  </li>
</ol>

#### 🔵 Modellauswertung (engl. model evaluation)
> Modellvergleich / Modellperformance; Evaluation (engl. model evaluation)
<ol>
    <details>
      <summary>🔵 Kohärenz (engl. coherence)</summary>
        <p><i>Topic Coherence:</i></p>
        <code>???</code>&nbsp;<code>????</code><br>
    <ol>
    </details>
    <details>
      <summary>🔵 Perplexität (engl. perplexity)</summary>
        <p><i>xxxxx</i></p>
        <code>???</code>&nbsp;<code>????</code><br>
    </details>
  </li>
</ol>

Topic Diversity:
`library 3`
`library 4`
Silhouette Score:
`library 3`
`library 4`
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
Als Entwicklungsumgebung (engl. integrated development environment - IDE) wurde Visual Studio Code (VSCode) genutzt. 

### Vorbereitende Installation (engl. preparatory setup)


#### Datenvalidierung (engl. data validation)
KI Detektor

Requirements
Python 3.x

```python
`pip install transformers`
```
python AITextDetector.py

Die Spaltenbeschriftung der textführende Spalte muss in "text" umgenannt werden.

### Erstinstallation (engl. initial setup)

#### Laufzeitumgebung (engl. runtime environment)
Als virtuelle Umgebungen stehen in Python "venv" und "conda" zur Verfügung. Aufgrund der Abhängigkeit von spaCy wird conda genutzt.

```console
`pip install conda`
```

#### Voraussetzungen (engl. prerequisite setup)
Python 3.8+

##### Abhängigkeiten (engl. dependencies)

#### Bibliotheken (engl. librarys)
```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import spacy as sp
import nltk

```    

##### Sprachverarbeitung (engl. NLP-Pipeline)
##### Datenkonsolidierung (engl. data consolidation)
###### Visualisierung
###### Aggregation

### Referenzen
<ul>
<li>Externe Software</li><br>
Jai Soorya N, K. (2023). AI-Text-Detector-python [Software]. https://github.com/Kishanjaisoorya/AI-Text-Detector-python<br>

<li>Bibliotheken (engl. librarys)</li><br>

<ins>Python-Standardbibliothek</ins><br><br>

`stdlib`


| Bibliothek    | Website                                                               | Dokumentation |Verwendung | Funktionen|
|-------------- |-----------------------------------------------------------------------|---------------|-----------|-----------|
|`re`           |                                                                       |               |           |           |
|`csv`          |                                                                       |               |           |           |
https://docs.python.org/3.9/py-modindex.html


<ins>externe Bibliothek</ins>
| Bibliothek    | Website                                                               | Dokumentation |Verwendung | Funktionen|
|-------------- |-----------------------------------------------------------------------|---------------|-----------|-----------|
|`pandas`       |https://pandas.pydata.org                                              |               |           |           |

<br>`pandas` 
<br>`gensim` Řehůřek, R. (2024, August 10). LDA Model. Gensim. https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html<br>
<br>`gensim` Řehůřek, R. (2025). Gensim: Topic modelling for humans. Gemsim. https://radimrehurek.com/gensim/<br>
<br>`seaborn` Waskom, M. (2021). seaborn: Statistical data visualization. Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021<br>


<li>Literatur</li>

###### Skripte
IU Internationale Hochschule. (2024). Advanced Data Analysis (DLBDSEDA01_D) [Lernskript]. 001-2024-1210.<br>
IU Internationale Hochschule. (2023). Artificial Intelligence (K. Schaaff, Übers.; DLBDSEAIS01_D) [Studienskript]. 001-2023-1213.

###### Artikel
Blei, David M. and Ng, Andrew Y. and Jordan, Michael I.: Latent dirichlet allocation. In: The Journal of Machine Learning Research. Nr. 3, 3. Januar 2003, S. 993–1022<br>
<br>Blum et al., 2020 ?<br>
<br>Abbott, D., Kommer, I., & Kommer, C. (2025). Datenvisualisierung im praktischen Einsatz: Ansprechende Diagramme und Dashboards gestalte (1. Auflage). dpunkt.verlag.<br>
<br>Steiner, D., & Zeneli, G. (2019). Texploration: Automatische Analyse von grossen Textsammlungen [Bachelorarbeit, Zürcher Hochschule für Angewandte Wissenschaften]. https://www.zhaw.ch/storage/engineering/institute-zentren/cai/BA19_Texploration_Steiner_Zeneli.pdf<br>

<li>Websites</li>
<br>Sanchhaya Education Private Ltd. (2025, September 3). NLP Gensim Tutorial—Complete Guide For Beginners [Bildungsplattform]. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/nlp-gensim-tutorial-complete-guide-for-beginners/<br>
<br>Freitas, G. & Lily Hulatt. (2025). Feature Selection: Methoden & Techniken [Bildungsplattform]. StudySmarter. https://www.studysmarter.de/schule/informatik/computerlinguistik-theorie/feature-selection/<br>
<br>Řehůřek, R. (2024, August 10). LDA Model. Gensim. https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html<br>
<br>Helm, C. (2025, Mai 8). spaCy vs NLTK – Was ist die bessere Wahl für NLP? Konfuzio. https://konfuzio.com/de/spacy-vs-nltk/<br>
<br>GitHub - https://docs.github.com/de/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax<br>


Lane, H., Howard, C, & Hapke, H. M. (2019). Natural language processing in action: Understanding, analyzing, and generating text with Python. Manning.
Alpar, P., Alt, R., Bensberg, F., & Czarnecki, C. (2023). Anwendungsorientierte Wirtschaftsinformatik: Strategische Planung, Entwicklung und Nutzung von Informationssystemen (10. Auflage). Springer Vieweg. https://doi.org/10.1007/978-3-658-40352-2


🟥🟨🟦🟫⬜🟧🟩🟪◼️◻️🔶🔸🔘
:red_square:

🟣🟢🟠🔴🔵🟡🟤⚫
https://emojiterra.com/de/gelber-kreis/

        <ul>
          <li>PyLDAvis</li>
        </ul>

[^01]: [Datensatz01] (https://www.kaggle.com/datasets/ashwinik/consumer-complaints-financial-products)
[^02]: [Datensatz02] (https://www.kaggle.com/datasets/selener/consumer-complaint-database)
[^03]: [Datensatz03] (https://www.kaggle.com/code/saurabhsawhney/nlp-complaints-classification)
[^04]: [Datensatz04] (https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)
[^05]: [Datensatz05] (https://www.kaggle.com/datasets/yasserh/comcast-telecom-complaints)
[^06]: [Datensatz07] (https://github.com/gurneetjuneja/NLP-Problem-Solving/blob/main/user_complaints.csv)
[^07]: [Datensatz08] (https://www.kaggle.com/code/mchirico/analyzing-text-in-consumer-complaints)
[^08]: [Datensatz09] (https://data.mendeley.com/datasets/w2cp7h53s5/1)
[^09]: [Datensatz10] (https://github.com/Schossi2908/DLBDSEDA02_D)
[^10]: [Datensatz11] (https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
[^11]: [Webseite] (Helm, C. (2025, Mai 8). spaCy vs NLTK – Was ist die bessere Wahl für NLP? Konfuzio. https://konfuzio.com/de/spacy-vs-nltk/)