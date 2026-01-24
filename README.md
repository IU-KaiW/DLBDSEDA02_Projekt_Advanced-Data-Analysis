# Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
NLP (natural language processing) lässt sich in drei Bereiche einteilen ASR (automatic speech recognition), NLU (natural language understanding) und NLG (natural language generation). Dieses Projekt befasst sich mit dem Bereich dem Verständnis natürlicher Sprache (NLU). 

Ziel der Aufgabe ist es NLP-Techniken auf einem organisch entstandenen Datensatz mit schriftlichen Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den unstrukturierten Texten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden.<br>

Das schriftliche Konzept hierzu soll die Schritte der NLP-Datenverarbeitung mit Python darlegen. Dabei sollen zwei Techniken zur Vektorisierung der Beschwerdetexte sowie zwei Ansätze zur Extraktion von Themen aus dem Datensatz genannt sowie die verwendeten (integrierten/externen)Python-Bibliotheken aufgeführt werden.

Durch einen Klick auf ► werden Erläuterungen, Unterschritte und Softwarebibiliotheken sichtbar.

## Konzeption
Die ausgearbeitete Konzeption lässt sich grob in 3 Phasen einteilen. Datenvorverarbeitung (engl. data pipeline), Sprachdatenverarbeitung (engl. NLP-Pipeline) sowie die Datennachverarbeitung (engl. data post-processing).

Der Grafik können die geplanten Phasen des Projekts sowie die zugeordneten Prozesse entnommen werden. 

<img src="https://github.com/IU-KaiW/DLBDSEDA02_Projekt_Advanced-Data-Analysis/blob/main/docs/Visualisierung.jpg" width="1200">

## ⚪ Datensatzverarbeitung (engl. dataset pipeline)
<img src="1 - Datensatzverarbeitung (engl. dataset pipeline).jpg" width="1200">

### Datensatzakquisition (engl. dataset acquisition)
In der Phase der Datenaquisition werden Datensätze für den Input der NLP-Pipeline gesucht, anhand eines [KI-Detektors](https://github.com/Kishanjaisoorya/AI-Text-Detector-python) bewertet und anschließend bewertungsbasiert ausgewählt. Hierzu wird ein trichterförmiger, vier stufiger Prozess bestehend aus Datensatzrecherche, –sammlung, –prüfung sowie –auswahl durchlaufen, um die Eingabe für die NLP-Pipeline zu bestimmen.</br>
<ol>
    <details>
      <summary>⚪ Datensatzrecherche (engl. dataset research)</b></summary>
      <i>Es wird eine Onlinerecherche auf verschiedenen Datenportalen (Kaggle, GitHub, GovData, MendeleyData, u.A.) durchgeführt und nach geeigneten deutschen und englischen Datensätzen gesucht.</i></br>
    </details>
</ol>
<ol>
    <details>
      <summary>⚪ Datensatzsammlung (engl. dataset collection)</summary>
      <i>Offensichtlich synthetisch erzeugte Datensätze werden ignoriert. Datenquellen mit vermutetem organischen Ursprung werden im CSV-Datenformat manuell oder per API heruntergeladen und lokal gespeichert.</i></br>
    </details>
</ol>
<ol>    
    <details>
      <summary>⚪ Datensatzprüfung (engl. dataset check)</summary>
      <i>Die gesammelten Datensätze werden anhand eines extern (vortrainierten?) entwickelten KI-Detektors auf synthetisch erzeugte Instanzen (engl. samples) geprüft und mit Labels (REAL / FAKE / ERROR) getaggt. [genutze Technik - BERT ? Beschränkung der Zeichen]</i><br><br>
      <div style="margin-left: 2em;">
       <code>transformers</code>&nbsp;<code>torch</code><br></br>
    </details>
    <details>
      <summary>⚪ Datensatzauswahl (engl. dataset selection)</summary>
      <i>Durch eine Häufigkeitsauswertung der Label wird der Datensatz mit dem prozentual höchsten Anteil an organischen (REAL-Label) Instanzen die die Formel:</i><br>
      <br>
      $$\%\text{ organisch} = \left(\frac{REAL}{REAL + FAKE + ERROR}\right) \cdot100$$
      <br>
      <br><i>ausgewertet. Die Wahrscheinlichkeit eines organischen Ursprungs erscheint höher, je höher der Prozentsatz organisch identifizierter Instanzen im Verhältnis zum Gesamtdatensatz ist. Kann ein Datensatz nicht in angemessener Zeit (30 min.) durch das Modell verarbeitet werden, wird die Prüfung abgebrochen und die Bewertung als n/a markiert. Der Datensatz fließt dann nicht in den Ergebnisvergleich ein. Der Datensatz mit der prozentualen höchsten Bewertung wird als Korpus für die nachfolgenden Schritte genutzt. Übersteigen seine Instanzen die Schwelle von 2000, wird der Datensatz für die folgenden Verarbeitungschritte darauf begrenzt.<br><br></i>

| Nr.| Bezeichnung                        | Bewertung | Größe     |Quelle                     |
|----|------------------------------------|-----------|-----------|---------------------------|
| 01 | Consumer_Complaints.csv            | n/a       | 59,40  MB |[^01] &nbsp; Kaggle        |
| 02 | rows.csv                           | n/a       | 176    MB |[^02] &nbsp; Kaggle        |
| 03 | Consumer_Complaints.csv            | 13,5 %    | 107,0  MB |[^03] &nbsp; Kaggle/GovData|
| 04 | complaints_processed.csv           | 64,72 %   | 19,8   MB |[^04] &nbsp; Kaggle        |
| 05 | complaints_data.csv                | 82 %      | 7,20   MB |[^05] &nbsp; GitHub        |
| 06 | user_complaints                    | 0,69 %    | 229,0  kB |[^06] &nbsp; GitHub        |
| 07 | consumer_complaints.csv            | n/a       | 175,39 MB |[^07] &nbsp; Kaggle        |
| 08 | Complaints_Reports_Data.sql        | n/a       | 3,28   MB |[^08] &nbsp; MendeleyData  |
| 09 | chatgpt_reviews.csv                | 35,03 %   | 119,9  MB |[^09] &nbsp; GitHub        |
| 10 | dataset-tickets-multi-lang3-4k.csv | n/a       | 6,87   MB |[^10] Kaggle               |

Es wird Datensatz Nr. 05[^05] mit dem Dateinamen "complaints_data.csv" gewählt da dieser ein Scoring von 82 % erreicht. 

  </details>
</ol>

### Datensatzsichtung (engl. dataset inspection)
In der Phase der Datensatzsichtung wird eine Explorative Datenanalyse (engl. exploratory data analysis) durchgeführt, um Muster, Qualitätsprobleme und Strukturen des Datensatzes zu erkennen, damit diese zur Datensatzaufbereitung (engl. dataset preparation) und in den anschließenden Phasen berücksichtigt werden können. Die EDA wurde mittels des selbstgeschrieben Python-Skripts "Explorative Datenanalyse (EDA).ipynb" durchgeführt, um den gewählten Datensatz besser zu verstehen. Hierdurch wurden eine Datentrukturanalyse, sowie eine Analyse der sturtutierten und untrukturierten Bestandteilen des Datensatzes durchgeführt. 
<ol>    
    <details>
      <summary>⚪ Datenstrukturanalyse (engl. data structure analysis)</summary>
      <i>Die Datenstruktur des gewählten Datensatzes ist ein Spezialfall einer "Delimiter Separated Value"-Datei welche als Trennzeichen Kommta (engl. comma) nutzt (Klein, 2023, p. 261-262)
      
  [^12].</i>
      <i>Diese sog. CSV-Datei verfügt im vorliegende Fall über eine Header und 5659 Zeilen, sprich 5660 Zeilen insgesamt, welche jeweils in 4 Spalten organisiert sind</i><br>
      
  <br><i>Die in der Datei enthaltenen Daten lassen sich in <ins>struktierte Daten</ins> und <ins>unstrukturierte Daten </ins> unterteilen, wobei letztere als Input für die NLP-Pipeline genutzt wird.<br></i>

|author                             |posted_on                 |rating |text              |
|-----------------------------------|--------------------------|-------|------------------|
|`<Benutzername>`of`<US-Ortsangabe>`|`<Monat>`.`<Tag>`,`<Jahr>`|`<0-5>`|`<Beschwerdetext>`|

Die in der Datei enthaltenen Daten lassen sich in <ins>struktierte Daten</ins> und <ins>unstrukturierte Daten</ins> unterteilen, wobei letztere als Input für die NLP-Pipeline genutzt wird.<br>
  </details>
</ol>

<ol>    
    <details>
      <summary>⚪ Explorative Datenanalyse (engl. exploratory data analysis)</summary>
      <i>In der EDA werden Textdaten untersucht, um Muster, Qualitätsprobleme und Strukturen zu erkennen.<br></i>


  </details>
</ol>

<ul>
  <li><ins>struktierte Daten</ins></li>
  In strukturierter Form liegen die Spalten "author", "posted_on" und "rating" vor. Diesen Informationen ist gemein, dass sie ohne größere Vorverarbeitung direkt weiterverarbeitet werden können, da die Informationen meist in einheitlicher (normalisierter Form) vorliegen.<br><br>

  <ul>
  <li><ins> "author"</ins>
  
  > Die Zeilen der Spalte enthalten jeweils den alphanumerischen`<Benutzernamen>` des Beschwerdeverfassers sowie eine, durch ein "of" getrente, US-Ortsangabe welche im Format `<Ortsname "of" US-Bundesstaat>` vorliegt. 
  
  Die Datenexploration durch eine Ortsdatenanalyse zeigte, dass 3 US-Ortsangaben ['BC', 'ON', 'PE'] ungültig sind, aus 17 Bundesstaaten eine dreistellige Anzahl an Beschwerden zu verzeichnen ist, von deren auf ['FL: 778', 'CA: 554', 'GA: 414'] entfallen und aus 5 Bundesstaaten ['IA', 'MT', 'OK', 'RI', 'SD'] keine Beschwerden erfasst wurden.
<br>
  
  <li><ins>"posted_on"</ins><br>

  > Die Zeilen der Spalte "posted_on" enhalten Datumsangaben mit alphabetisch abgekürzer Monatsangabe über einen Zeitraum von 16 Jahren (2000 - 2016) im amerikanischem Format `<Monat>`.`<Tag>`,`<Jahr>`. 
  
  
   Im Rahmen der Datenexploration wurde eine Zeitdatenanlyse durchgeführt welche Muster in der (jährlichen, monatlichen, wöchentlichen) Verteilung der Beschwerden im Datensatz über den Zeitraum vom 31.07.2000 bis 22.11.201 zeigte.

   <ins>jährliche Verteilung</ins><br>
   Die EDA zeigt, dass die meisten Beschwerden im Jahr 2015 erfolgten sind. 
   
   <ins>monatliche Verteilung</ins><br>
   Die EDA zeigte weiter, dass die meisten Beschwerden im August (540) und wie wenigsten Beschwerden im April (369) abgesetzt wurden, wobei der Datensatz ein saisonales Muster zeigt. 
   
   <ins>wöchentliche Verteilung</ins><br>
   Die Verteilung der Beschwerden aufgeschlüsselt nach Wochentagen zeigt, dass die meisten Beschwerden mittwochs (993), dienstags (960) und donnerstags (861), gefolgt von Montag (820) und Freitag (802) abgesezt wurden, wohingegen an den Tagen der Wochenenden weniger Beschwerden zu verzeichnen sind Samstag (659) und Sonntags (564). Dieses Muster deutet auf einen organischen Ursprung des Datensatz hin.

  <li><ins>"rating"</ins><br>
  
  > Die Zeilen der Spalte "rating" enthalten Bewertungen auf einer Skala `<0-5>`.<br>

  Die Anzahl der Bewertungen nach "rating" ist wie folgt verteilt [rating: 0=1560 (27.57%); 1=3734 (65.98%); 2=260 (4.59%); 3=54 (0.95%); 4=19 (0.34%); 5=32 (0.57%)] was einen Überhang niedriger Bewertungen zeigt. Dies weist ebenfalls auf einen organischen Datensatz hin, da Beschwerden grundsätzlich negativ sind. 

  </ul>
</ul>

<ul>
  <li><ins>unstruktuierte Daten</ins></li>
  Unstrukturierte Daten sind Informationen, die in einer nicht identifizierbaren Datenstruktur vorliegen. Ein typisches Beispiel dafür sind natürlichsprachliche Texte wie sie in der Spalte "text" vorhanden sind.<br>
  <ul>
  <li><ins>"text"</ins></li>
  
  > In den Zeilen der Spalte "text" befindet sich ein englischer `<Beschwerdetext>`. Er besteht aus Wörtern (Zeichenketten, sprich Folgen von Buchstaben, Ziffern, Satzzeichen, ect.) die konkateniert Sätze bilden die Zeit- und Datumsangaben in unterschiedlichen Formatierungen, Großschreibungen, Aufzählungen und Sonderzeichen enthalten was bei der Sprachverarbeitung zu beachten ist.<br>  
  
  Die EDA zeigte dass die Texte im Median aus 1.239,94 Zeichen bestehen.
  </ul>
</ul>

<ul>
  <li><ins>fehlerhafte Daten</ins></li>
  Im Zuge der Datenexploration ist aufgefallen, dass sich sowohl in den unstrukturiert vorliegenden Daten 30 NaNs, nicht vorhandene Informationen befanden. die bereinigt werden müssen, um Verzerrungen in der späteren Modellbildung zu vermeiden. Es wurden 30 NaNs in der Spalte 'text' und fehlende Daten und (XXX) Duplikate gefunden<br>
  <ul>
  <li><ins>XXXXX</ins></li>
  
  > Die Analyse fehlender Daten zeigte, dass der Datensatz <br>  
  </ul>
</ul>

## Datensatzaufbereitung (engl. dataset preparation)
In der Phase der Datensatzbereinigung werden die in der EDA gewonnen Erkenntnisse genutzt, um den Datensatz für den Anwendungsfall vorzubereiten. Hierzu wird eine Datenbereinigung sowie eine Datenvalisierung durchgeführt, wodurch diejenigen Daten bestimmt werden, die weiter verarbeitet werden sollen.
### Datensatzbereinigung (engl. dataset cleaning)
###### Fehlwertbehandlung
Die Behandlung von Fehlwerten wie NaNs (Not a Number) oder NaTs (Not a Text) kann durch Listenweisen Fallausschluss, durch welchen Zeilen ohne Text oder Text unter einer Mindeslänge entfernt wird oder Imputation, das Auffüllen oder Ersetzen fehlender oder unvollständiger Textelemente durch geschätzte Werte, damit der Datensatz für Modelltraining oder Analyse vollständig nutzbar bleibt.
###### Duplikatentfernung
Durch die Duplikatentfernung werden doppelte Zeilen im Datensatz entfernt, um Verzerrungen des NLP-Models zu vermeiden. 
### Datensatzvalidierung (engl. dataset validation)
Im Rahmen der Datensatzvalisierung werden fehlerhafte Daten korrigiert, verworfen oder speziell behandelt um Datenqualität und Aussagekraft zu sichern.


## 🟠 Sprachverarbeitung (engl. NLP-Pipeline)
Merkmale (engl. features) eines Textes oder Dokuments "sind kategorielle oder numerische Größen, anhand derer Machine-Learning-Algorithmen oder neuronale Netze […] klassifizieren können (Timmermann, 2019)."[^16] Merkmale eines Textes oder Dokuments können Informationen auf lexikalischer, syntaktischer oder semantischer Ebene umfassen. Die Themenmodellierung liegt auf der semantischer Ebene.

###### Pipeline Eingabe (engl. pipeline input)

```python
df = pd.read.csv ('URL')
```

<div style="margin-left: 2em;">
  <code>pandas</code>&nbsp;<code>????</code><br>
</div>

#### 🔴 Datenvorverarbeiten (engl. data pre-processing)
> Während der Datenvorverarbeitung erfolgt eine *Merkmalsvorbereitung (engl. feature preparation)* für nachfolgende Schritte in einem mehrstufigen Prozess, welcher sich grob in Textbereinigung (engl. text cleaning) und Merkmalsextraktion unterteilen lässt. 
<div style="margin-left: 2em;">
  <code>spaCy</code>&nbsp;<code>NLTK</code><br><br>
</div>
<ol type="1">
  <details>
    <summary>🔴 Textbereinigung (engl. text cleaning)</summary>
    <p><i>Im Rahmen der Textbereinigung werden Texte zunächst standardisiert und anschließend von Rauschen befreit.</i></p>
    <ol type="1">
      </li>
      <li>Rauschentfernung (engl. noise reduction)</li><br>
        <i>Ziel der Rauschentfernung ist es irrelevante Token (Zeichen und Zeichenketten) für nachfolgende Prozesse zu identifizieren und zu löschen.</i>
            <div style="margin-left: 2em;">
              <code>NLTK</code>&nbsp;<code>regex</code><br></br>
            </div>
            <ol type="2">
            <li><ins>Stoppworte (engl. stopwords)</ins></li>
              <div style="margin-left: 2em;">
                <code>NLTK(stopwords)</code>&nbsp;<code>????</code><br></br>
              </div>
            <li><ins>Satzzeichen (engl. punctuation marks)</ins></li>
              <div style="margin-left: 2em;">
                <code>regex</code>&nbsp;<code>????</code><br></br>
              </div>
            <li><ins>Leerzeichen (engl. white space)</ins></li>
              <div style="margin-left: 2em;">
                <code>regex</code>&nbsp;<code>????</code><br></br>
              </div>
            <li><ins>Nummern (engl. removing numbers)</ins></li>
              <div style="margin-left: 2em;">
                <code>regex</code>&nbsp;<code>????</code><br></br>
              </div>
            <li><ins>Sonderzeichen (engl. special character)</ins></li>
              <div style="margin-left: 2em;">
                <code>regex</code>&nbsp;<code>????</code><br></br>
              </div>
      </ol>
      <li>Standardisierung (engl. standardisation)<br></li>
        <i>Im Rahmen der Textbereinigung werden Texte zunächst standadisiert, um inhaltlich relevanten Tokens zu vereinheitlichen. Hierdurch wird vermieden, dass gleiche Inhalte nicht in mehreren, leicht unterschiedlichen Varianten auftreten.</i>
          <div style="margin-left: 2em;">
            <code>???</code>&nbsp;<code>????</code><br></br>
          </div>
      <ol type="2">
            <li><ins>Normalisierung (engl. normalisation)</ins></li>
            Durch die Normalisierung wird Text .... Sie setzt sich zusammen aus: 
            <div style="margin-left: 2em;">
              <code>stdlib</code>&nbsp;<code>????</code><br></br>
            </div>
            <ul>
              <li>Kasusumwandlung (engl. case conversion)</li>
              In diesem Schritt erfolgt eine konsequente Kleinschreibung (engl. lowercasing) aller Wörter.
              <div style="margin-left: 2em;">
                <code>stdlib(.lower)</code>&nbsp;<code>????</code><br></br>
              <li>Formatnormalisierungen (engl. format normalisations)</li>
              In der Formatnormalisierung erfolgt die Normalisierung von Schreibweisen (Datenformate oder Zahlenformaten) und Sonderformen (Emojis).
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            </ul>
            <li><ins>Rechtschreibfehlerkorrektur (engl. spelling correction)</ins></li>
            Durch Rechtschreibkorrekrur werden Tipp bzw. Schreibfehler korrigiert.
              <div style="margin-left: 2em;">
                <code>pyspellchecker</code>&nbsp;<code>????</code><br></br>
              </div>
      </ol>
    </ol>
  </details>
  <details>
    <summary>🔴 Linguistische Analyse (engl. Linguistic Processing)</summary>
    <p><i>Im Rahmen der lingusitischen Analyse erfolgt eine Merkmalsvorbereitung.</i></p>
    <ol type="1">
      <li>
        Tokenisierung (engl. tokenization)<br>
        <i>Durch Tokenisierung wird der vorbereitet Text in Einzeltoken (Worte) oder N-Gramme (Phrasen - engl. chunks?) wie z.B. Sätze zerlegt. BEARBEITEN
        Tokenisierung zerlegt Text in Token (Wörter, Subwörter oder Zeichen), aus denen das Vokabular als Menge eindeutiger Token-IDs entsteht.
        </i><br>
      </li>
      <div style="margin-left: 2em;">
        <code>SpaCy</code>&nbsp;<code>NLTK(word_tokenize; sent_tokenize)</code><br></br>
      </div>
      </div>
       <li>Grundformreduktion (engl. inflection reduction)</li>
        Durch Grundformreduktion werden Wörter auf ihre Grundformen reduziert. Da Lemmatisierung (engl. lemmatization) genauer als Stammformreduktion (engl. stemming) ist, wird diese eingesetzt.
      <div style="margin-left: 2em;">
         <code>NLTK (WordNetLemmatizer)</code>&nbsp;<code>spaCy()</code><br></br>
      </div>
      <li>
        Vokabularerstellung/Wortschatzaufbau (engl. vocabulary construction)<br>
        <i>Im Schritt des Wortschatzaufbaus wird aus dem tokenisierten Textkorpus ein endliches Vokabular erstellt, das allen Token eine eindeutige Token-IDs zuweist. Das Vokabular stellt eine Menge eindeutiger Token-IDs dar.</i>
      </li>
      <div style="margin-left: 2em;">
        <code>sklearn(CountVectorizer)</code>&nbsp;<code>????</code><br></br>
      </div>
    </ol>
  </details>
</ol>

#### 🟡 Datenverarbeitung (engl. data processing)
> Datenverarbeitung kann mit oder ohne Merkmalsaufbereitung (engl. feature engineering) erfolgen. Bei Datenverarbeitung mit Merkmalsaufbereitung werden die Merkmale (engl. features) durch Merkmalsextraktion, Merkmalsumwandlung oder Merkmalskonstruktion gewonnen. Bei Datenverarbeitung ohne Merkmalsaufbereitung werden Merkmale direkt aus Rohtexten mittels trainierter Modelle durch automatisches Feature Engineering gewonnen.
<div style="margin-left: 2em;">
  <code>???</code>&nbsp;<code>???</code><br><br>
</div>

<ol>
  <details>
    <summary>🟡 Vektorisierung (engl. vectorization)</summary>
    <p><i> Die eindeutigen Token (Wörter, Subwörter oder Zeichen) aus dem Vokabular werden durch Vektorisierungstechniken in numerische Repräsentationen überführt, die als Merkmalsvektoren in einem n‑dimensionalen Merkmalsraum dargestellt und zu Merkmalsmatrizen zusammengefasst werden. Dabei wird zwischen Merkmalsvektoren (engl. feature vectors), die Merkmale als dünn besetzte Vektoren (engl. sparse vectors) repräsentieren, und Einbettungen (engl. embeddings), die Merkmale als dicht besetzte Vektoren darstellen, unterschieden. Frequenzbasierte Methoden erzeugen dünn besetzte Merkmalsvektoren basierend auf Vokabularpositionen, während Embeddings jedem Token einen dichten Vektor in einem semantischen Raum zuweisen, in dem Abstände bzw. Ähnlichkeitsmaße zur Bestimmung der semantischen Ähnlichkeit dienen. In diesem Zusammenhang unterscheidet man daher zwischen unsemantischen Embeddings, rein frequenzbasierten Vektorrepräsentationen und semantischen Embeddings, die auf vorhersage- oder kontextbasierten Verfahren beruhen. Vektorisierungstechniken nutzen Merkmalsextraktion, um Texte je nach Anwendungsfall als Bag‑Vektoren, Wort‑Vektoren, Satz‑Vektoren, Segment‑Vektoren oder Dokumenten‑Vektoren für Modelle aufzubereiten.</i></p>
    <ol type="1">
       <li>unsemantische Embeddings</li>
       Spannen keinen semantischen Merkmalsraum auf, sondern liefern dünnbesetzte Vektoren (engl. sparse vektors) 
        <ul>
            <li>Frequency Based Embedding</li>
                Bag-of-X<br>
                BoW: Ein Bag-of-Words „Ein Bag-of-Words-Vektor hat für jedes Wort eine eigene Dimension. Wenn das Vokabular n Wörter umfasst, wird ein Dokument zu einem Punkt 1 in einem n-dimensionalen Raum.“ (Zheng und Casari, 2019, p. 41)<br>
                <div style="margin-left: 2em;">
                  <code>sklearn (CountVectorizer)</code>&nbsp;<code>????</code><br></br>
                </div>
                TF-IDF (term frequency times invers documentfrequency)<br>
                <div style="margin-left: 2em;">
                  <code>sklearn (TfidfVecrorizer)</code>
                </div>
          </ul>
            <li>semantische Embeddings</li>
            Spannen einen semantischen Merkmalsraum auf und liefern dichtbesetzte Vektoren (engl. dense vektors).
                (word vectors)
          <ul>
            <li>Prediction Based Word Embedding</li>
            (word vectors)
                GloVE (global vectors for word vectorization)<br>
                Word2Vec<br>
                FastText<br>
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
            </div>
            <li>Contextualized Based Word Embedding</li>
            (word vectors)
                ELMO (Embeddings from Language Models)<br>
                BERT (Bidirectional Encoder Representations from Transformers)<br>
                GPT  (Generative Pre-trained Transformer)<br>
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
              (sentence vectors)
                SBERT Embedding (Embeddings from Language Models)<br>
                <br>
            <div style="margin-left: 2em;">
              <code>???</code>&nbsp;<code>????</code><br></br>
            </div>
          </ul>

          

  </details>
  <details>
    <summary>🟡 Text Analyse (engl. Text Analytics)</summary>
    <p><i>Textanalyse kann durch Merkmalsmodellierung (engl. feature modeling) oder Merkmalserkennung (engl. feature recognition) auf syntaktischer, lexikalischer oder semantischer Ebene erfolgen. Syntaktische Analyse befasst sich mit den Merkmalen der Sprache wie Kategorien, Wortgrenzen und grammatikalischen Funktionen wie Wortarten und ihre Zusammensetzung in Phrasen (Alpar et al., 2023, p. 45). Lexikalische Analysen befassen sich mit der Bedeutung einzelner Wörter. Semantische Analysen hingehen konzentrieren sich auf die Bedeutung größerer Textstücke. Dabei geht es um das Verstehen ganzer, in natürlicher Sprache verfasster Texte (Lane et al., 2019).</i></p><!--syntaktisch: Tokenization / Part-of-Speech(POS)-Tagging-->
    <ol type="1">
       <li>🟡 Merkmalsmodellierung (engl. feature modeling)</li>
              Unter Merkmalsmodellierung versteht man die inhaltliche Strukturierung und Deutung von Text. Sie legt fest, wie Merkmale für die Textanalyse genutzt werden können. Über Themenmodellierung (engl. topic modeling) können Themen unüberwacht (engl. unsuperviced) mittels Merkmalsextraktion (engl. feature extraction) oder Merkmalsumwandlung (engl. feature transformation) identifiziert werden.<br><br>
          <ul>
            <li>Merkmalsextraktion (engl. feature extraction)</li><br>
            Latent Dirichlet Allocation (LDA) identifitiert unüberwacht durch <i>Merkmalsextraktion</i> latente Themen in einer Sammlung von Dokumenten und stellt diese basierend auf ihren Verteilungen über die Themen dar.<br>
              <div style="margin-left: 2em;">
                <code>gensim</code>&nbsp;<code>????</code><br></br>
              </div>
           <li>Merkmalsumwandlung (engl. feature transformation)<br></i></li>
           Latent Semantic Analysis (LSA) identifiziert unüberwacht Themen mittels <i>Merkmalsumwandlung</i> durch eine Singulärwertzerlegung (engl. Singular Value Decomposition - SVD).<br> Die Anzahl der Themen (k) muss dabei optimal gewählt werden, weshalb Techniken wie den Silhouettenkoeffizienten (engl. silhouette score) oder Themenkohärenz (engl. topic coherence) Anwendung finden, um die Drehpunkte für die Themenextraktion zu bestimmen.
              <div style="margin-left: 2em;">
                <code>sklearn</code><br>
              </div><br>
            <li>Merkmalsauswahl (engl. feature selection)<br>
            <i>Zum Ende werden die besten Merkmale ausgewählt.</i>
              <div style="margin-left: 2em;">
                <code>???</code>
              </div><br>
          </ul>
       <li>🟡 Merkmalserkennung (engl. feature recognition)</li>
              Unter Merkmalserkennung 
          <ul>
            <li>regelbasiert</li>
            xxx
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            <li>lernbasiert</li>
            xxx
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
          </ul>
  </details>  
  <details>
    <summary>🟡 Merkmalslernen (engl. feature learning / representation learning)</summary></br>
    In der Datenverarbeitung beginnt die Merkmalsaufbereitung (engl. feature engineering) und das Merkmalslernen (engl. feature learning / representation learning).
    <i>Modellauswertung (engl. model evaluation)<i>
              <ul>
            <li>Kohärenz (engl. coherence)</li>
            xxx
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
            <li>Perplexität (engl. perplexity)</li>
            xxx
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br></br>
              </div>
          </ul>
  </details>
</ol>

###### Pipeline Ausgabe (engl. pipeline output)
Die verarbeiteten Daten fließen in die Datenkonsolidierung ein.
______________

### 🔵 Datenkonsolidierung (engl. data consolidation)
Im Rahmen der Datenkonsolidierung erfolgt die Datennachverarbeitung (engl. data post-processing) in der Merkmalsanalysen (engl. feature analysis) durchgeführt und letztlich als Datenpräsentation (engl. data presentation) aufbereitet werden.

Datenauswertung (engl. data analysis)
Datenpräsentation  (engl. data presentation)
<div style="margin-left: 2em;">
  <code>???</code>&nbsp;<code>???</code><br>
</div>

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

Topic Diversity:
`library 3`
`library 4`

______________


## Installation (engl. setup)
Als Programmiersprsche wird Python in der Version 3.9 genutzt. Als Entwicklungsumgebung (engl. integrated development environment - IDE) wird Visual Studio Code (VSCode) genutzt.

### Vorbereitende Installation (engl. preparatory setup)

###### Laufzeitumgebung (engl. runtime environment)
Als virtuelle Umgebungen stehen in Python "venv" und "conda" zur Verfügung, welche über

```console
`pip install conda`
```
oder 

```console
`pip install venv?`
```
installiert werden können.

###### Datenvalidierung (engl. data validation)
Die Spaltenbeschriftung der textführende Spalte des gewählten Datensatzes muss mit "text" benannt sein damit das Skript den Datensatz durch das externe Modul "AITextDetector.py" verarbeiten kann.
<ul>
<li><ins>KI Detektor</ins></li>
Python in einer Version 3.8+

```python
`pip install transformers`
`pip install torch`
```
Jai Soorya N, K. (2023). AI-Text-Detector-python [Software]. https://github.com/Kishanjaisoorya/AI-Text-Detector-python<br><br>

</ul>

## Abhängigkeiten (engl. dependencies)

###### Module
Ein Python Skript mit der Endung ".py" wird als Modul bezeichnet.[^15]<br>

###### Packages (engl. package)
Eine Sammlung von Modulen in einem Ordner, wird Paket genannt.[^15]<br>

###### Bibliotheken (engl. librarys)
Eine Sammlung von Paketen innerhalb eines größeren Projekts wird Bibliothek genannt. Als Rahmenwerk (engl. framework) werden große, grundlegende Bibliotheken mit vielen aufeinander aufbauenden oder voneinander abhängenden Paketen bezeichnet. Viele Funktionen sind Bestandteil von Bibliotheken und können über: 
```python
<paketname>.<funktionsname>(<funktionsargumente>)
```
aufgerufen werden.[^15]<br>

<ul>
  <li><ins>Python-Standardbibliothek</ins></li><br>
  Dokumentation: https://docs.python.org/3.9/py-modindex.html

  |`stdlib`         | Dokumentation                                                      |Verwendung      |
  |-----------------|--------------------------------------------------------------------|----------------|
  |[`re`]           |https://docs.python.org/3.9/library/re.html#module-re               |NLP             |
  |[`csv`]          |https://docs.python.org/3.9/library/csv.html#module-csv             |Datahandling    |
  |[`venv`]         |https://docs.python.org/3.9/library/venv.html#module-venv           |Laufzeitumgebung|

  <li><ins>externe Bibliothek</ins></li><br>

  | Bibliothek           | Website                                                                                                                                           |Verwendung              |
  |--------------------- |---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
  |`torch`               |Website: https://pypi.org/project/torch/                                  <br>Dokumentation: https://docs.pytorch.org/docs/stable/index.html       |KI-Detektor             |
  |`transformers`        |Website: https://pypi.org/project/transformers/                           <br>Dokumentation:                                                       |KI-Detektor             |
  |`pandas`              |Website: https://pandas.pydata.org                                        <br>Dokumentation: https://pandas.pydata.org/docs/                       |Datahandling            |
  |`spacy`               |Website: https://spacy.io                                                 <br>Dokumentation:                                                       |NLP                     |
  |`nltk`                |Website: https://github.com/nltk                                          <br>Dokumentation: https://www.nltk.org                                  |NLP                     |
  |`pyspellchecker`      |Website: https://pypi.org/project/pyspellchecker/                         <br>Dokumentation: https://pyspellchecker.readthedocs.io/en/latest/      |NLP                     |
  |`gensim`              |Website: https://pypi.org/project/gensim/                                 <br>Dokumentation:                                                       |NLP - Themenmodellierung|
  |`numpy`               |Website: https://numpy.org                                                <br>Dokumentation: https://numpy.org/doc/stable/.                        |                        |
  |`sklearn`             |Website: https://scikit-learn.org/stable/index.html                       <br>Dokumentation:                                                       |                        |
  |`torchmetrics`        |Website: https://pypi.org/project/torchmetrics/                           <br>Dokumentation: https://lightning.ai/docs/torchmetrics/stable/.       |Evaluation              |
  |`matplotlib`          |Website: https://matplotlib.org                                           <br>Dokumentation: https://matplotlib.org/stable/index.html              |Visualisierung          |
  |`seaborn`             |Website: https://seaborn.pydata.org                                       <br>Dokumentation: https://seaborn.pydata.org/tutorial.html              |Visualisierung          |
  |`PyLDAvis`            |Website: https://pypi.org/project/pyLDAvis/                               <br>Dokumentation: https://pyldavis.readthedocs.io/en/latest/            |Visualisierung          |
  |`cartopy`             |Website: https://github.com/SciTools                                      <br>Dokumentation: https://cartopy.readthedocs.io/stable/                |Visualisierung          |
  |`wordcloud`           |Website: https://pypi.org/project/wordcloud/                              <br>Dokumentation: https://amueller.github.io/word_cloud/                |Visualisierung          |
  |`Top2Vec`             |Website: https://pypi.org/project/top2vec/                                <br>Dokumentation: https://top2vec.readthedocs.io/en/stable/Top2Vec.html |Visualisierung          |
  |`stanza `             |Website: https://stanfordnlp.github.io/stanza/                            <br>Dokumentation:                                                       |NLP                     |



</ul>

## Projektstruktur
### Ordnerstruktur
```markdown
├── dataset/   # gewählter Datensatz
├── src/       # Python-Module
├─────────────── / Explorative Datenanalyse (EDA)
├─────────────── / NLP-Pipeline
├─────────────── / Datennachverarbeitung
├── docs/      # Übersichten (Datensatzauswertungen, Pipeline, Installation)
└── README.md
```



## Referenzen

###### Skripte
IU Internationale Hochschule. (2024). Advanced Data Analysis (DLBDSEDA01_D) [Lernskript]. 001-2024-1210.<br>
<br>IU Internationale Hochschule. (2023). Artificial Intelligence (K. Schaaff, Übers.; DLBDSEAIS01_D) [Studienskript]. 001-2023-1213.<br>
<br>IU Internationale Hochschule. (2025). Data Analytics und Big Data (DLBINGDABD01) [Lernskript]. 002-2025-0108.

###### Abschlussarbeiten
Kruse, C. (2022). Vergleichende Evaluation von  Topic-Modellen für die  Analyse von  Softwareinzidenztickets [Masterarbeit, Technische Hochschule Ingolstadt]. https://opus4.kobv.de/opus4-haw/frontdoor/deliver/index/docId/3478/file/I001169705Abschlussarbeit.pdf
Steiner, D., & Zeneli, G. (2019). Texploration: Automatische Analyse von grossen Textsammlungen [Bachelorarbeit, Zürcher Hochschule für Angewandte Wissenschaften]. https://www.zhaw.ch/storage/engineering/institute-zentren/cai/BA19_Texploration_Steiner_Zeneli.pdf<br>

###### Bücher
Lane, H., Howard, C, & Hapke, H. M. (2019). Natural language processing in action: Understanding, analyzing, and generating text with Python. Manning.<br>
<br>Abbott, D., Kommer, I., & Kommer, C. (2025). Datenvisualisierung im praktischen Einsatz: Ansprechende Diagramme und Dashboards gestalte (1. Auflage). dpunkt.verlag.<br>
<br>Alpar, P., Alt, R., Bensberg, F., & Czarnecki, C. (2023). Anwendungsorientierte Wirtschaftsinformatik: Strategische Planung, Entwicklung und Nutzung von Informationssystemen (10. Auflage). Springer Vieweg. https://doi.org/10.1007/978-3-658-40352-2<br>

###### Fachzeitschriften
Blei, David M. and Ng, Andrew Y. and Jordan, Michael I.: Latent dirichlet allocation. In: The Journal of Machine Learning Research. Nr. 3, 3. Januar 2003, S. 993–1022<br>
<br>Blum et al., 2020 ?<br>

###### Websites
Freitas, G. & Lily Hulatt. (2025). Feature Selection: Methoden & Techniken [Bildungsplattform]. StudySmarter. https://www.studysmarter.de/schule/informatik/computerlinguistik-theorie/feature-selection/<br>
<br>Helm, C. (2025, Mai 8). spaCy vs NLTK – Was ist die bessere Wahl für NLP? Konfuzio. https://konfuzio.com/de/spacy-vs-nltk/<br>
<br>Bonart, M., & Förstner, K. (o. J.). Python Pakete und Bibliothekten: Data librarian—Modul 3—Daten analysieren und darstellen. Data Librarian. Abgerufen 11. Oktober 2025, von https://bonartm.github.io/data-librarian/organisation/packages/

###### Anleitungen/Tutorials
`gensim` Řehůřek, R. (2024, August 10). LDA Model. Gensim. https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html<br>
<br>`gensim` Řehůřek, R. (2025). Gensim: Topic modelling for humans. Gemsim. https://radimrehurek.com/gensim/<br>
<br>`gensim` Řehůřek, R. (2024, August 10). LDA Model. Gensim. https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html<br>
<br>Sanchhaya Education Private Ltd. (2025, September 3). NLP Gensim Tutorial—Complete Guide For Beginners [Bildungsplattform]. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/nlp-gensim-tutorial-complete-guide-for-beginners/<br>
<br>Sanchhaya Education Private Ltd. (2025, Juli 23). Normalizing Textual Data with Python [Bildungsplattform]. GeeksforGeeks. https://www.geeksforgeeks.org/python/normalizing-textual-data-with-python/<br>






Formatierung
🟥🟨🟦🟫⬜🟧🟩🟪◼️◻️🔶🔸🔘
:red_square:
<br>GitHub - https://docs.github.com/de/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax<br>

🟣🟢🟠🔴🔵🟡🟤⚫


[^01]: [Datensatz01] (https://www.kaggle.com/datasets/ashwinik/consumer-complaints-financial-products)
[^02]: [Datensatz02] (https://www.kaggle.com/datasets/selener/consumer-complaint-database)
[^03]: [Datensatz03] (https://www.kaggle.com/code/saurabhsawhney/nlp-complaints-classification)
[^04]: [Datensatz04] (https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)
[^05]: [Datensatz05] (https://github.com/sanax-997/nlp_customer_complaints_analysis/blob/main/Data/complaints_data.csv)
[^06]: [Datensatz07] (https://github.com/gurneetjuneja/NLP-Problem-Solving/blob/main/user_complaints.csv)
[^07]: [Datensatz08] (https://www.kaggle.com/code/mchirico/analyzing-text-in-consumer-complaints)
[^08]: [Datensatz09] (https://data.mendeley.com/datasets/w2cp7h53s5/1)
[^09]: [Datensatz10] (https://github.com/Schossi2908/DLBDSEDA02_D)
[^10]: [Datensatz11] (https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
[^11]: [Webseite] (Helm, C. (2025, Mai 8). spaCy vs NLTK – Was ist die bessere Wahl für NLP? Konfuzio. https://konfuzio.com/de/spacy-vs-nltk/)
[^16]: [Webseite] (Timmermann, T. (2019, März 7). Natural Language Processing—Einsteigen und loslegen! [Unternehmenswebseite]. codecentric AG. https://www.codecentric.de/wissens-hub/blog/natural-language-processing-basics)

[^12]: [Buch] Klein, B. (2023). Numerisches Python: Arbeiten mit NumPy, Matplotlib und Pandas (2., aktualisierte u. erweiterte Auflage). Hanser.
[^13]: `seaborn` Waskom, M. (2021). seaborn: Statistical data visualization. Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021<br>
[^14]: Elson, P., Andrade, E. S. de, Lucas, G., May, R., Hattersley, R., Campbell, E., Comer, R., Dawson, A., Little, B., Raynaud, S., scmc72, Snow, A. D., lgolston, Blay, B., Killick, P., lbdreyer, Peglar, P., Wilson, N., Andrew, … Kirkham, D. (2024). SciTools/cartopy: REL: v0.24.1 (Version v0.24.1) [Software]. Zenodo. https://doi.org/10.5281/ZENODO.1182735
[^15]: Bonart, M., & Förstner, K. (o. J.). Python Pakete und Bibliothekten: Data librarian—Modul 3—Daten analysieren und darstellen. Data Librarian. Abgerufen 11. Oktober 2025, von https://bonartm.github.io/data-librarian/organisation/packages/

```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import spacy as sp
import nltk 
import sklearn as
import gensim as
import torch 
import transformers
import pyLDAvis as
import wordcloud as wc 
import top2vec as t2v 
import re
import csv
import venv
```

