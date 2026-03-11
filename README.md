# Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Ziel der Aufgabe ist es NLP-Techniken auf einem organisch entstandenen Datensatz mit Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den Beschwerdetexten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger einer örtlichen Stadtverwaltung aufbereitet werden.<br>

Das schriftliche Konzept hierzu soll die Schritte der NLP-Datenverarbeitung mit Python darlegen. Dabei sollen zwei Techniken zur Vektorisierung der Beschwerdetexte sowie zwei Ansätze zur Extraktion von Themen aus dem Datensatz genannt und die verwendeten Python-Bibliotheken aufgeführt werden.

Dieses Projekt fokussiert sich auf NLU, also die Analyse und Interpretation von Textinhalten.

Konzept Umsetzung

## Installation (engl. setup)
Die Projektumgebung wurde mit Microsoft Visual Studio Code (VSCode) und einer lokalen virtuellen Python-Umgebung (`conda) eingerichtet. Für die Ausführung in VSCode wird folgender Standardweg verwendet:

```console
conda env create -f environment.yml
conda activate Projekt_Advanced-Data-Analysis
```

Anschließend ist in VSCode der Kernel `ada_env (Python)` zu wählen und das Notebook `src/Projekt_Advanced_Data_Analysis.ipynb` auszuführen.

## Projektstruktur
### Ordnerstruktur
```markdown
├── datasets/  # Datensatz
├─────────────── / complaints_data.csv
├─────────────── / complaints_data_cleaned.csv
├── src/       # Python-Skripte
├─────────────── / Projekt_Advanced_Data_Analysis.ipynb
├── docs/      # Arbeitsunterlagen (Datensatzauswertungen, Übersichten)
├─────────────── / 1 - Datenauswertung
├─────────────── / 2 - 
├─────────────── / 3 - 
├─────────────── / X - 
└── README.md
```
_________________________________________________________________________________________________________________________________________________________

## Konzept
Die ausgearbeitete Konzeption lässt sich grob in 3 Phasen einteilen. Datensatzverarbeitung (engl. dataset pipeline), Datenverarbeitung (engl. data processing) und Datennachverarbeitung (engl. data post-processing).

Durch einen Klick auf ► werden Erläuterungen und Unterschritte sichtbar. 

## ⬜ Datensatzverarbeitung (engl. dataset pipeline)
<img src="docs/1 - Datensatzverarbeitung (engl. dataset pipeline).jpg" width="1200">


### Datensatzakquisition (engl. dataset acquisition)
> In der Phase der Datenakquisition wurden Datensätze gesucht und bewertungsbasiert ausgewählt. Hierzu wird ein trichterförmiger, vierstufiger Prozess bestehend aus Datensatzrecherche, –sammlung, –prüfung und –auswahl durchlaufen, um den Korpus für die NLP-Pipeline zu bestimmen.<br>
<ol>
    <details>
      <summary>⬜ Datensatzrecherche (engl. dataset research)</b></summary>
      <i>Es wurde eine Onlinerecherche auf verschiedenen Datenportalen (Kaggle, GitHub, GovData, MendeleyData, u.A.) durchgeführt und nach geeigneten deutschen und englischen Datensätzen gesucht.</i><br>
    </details>
</ol>
<ol>
    <details>
      <summary>⬜ Datensatzsammlung (engl. dataset collection)</summary>
      <i>Offensichtlich synthetisch erzeugte Datensätze wurden ignoriert. Datenquellen mit vermutetem organischen Ursprung wurden im CSV-Datenformat manuell oder per API heruntergeladen und lokal gespeichert.</i><br>
    </details>
</ol>
<ol>    
    <details>
      <summary>⬜ Datensatzprüfung (engl. dataset check)</summary>
      <i>Die gesammelten Datensätze wurden anhand eines extern entwickelten und vortrainierten KI-Detektor (https://github.com/Kishanjaisoorya/AI-Text-Detector-python) auf synthetisch erzeugte Instanzen (engl. samples) geprüft. Das Programm nutzt dabei ein vortrainiertes Modell zur Sentimentanalyse, um Texte hinsichtlich eines vermuteten organischen (menschlichen) oder synthetischen (KI-basierten) Ursprungs durch Label (REAL / FAKE / ERROR) zu klassifizieren. Während der Anwendung zeigte sich, dass nicht alle Datensätze verarbeitet werden konnten. Dies ist plausibel auf lange Beschwerdetexte (>512 Tokens) in Verbindung mit begrenzten Rechenressourcen sowie den Modellgrenzen des Sprachmodells „bert-base-multilingual-uncased“ (110M Parametern) zurückzuführen, welches durch das verwendete Embedding genutzt wird. Die verarbeitbaren Datensätze wurden anhand der Klassifikationslabel (REAL / FAKE / ERROR) über eine Tabellenkalkulationsprogramm ausgewertet.</i><br><br>
      <div style="margin-left: 2em;">
       <code>transformers</code>&nbsp;<code>torch</code><br><br>
    </details>
    <details>
      <summary>⬜ Datensatzauswahl (engl. dataset selection)</summary>
      <i>Durch eine Häufigkeitsauswertung der Label durch ein Tabellenkalkulationsprogramm wurde der Datensatz mit dem prozentual höchsten Anteil an organischen (REAL-Label) Instanzen über die Formel:</i><br>
      <br>

$$\%\text{ organisch} = \left(\frac{REAL}{REAL + FAKE + ERROR}\right) \cdot100$$
<br><i>ermittelt. Die Wahrscheinlichkeit eines organischen Ursprungs erscheint höher, je höher der Prozentsatz als organisch identifizierter Instanzen im Verhältnis zum Gesamtdatensatz ist. Konnte ein Datensatz nicht in angemessener Zeit (30 min.) durch den KI-Detektor verarbeitet werden, wurde die Prüfung abgebrochen und der Datensatz mit n/a bewertet. Diese Datensätze flossen dann nicht in den Ergebnisvergleich ein. Der Datensatz mit der prozentualen höchsten Bewertung wurde ausgewählt.<br><br></i>

| Nr.| Bezeichnung                        | Bewertung | Größe     |Quelle                     |
|----|------------------------------------|-----------|-----------|---------------------------|
| 01 | Consumer_Complaints.csv            | n/a       | 59,40  MB |[^01] &nbsp; Kaggle        |
| 02 | rows.csv                           | n/a       | 176    MB |[^02] &nbsp; Kaggle        |
| 03 | Consumer_Complaints.csv            | 13,50 %   | 107,0  MB |[^03] &nbsp; Kaggle/GovData|
| 04 | complaints_processed.csv           | 64,72 %   | 19,8   MB |[^04] &nbsp; Kaggle        |
| 05 | complaints_data.csv                | 82,00 %   | 7,20   MB |[^05] &nbsp; GitHub        |
| 06 | user_complaints                    | 00,69 %   | 229,0  kB |[^06] &nbsp; GitHub        |
| 07 | consumer_complaints.csv            | n/a       | 175,39 MB |[^07] &nbsp; Kaggle        |
| 08 | Complaints_Reports_Data.sql        | n/a       | 3,28   MB |[^08] &nbsp; MendeleyData  |
| 09 | chatgpt_reviews.csv                | 35,03 %   | 119,9  MB |[^09] &nbsp; GitHub        |
| 10 | dataset-tickets-multi-lang3-4k.csv | n/a       | 6,87   MB |[^10] Kaggle               |

Es wird Datensatz Nr. 05[^05] *"complaints_data.csv"* gewählt da dieser ein Scoring von 82 % erreicht.

  </details>
</ol>

### Datensatzsichtung (engl. dataset inspection)
> In der Phase der Datensatzsichtung wird der ausgewählte Datensatz einer explorativen Datenanalyse (engl. exploratory data analysis - EDA) unterzogen, um strukturelle Merkmale, Muster und potenzielle Qualitätsprobleme systematisch zu identifizieren. Zu diesem Zweck werden sowohl die Datenstruktur als auch der Dateninhalt analysiert, sodass die gewonnene Erkenntnisse in die Datensatzaufbereitung (engl. dataset preparation) überführt und den nachgelagerten Prozessphasen methodisch berücksichtigt werden können.

<ol>   
  <details>
  <summary>⬜ Datenstrukturanalyse (engl. data structure analysis)</summary>
  Bei der Datenstrukturanalyse wird die Strukturierung eines Datensatzes erkundet, um einen Überblick über den Aufbau des Datensatzes zu erhalten.

  <i>Die Datenstruktur des gewählten Datensatzes ist ein Spezialfall einer "Delimiter Separated Value"-Datei welche als Trennzeichen Komma (engl. comma) nutzt (Klein, 2023, p. 261-262).</i>
  <sup id="ref-12"><a href="#fn-12">[12]</a></sup><i> Diese sog. CSV-Datei verfügt im vorliegenden Fall über eine Header und
  5659 Zeilen, welche in 4 Spalten organisiert wurden.</i><br><br>

|author                             |posted_on                 |rating |text              |
|-----------------------------------|--------------------------|-------|------------------|
|`<Benutzername>`of`<US-Ortsangabe>`|`<Monat>`.`<Tag>`,`<Jahr>`|`<0-5>`|`<Beschwerdetext>`|

<i>Die in der Datei enthaltenen Daten lassen sich in <ins>strukturierte Daten</ins> und <ins>unstrukturierte Daten</ins> unterteilen, wobei letztere als Input für die NLP-Pipeline genutzt wird. 

„Strukturierte Daten sind hochgradig organisiert und folgen dabei klar definierten Strukturen.“ (Hebing und Manhembué, 2024, p. 37)
<br></i>
</ol>
  </details>
<ol>   
  <details>
  <summary>⬜ Dateninhaltsanalyse (engl. data content analysis)</summary>
  <i></i><br><br>
  <ul>
    <li><ins>EDA der strukturierten Daten</ins></li>
    In strukturierter Form liegen die Spalten "author", "posted_on" und "rating" vor. Diesen Informationen ist gemein, dass sie ohne größere Vorverarbeitung direkt weiterverarbeitet werden können, da die Informationen meist in einheitlicher (normalisierter Form) vorliegen.<br><br>
    <ul>
        <li><ins> "author"</ins>

  > Die Zeilen der Spalte enthalten jeweils den alphanumerischen`<Benutzernamen>` des Beschwerdeverfassers sowie eine, durch ein "of" getrennte, US-Ortsangabe welche im Format `<Ortsname "of" US-Bundesstaat>` vorliegt. 
  

  Die Datenexploration durch eine Ortsdatenanalyse zeigte, dass 3 US-Ortsangaben ['BC', 'ON', 'PE'] ungültig sind, aus 17 Bundesstaaten eine dreistellige Anzahl an Beschwerden zu verzeichnen ist, von deren auf ['FL: 778', 'CA: 554', 'GA: 414'] entfallen und aus 5 Bundesstaaten ['IA', 'MT', 'OK', 'RI', 'SD'] keine Beschwerden erfasst wurden.<br>
        <li><ins>"posted_on"</ins><br>

  > Die Zeilen der Spalte "posted_on" enthalten Datumsangaben mit alphabetisch abgekürzter Monatsangabe über einen Zeitraum von 16 Jahren im amerikanischem Format `<Monat>`.`<Tag>`,`<Jahr>`. 
  
   Im Rahmen der Datenexploration wurde eine Zeitdatenanlyse durchgeführt welche Muster in der (jährlichen, monatlichen, wöchentlichen) Verteilung der Beschwerden im Datensatz über den Zeitraum vom 31.07.2000 bis 22.11.2016 zeigte.

   <ins>jährliche Verteilung:</ins> Die EDA zeigt, dass die meisten Beschwerden im Jahr 2015 erfolgt sind. 
   
   <ins>monatliche Verteilung:</ins> Die EDA zeigte weiter, dass die meisten Beschwerden im August (540) und die wenigsten Beschwerden im April (369) abgesetzt wurden, wobei der Datensatz ein saisonales Muster zeigt. 
   
   <ins>wöchentliche Verteilung:</ins> Die Verteilung der Beschwerden aufgeschlüsselt nach Wochentagen zeigt, dass die meisten Beschwerden mittwochs (993), dienstags (960) und donnerstags (861), gefolgt von montags (820) und freitags (802) abgesetzt wurden, wohingegen an den Tagen der Wochenenden Samstag (659) und Sonntag (564) weniger Beschwerden zu verzeichnen sind. Dieses Muster deutet ebenfalls auf einen organischen Ursprung des Datensatzes hin.

  <li><ins>"rating"</ins><br>
  
  > Die Zeilen der Spalte "rating" enthalten Bewertungen auf einer Skala `<0-5>`.<br>

  Die Anzahl der Bewertungen nach "rating" ist wie folgt verteilt [rating: 0=1560 (27.57%); 1=3734 (65.98%); 2=260 (4.59%); 3=54 (0.95%); 4=19 (0.34%); 5=32 (0.57%)] was einen Überhang niedriger Bewertungen zeigt. Dies weist ebenfalls auf einen organischen Datensatz hin, da Beschwerden grundsätzlich negativ sind. 

  </ul>
</ul>

<ul>
  <li><ins>EDA der unstrukturierten Daten</ins></li>
  Unstrukturierte Daten sind Informationen, die in einer nicht identifizierbaren Datenstruktur vorliegen. Ein typisches Beispiel dafür sind natürlichsprachliche Texte wie sie in der Spalte "text" vorhanden sind.<br><br>
  <ul>
  <li><ins>"text"</ins></li>
  
  > In den Zeilen der Spalte "text" befindet sich ein englischer `<Beschwerdetext>`. Er besteht aus Wörtern (Zeichenketten, sprich Folgen von Buchstaben, Ziffern, Satzzeichen, etc.) die konkateniert Sätze bilden, die Zeit- und Datumsangaben in unterschiedlichen Formatierungen, Großschreibungen, Aufzählungen und Sonderzeichen enthalten was bei der Sprachverarbeitung zu beachten ist.<br>  
  
  Die EDA zeigte, dass die Texte im Median aus 864 Zeichen bestehen.
  </ul>
</ul>

<ul>
  <li><ins>fehlerhafte Daten</ins></li>
  Im Zuge der Datenexploration ist aufgefallen, dass in den unstrukturierten Daten 30 fehlende Werte (NaNs) in der Spalte 'text' vorliegen, die bereinigt werden müssen, um Verzerrungen in der späteren Modellbildung zu vermeiden. Zudem wurde 1 Duplikat erkannt (bzw. 2 Zeilen in der Paarbetrachtung mit <code>duplicated(keep=False)</code>).<br>
  <ul>
  <li><ins>Fehlwert- und Duplikatbefund</ins></li>
  
  > Die Analyse fehlender Daten zeigte keine Fehlwerte in den strukturierten Spalten 'author', 'posted_on' und 'rating', aber 30 Fehlwerte in der Spalte 'text'. Die Duplikatanalyse zeigte 1 doppelte Zeile.<br>  
  </ul>
</ul>
</ol>
  </details>

### Datensatzaufbereitung (engl. dataset preparation)
> In der Phase der Datensatzbereinigung werden die in der EDA gewonnenen Erkenntnisse genutzt, um den Datensatz für den Anwendungsfall vorzubereiten. Hierzu wird eine Datenbereinigung sowie eine Datenvalidierung durchgeführt, wodurch diejenigen Daten bestimmt werden, die weiter verarbeitet werden.
<ol>
    <details>
      <summary>⬜ Datensatzbereinigung (engl. dataset cleaning)</b></summary>
      <i>
      
###### Fehlwertbehandlung
Die Behandlung von Fehlwerten wie NaNs (Not a Number) oder NaTs (Not a Text) kann durch listenweisen Fallausschluss, durch welchen Zeilen ohne Text oder Text unter einer Mindestlänge entfernt wird oder Imputation, das Auffüllen oder Ersetzen fehlender oder unvollständiger Textelemente durch geschätzte Werte, damit der Datensatz für Modelltraining oder Analyse vollständig nutzbar bleibt.

###### Duplikatentfernung
Durch die Duplikatentfernung werden doppelte Zeilen im Datensatz entfernt, um Verzerrungen des NLP-Modells zu vermeiden. </i><br>
    </details>
</ol>
<ol>
    <details>
      <summary>⬜ Datensatzvalidierung (engl. dataset validation)</b></summary>
      <i>Im Rahmen der Datensatzvalidierung werden fehlerhafte Daten korrigiert, verworfen oder speziell behandelt, um Datenqualität und Aussagekraft zu sichern.</i><br>
    </details>
</ol>

###### Datensatz-Pipeline Ausgabe (engl. dataset pipeline output)
Der so aufbereiteten Datensatzes wird als *"complaints_data_cleaned.csv"* exportiert, welcher als Korpus für die nachfolgenden NLP-Schritte genutzt wird. Die Instanzen des Datensatzes (Zeilen) werden auch als Dokumente bezeichnet. 
_________________________________________________________________________________________________________________________________________________________

## 🟧 Linguistische Datenverarbeitung - LDV (engl. NLP-Pipeline)
###### NLP-Pipeline Eingabe (engl. pipeline input)
Die Sprachverarbeitung beginnt mit dem Import des aufbereiteten Datensatzes *"complaints_data_cleaned.csv"*, genauer dem Import der Spalte `<text>`, welche als Korpus für die folgenden NLP-Schritte genutzt wird. Die Zeilen des Datensatzes werden auch als Dokumente bezeichnet. 

<div style="margin-left: 2em;">
  <code>pandas</code>
</div>

> Natural Language Processing (NLP) ist eine Teildisziplin des maschinellen Lernens (engl. machine learning - ML), die sich mit der algorithmischen Verarbeitung und dem Verstehen natürlichsprachlicher Texte (engl. natural language understandung - NLU) befasst. Im maschinellen Lernen stellen Merkmale (engl. features) kategorielle oder numerische Größen dar, anhand derer die Algorithmen oder neuronale Netze Texte klassifizieren oder clustern können.[^16] Innerhalb von NLU dienen die Merkmale als Brücke zwischen rohem Text und algorithmischer Verarbeitung, welche als Pipeline bezeichnet wird und relevante linguistische Informationen auf lexikalischer, syntaktischer oder semantischer Ebene durch Verarbeitungsschritte extrahiert.

## Upstream-Aufgaben (engl. upstream tasks)
> Als Upstream tasks werden in NLP-Pipelines werden die frühen, vorbereitenden Verarbeitungsschritte, die Daten generieren oder vorverarbeiten, um spätere Komponenten zu ermöglichen bezeichnet.

### 🟥 Datenvorverarbeitung (engl. data pre-processing)
> Während der Datenvorverarbeitung erfolgt die *Merkmalsvorbereitung (engl. feature preparation)* für nachfolgende Schritte in einem mehrstufigen Prozess, welcher sich grob in Textvorverarbeitung und linguistische Vorverarbeitung unterteilen lässt. 
<div style="margin-left: 2em;">
  <code>spaCy</code>
</div>


<ol type="1">
  <details>
    <summary>🟥 <b>textuelle Vorverarbeitung</b> (engl. text pre-processing)</summary>
    <p><i>In der textuelle Vorverarbeitung erfolgt eine Textbereinigung (engl. text cleaning) welche durch Rauschentfernung (engl. noise reduction) und Standardisierung (engl. standardisation) anhand eines Filterblocks aus Standard- und Individualfiltern realisiert wird, um sprachspezifische Stopp- und Kurzworte, Satzzeichen, Nummern, Zeitangaben, Emoijs, Schreibfehler und Pronomen zu filtern.</i></p>
    <img src="docs/2 - Textbereinigung (engl. text cleaning).jpg">
    <ol type="1">
        <details>
          <summary>🔴 Rauschentfernung (engl. noise reduction)</summary>
          <p><i>Ziel der Rauschentfernung ist es irrelevante Token (Zeichen und Zeichenketten) für nachfolgende Prozesse zu identifizieren und zu löschen.</i></p>
          <ul>
          <details>
            <summary>🔴 Wortbereinigung (engl. word cleaning)</b></summary>
            <p><i>Wortfilter</i></p>
              <ul>
              <li><ins>Stoppworte</ins></li><br>
              Stoppworte werden beim Token-Filtering entfernt.
             <div style="margin-left: 2em;">
                <code>spaCy (token.is_stop)</code><br><br>
              </div>
              </ul>
              <ul>
              <li><ins>Pronomenfilter (projektspezifisch)</ins><br>
              XXXX
             <div style="margin-left: 2em;">
                <code>spaCy (token.pos_ != "PRON")</code><br><br>
              </div>
              </ul>
              <ul>
              <li><ins>Weitere projektspezifische Wortfilter</ins><br>
              Einzelne domänenspezifische Tokens werden ausgeschlossen.
             <div style="margin-left: 2em;">
                <code></code>&nbsp;<code>spaCy (token.text)</code><br><br>
              </div>
              </ul>
          </details>
          <details>
          <summary>🔴 Zeichenbereinigung (engl. character cleaning)</summary>
          <p><i>Satzzeichen werden beim Token-Filtering entfernt.</i></p>
              <div style="margin-left: 2em;">
                <code>spaCy (token.is_punct)</code><br><br>
              </div>
            <ol type="1">
              <details>
                <summary>🔴 Satzzeichen (engl. word embeddings)</summary>
                <p><i>XXXXX</i></p>
                <ul>
                <li><ins>xxxx</ins></li>
                nxxx
              </details>
            </ol>
    </ul>
      </li>
        </details>
        <details>
          <summary>🔴 Standardisierung (engl. standardisation)</summary>
          <p><i>Durch Standardisierung werden relevante Token vereinheitlicht. Hierdurch wird vermieden, dass gleiche Inhalte in mehreren leicht unterschiedlichen Varianten auftreten.</i></p>
            <ins>🔴 Normalisierung (engl. normalisation)</ins><br>
              Die Texte werden in ein konsistentes Token-Format überführt.
              <ul>
                <li><ins>🔴 Kasusumwandlung (engl. case conversion)</ins><br>
                  Tokens werden in Kleinschreibung überführt.
                  <div style="margin-left: 2em;">
                    <code>spaCy (token.lemma_)</code>&nbsp;<code>stdlib(.lower)</code><br><br>
                  </div>
                </li>
                <li><ins>🔴 Lemmatisierung (engl. lemmatization)</ins><br>
                  Tokens werden auf ihre Grundform reduziert.
                  <div style="margin-left: 2em;">
                    <code>spaCy (token.lemma_)</code><br><br>
                  </div>
                </li>
                <li><ins>🔴 Batch-Verarbeitung der Dokumente</ins><br>
                  Die Verarbeitung erfolgt effizient über Dokument-Batches.
                  <div style="margin-left: 2em;">
                    <code>spaCy (nlp.pipe)</code><br><br>
                  </div>
                </li>
              </ul>
            </li>
              <ins>🔴 Rechtschreibfehlerkorrektur (engl. spelling correction)</ins><br>
              Im Text-Cleaning der Beschwerden nicht umgesetzt.
                <div style="margin-left: 2em;">
                <code>nicht umgesetzt im Notebook</code><br><br>
              </div>
            </li>
          </ul>
        </details>
</ol>
</ol>


<ol type="1">
  <details>
    <summary>🟥 <b> linguistische Vorverarbeitung</b> (engl. linguistic processing)</summary>
    Im Rahmen der linguistischen Analyse erfolgt, je nach Anwendungsfall neben einer lexikalischen, eine syntaktische und/oder semantische Verarbeitung von bereinigten Texten zur Merkmalsvorbereitung.
<ul>
  <details>
    <summary>🔴 lexikalische Verarbeitung (engl. lexical processing)</summary>
    <p><i>Im Rahmen der lexikalischen Analyse werden Texte tokenisiert, Token auf ihre Grundform reduziert und ein Vokabular aufgebaut.</i></p>
    <ul style="list-style: none; padding-left: 0;">
    <ol type="1">
  <details>
    <summary>🔴 Tokenisierung (engl. tokenization)</summary>
        <i>Die Beschwerden werden mit spaCy tokenisiert und im Batch verarbeitet.</i><br>
        <div style="margin-left: 2em;">
          <code>spaCy (nlp.pipe)</code><br><br>
        </div>
  </details>
  <details>
    <summary>🔴 Grundformreduktion (engl. inflection reduction)</summary>
        <i>Die Tokens werden lemmatisiert und in Kleinschreibung überführt.</i><br>
        <div style="margin-left: 2em;">
          <code>spaCy (token.lemma_)</code><br><br>
        </div>
  </details>
  <details>
    <summary>🔴 Vokabularerstellung/Wortschatzaufbau (engl. vocabulary construction)</summary>
        <i>Aus den bereinigten Texten wird ein Vokabular mit Token-IDs aufgebaut.</i><br>
        <div style="margin-left: 2em;">
          <code>sklearn (CountVectorizer)</code><br><br>
        </div>
  </details>
  <details>
    <summary>🔴 lexikalisches POS-Tagging (für Filterregeln)</summary>
        <i>POS-Informationen werden für projektspezifische Filter (z. B. Pronomenfilter) genutzt.</i><br>
        <div style="margin-left: 2em;">
          <code>spaCy (token.pos_)</code><br><br>
        </div>
  </details>
  </details>
  <details>
    <summary>🔴 syntaktische Verarbeitung (engl. syntactic processing)</summary>
    <p><i>Im Notebook nicht als eigener Verarbeitungsschritt umgesetzt.</i></p>
    <div style="margin-left: 2em;">
      <code>nicht umgesetzt</code><br><br>
    </div>
  </details>
  <details>
    <summary>🔴 semantische Verarbeitung (engl. semantic processing)</summary>
    <p><i>Im Notebook nicht als eigener spaCy-NLP-Schritt (z. B. NER/Coreference/Relation Extraction) umgesetzt.</i></p>
    <ul>
      <details>
        <summary>🔴 Eigennamenerkennung (engl. Named Entity Recognition - NER)</summary>
        <div style="margin-left: 2em;">
          <code>nicht umgesetzt</code><br><br>
        </div>
      </details>
      <details>
        <summary>🔴 Koreferenzauflösung (engl. Coreference Resolution - CR)</summary>
        <div style="margin-left: 2em;">
          <code>nicht umgesetzt</code><br><br>
        </div>
      </details>
      <details>
        <summary>🔴 Beziehungsextraktion (engl. Relationship Extraction - RE)</summary>
        <div style="margin-left: 2em;">
          <code>nicht umgesetzt</code><br><br>
        </div>
      </details>
    </ul>
  </details>
</ul>
</ol>

### 🟨 Datenvorbereitung (engl. data preparation)
> Im Rahmen der Datenvorverarbeitung werden Merkmale (engl. features) erzeugt und ausgewählt, was durch *Merkmalsgenerierung (engl. feature generation/featurization)* und *Merkmalsauswahl (engl. feature selection)* erfolgt.<br> 
<ol type="1">
  <details>
    <summary>🟨 <b> Merkmalsgenerierung </b> (engl. feature generation/featurization)</summary>
   Merkmalsgenerierung bezeichnet den Prozess, aus rohem oder vorverarbeitetem Text neue, informative Merkmale zu erzeugen, die Machine-Learning-Modelle effizient nutzen können. Sie wandelt ustrukturierte Daten (Texte) durch Merkmalskodierung (engl. feature encoding)' in numerische oder kategorische Repräsentationen um, welche syntaktische, semantische oder kontextuelle Aspekte einfangen. Dabei werden Attribute/Features in eine für die Modellierung adäquate Form überführt, weshalb von Merkmalsaufbereitung (engl. feature engineering) gesprochen wird (Baars und Kemper, 2021, p. 159). Dies kann mittels Merkmalskonstruktion, Merkmalsextraktion oder Merkmalsumwandlung erfolgen oder automatisch über trainierte Modelle vorgenommen werden. In diesem Fall spricht man von Merkmalslernen (engl. feature learning / representation learning), wobei Merkmale direkt aus Rohtexten gewonnen werden.

   - Feature Engineering
   „Der Prozess, bei dem Domänenwissen über die Daten genutzt wird, um Merkmale zu erstellen, die maschinelle Lernalgorithmen funktionsfähig machen, wird als Feature Engineering bezeichnet (übersetzt: Shekhar, 2018).“ (iu., DLMDSEDE01 -  2022, p. 132)

   - Feauture Learning
<ol type="1">
  <details>
    <summary>🟨 Vektorisierung (engl. vectorization)</summary>
    <p><i>Als Vektorisierung wird die Merkmalskodierung (engl. feature encoding) von Textdaten bezeichnet. Die Token (Wörter, Subwörter oder Zeichen) aus dem Vokabular werden durch Vektorisierungstechniken in numerische Repräsentationen überführt, die als Merkmalsvektoren in einem n‑dimensionalen Merkmalsraum (engl. feature space) dargestellt und zu Merkmalsmatrizen zusammengefasst werden. Vektorisierungstechniken nutzen Merkmalsextraktion, um Texte je nach Anwendungsfall auf Silben,- Wort-, Satz-, Segment‑ oder Dokumenten‑Ebene für Modelle aufzubereiten, um lexikalische, syntaktische oder kontextuelle Aspekte eines Textes einzufangen.</i></p>
<div style="margin-left: 2em;">
  <code>sklearn (CountVectorizer, TfidfVectorizer)</code>&nbsp;<code>sentence-transformers</code><br><br>
</div>
<ol type="1">
        <details>
          <summary>🟡 Merkmalsvektoren (engl. feature vectors)</summary>
          <p><i>Spannen keinen semantischen Merkmalsraum auf, sondern erzeugen dünn besetzte Vektoren (engl. sparse vectors) auf Basis von Tokenfrequenzen, was Modellen eine algebraische bzw. statistische Auswertung ermöglicht. Teils werden die Merkmalsvektoren auch als unsemantische oder häufigkeitsbasierte Embeddings (engl. frequency based embeddings) bezeichnet. Diese frequenzbasierten Methoden erzeugen dünn besetzte Merkmalsvektoren basierend auf Vokabularpositionen, wobei zwischen Methoden mit und ohne Informationsgewichtung diffrenziert wird.</i></p>
          <ul>
          <details>
            <summary>🟡 TF-IDF (term frequency times inverse document frequency)</b></summary>
            <p><i>Bei der TF-IDF-Methode handelt es sich um eine statistische Erweiterung von BoX, durch welche eine Informationsgewichtung der Token bzw. Tokensequenzen vorgenommen wird.</i></p>
              <ul>
              <li><ins>TF-IDF auf Einzeltoken</ins><br>
              Wird die TF-IDF-Methode auf Wortebene durchgeführt, werden Einzelwörter gewichtet, um ihre Relevanz im Dokument und im Korpus auszudrücken.
              <div style="margin-left: 2em;">
                <code>sklearn (TfidfVectorizer)</code>
              </ul>
          </details>
</ol> 
      <ol type="1">
        <details>
          <summary>🟡 Merkmalseinbettungen (engl. feature embeddings)</summary>
          <p><i>Spannen einen semantischen Merkmalsraum auf und liefern dichtbesetzte Vektoren (engl. dense vectors), was Modellen eine Auswertung von semantischen Ähnlichkeiten über Abstände bzw. Ähnlichkeitsmaße ermöglicht. Einbettungen (engl. embeddings) weisen jedem Merkmal einen dichten Vektor im semantischen Raum zu und erfassen so statisch oder dynamisch die Bedeutungsdimension mittels vorhersage- oder kontextbasierter Verfahren anhand vortrainierter Modelle auf Wort-, Satz-, Segment‑ oder Dokumenten‑Ebene. Hierdurch werden semi-explizite Merkmale generiert, welche zwischen expliziten und latenten Merkmalen einzuordnen sind. Kontextmodelle: 
          <p><i>Worteinbettungen sind dichte Vektoren, die semantische Bedeutungen von Wörtern oder Sätzen repräsentieren. Sie entstehen durch das Training von Modellen auf Textdaten und erfassen semantische und syntaktische Beziehungen.</i></p>
          </i></p>
          <ol type="1">
              <details>
                <summary>🟡 <b>Worteinbettungen </b>(engl. word embeddings)</summary>
                <p><i>Worteinbettungen weisen jedem Wort einen dichten Vektor im semantischen Raum zu und ermöglichen hierdurch Modellen eine Auswertung von semantischen Ähnlichkeiten zwischen Wörtern.</i></p>
                <ul>
                <li><ins>nichtkontextuelle / vorhersagebasierte Wort-Einbettungen (engl. prediction based word embeddings)</ins></li>
                nichtkontextuelle sprich vorhersagebasierte Wort-Einbettungen sind statische Einbettungen, die jedes Wort zu einem festen Vektor übersetzen – unabhängig vom Kontext, in dem es steht. Ihr Training erfolgt auf riesigen Textkorpora, dabei lernen die Modelle, Wörter mit ähnlichen Kontexten auch im Vektorraum zusammenzubringen. Sprachliche Vieldeutigkeiten (Polysemie) und Kontextänderungen werden dabei jedoch nicht abgebildet (Papp et al., 2022, p. 329; Wehner, 2026).
              <ul>
                <li>GloVe (Global Vectors for Word Representation)</li>
                  <div style="margin-left: 2em;">
                    <code>gensim</code>&nbsp;<code>glove-python</code><br><br>
                  </div>
                <li>Word2Vec (Skip-gram, CBOW)</li>
                  <div style="margin-left: 2em;">
                    <code>gensim</code><br><br>
                  </div>
                <li>FastText
                  <div style="margin-left: 2em;">
                    <code>gensim</code><br><br>
                  </div>
                </li>
              </ul>
                <li><ins>kontextbasierte Wort-Einbettungen (engl. contextualized word embeddings)</ins></li>
                Bei kontextbasierte Wort-Einbettungen handelt es sich um dynamische Worteinbettungen bei denen jedes „Wort, jede Phrase, jeder Satz [...] situativ angepasste Embedding-Vektoren – in Abhängigkeit vom umgebenden Kontext [bekommen]" (Wehner, 2026). Semantische und syntaktische Unterschiede können so erstmals maschinell berücksichtigt werden.<br>
                <ul>
                <li>bidirektionale kontextbasierte Wort-Einbettungen</li>
                Diese Embeddings werden durch große Sprachmodelle erzeugt, die Kontext bidirektional nutzen. ---- Bidirektionale Kontextmodelle:</b> ELMo, BERT
                <ul>
                <li>BERT (Bidirectional Encoder Representations from Transformers)
                  <div style="margin-left: 2em;">
                    <code>sentence-transformers</code>&nbsp;<code>transformers</code><br><br>
                  </div>
                </li>
                <li>ELMo (Embeddings from Language Models)</li>
                bi-directional LSTM Network.
                  <div style="margin-left: 2em;">
                    <code>tensorflow</code>&nbsp;<code>tensorflow_hub</code><br><br>
                  </div>
                </li>
                </ul>
                <li>unidirektionale kontextbasierte Wort-Einbettungen</li>
                Diese Embeddings werden durch generative Sprachmodelle erzeugt, die Kontext von links nach rechts (unidirektional) nutzen.
                <ul>
                <li>GPT (Generative Pre-trained Transformer)</li>
                  <div style="margin-left: 2em;">
                    <code>transformers</code>&nbsp;<code>sentence-transformers</code><br><br>
                  </div>
                </li>
                </ul>
                </ul>
              </details>
              <details>
                <summary>🟡 <b>Satzeinbettungen </b>(engl. sentence embeddings)</summary>
                <p><i>Satzeinbettungen weisen jedem Satz einen dichten Vektor im semantischen Raum zu und ermöglichen hierdurch Modellen eine Auswertung von semantischen Ähnlichkeiten zwischen Sätzen.</i></p>
                <ul>
                <li><ins>vorhersagebasierte Satzeinbettungen (engl. prediction based sentence embeddings)</ins></li>
                <i>vorhersagebasierte Satzeinbettungen sind statische Einbettungen auf Satzebene und nutzen Encoder-Decoder-Architekturen oder ähnliche Verfahren, um Sätze in feste Vektoren zu übersetzen.</i>
                <ul>
                <li>SkipThought Embeddings
                Als vorhersagebasierte Satzeinbettung mit Encoder-Decoder-Architektur verarbeiten SkipThought Embeddings die Eingabesequenzen sequenziell. Das ursprüngliche SkipThought-Modell basiert typischerweise auf RNNs/LSTMs, die Sequenzen Token für Token(Wort für Wort) verarbeiten. Bei der klassischen SkipThought-Architektur mit RNN/LSTM-Encoder-Decoder verarbeitet der Encoder die Eingabesequenz in einer Richtung (von links nach rechts). Das Modell nutzt dabei nur Informationen aus den vorangegangenen Tokens, um zukünftige Tokens vorherzusagen.
                  <div style="margin-left: 2em;">
                    <code>nicht umgesetzt im Notebook</code><br><br>
                  </div>
                </li>
                </ul>
                <li><ins>kontextbasierte Satzeinbettungen (engl. contextualized sentence embeddings)</ins></li>
                <i>kontextbasierte Satzeinbettungen werden durch Transformer-basierte oder RNN-basierte Modelle erzeugt und erfassen Satzebenen-Semantik.</i>
                <ul>
                <li><ins>Unidirektionale Satzeinbettungen (unidirektional)</ins></li>
                <i>Grundsätzlich wären auch unidirektionale Kontextmodelle auf Satzebene möglich – beispielsweise indem man GPT-basierte Modelle für Satzrepräsentationen nutzt. Solche Modelle würden den Kontext von links nach rechts verarbeiten und Sätze als ganze Einheiten einbetten können.</i></li>
                <li><ins>Bidirektionale Kontextmodelle (bidirektional)</ins></li>
                <ul>
                  <li>SBERT (Sentence-BERT)
                    <div style="margin-left: 2em;">
                      <code>sentence-transformers</code>&nbsp;<code>transformers</code><br><br>
                    </div>
                  </li>
                  <li>USE Embedding (Universal Sentence Encoder)
                    <div style="margin-left: 2em;">
                      <code>tensorflow-hub</code><br><br>
                    </div>
                  </li>
                </ul>
                </li>
                </ul>
              </details>
            </ol>
          </ol>
    </ul>
      </li>
        </details>
    </ol>
  </details>
  </details>
</ol>

<ol type="1">
  <details>
    <summary>🟨 <b> Merkmalsauswahl </b> (engl. feature selection)</summary>
   Merkmalsauswahl ist ein zur Merkmalsgenerierung komplementärer Prozess, der aus einer großen Menge von erzeugten Merkmalen die relevantesten auswählt. Dies reduziert Dimensionalität, verbessert Modellperformance und verringert Rechenaufwand, indem irrelevante oder redundante Merkmale entfernt werden.
</ol>

## Downstream-Aufgaben (engl. downstream tasks)
> Als Downstream tasks werden in NLP-Pipelines spezifische, anwendungsorientierte Aufgaben, die auf den Ergebnissen grundlegender (upstream) Verarbeitungsschritte aufbauen, bezeichnet. Sie stellen die "Endprodukte" dar, bei denen Modelle an reale Probleme angepasst werden, um direkt nutzbaren Output zu liefern.

### 🟧 Datenverarbeitung (engl. data processing)
> Im Rahmen der Datenverarbeitung erfolgt eine *Modellierung (engl. modeling)*, sprich die Entwicklung mathematischer oder statistischer Funktionen (z.B. Machine-Learning-Modelle), die Muster in Daten erfassen und nutzbar machen. Man spricht in diesem Zusammenhang auch vom *Merkmalsmodellierung (engl. feature modeling)*. Sie bezeichnet den Prozess der systematischen Konstruktion und Optimierung von Merkmalen in einem formalisierten mathematischen, sprich deterministischen oder probabilistischen Rahmen. Ziel ist es, rohe, unstrukturierte oder hochdimensionale Daten in eine kompakte, diskriminative oder interpretierbare numerische Repräsentation zu überführen, welche die wesentlichen Informationen bewahrt oder verstärkt. Dabei werden werden ML-Algorithmen auf die vorverarbeiteten Daten angewendet, die eingesetzten Kernverfahren umfassen *überwachtes Lernen (engl. supervised learning)* – Regression, Klassifikation oder *unüberwachtes Lernen (engl. unsupervised learning)* – Clustering. Als Algorithmus wird das Lernverfahren bezeichnet, als Modell eine parametrisierte Instanz des Algorithmus.
<ol type="1">
  <details>
    <summary>🟧<b> Modellentwicklung</b> (engl. model development)</summary>
    
  > Ein Modell ist also ein Lösungsverfahren (mathematische Funktion) das die Eingabedaten (engl. input-features) in Ausgabedaten (engl. output-features – Vorhersagen / Klassifikationen / Cluster) transformiert. Als Modellentwicklung wird der Prozess zur Erstellung eines für eine Aufgabe geeigneten Modells bezeichnet und umfasst die Modellauswahl und Modellbildung. 
  
  <ol type="1">
  <details>
    <summary>🟠<b> Modellauswahl</b> (engl. model selection)</summary>
    Die Modellauswahl definiert die algorithmische und strukturelle Grundlage des ML-Lösungsverfahrens. Hierin wird ein geeigneter Algorithmus, sprich eine geeignete Modellarchitektur festgelegt. Modellarchitekturen können dabei nicht-neuronale (z.B. algebraische, lineare, probabilistische) oder neuronale Strukturen (z.B. Transformer, LSTM) sein und müssen zur Aufgabe passen.

###### Kernverfahren
<ol type="1">
  <details>
      <summary>🟠 Regression (engl. regression)</summary>

> Regression ist ein überwachtes Lernverfahren, das eine Funktion $f: X \rightarrow \mathbb{R}^k$ schätzt, um aus Merkmalsvektoren kontinuierliche Zielgrößen vorherzusagen, indem ein geeigneter Verlust zwischen beobachteten und modellierten Werten minimiert wird.
  </details>
  <details>
      <summary>🟠 Klassifikation (engl. classification)</summary>
      
> Klassifikation ist ein überwachtes Lernverfahren, bei dem Eingaben diskreten Klassen zugeordnet werden; formal wird eine Entscheidungsfunktion $f: X \rightarrow {1,\dots,C}$ oder eine Klassenwahrscheinlichkeitsverteilung $P(Y\mid X)$ gelernt.
  </details>
  <details>
      <summary>🟠 Gruppenbildung (engl. clustering)</summary>

> Clustering ist ein unüberwachtes Lernverfahren zur Strukturentdeckung in ungelabelten Daten, bei dem Objekte anhand eines Ähnlichkeits- oder Distanzmaßes so in Gruppen (Cluster) partitioniert werden, dass die Intra-Cluster-Ähnlichkeit hoch und die Inter-Cluster-Ähnlichkeit gering ist.

<ul>
  <details>
    <summary>🟠 Themenmodellierung (engl. topic modeling)</summary>

Themenmodellierung ist ein unüberwachtes Lernverfahren zur Identifikation latenter semantischer Strukturen in Textsammlungen. Durch Themenmodelle (engl. topic models) werden aus bestehenden Merkmalen automatisch neue, interpretierbare Themendimensionen extrahiert oder transformiert, die nicht explizit im Text vorhanden sind. Die methodische Differenzierung erfolgt nach dem verwendeten Modellierungsverfahren: Abstraktionsbasierte Ansätze extrahieren Themen durch probabilistische oder semantische Prozesse, während projektionsbasierte Ansätze algebraische Matrixfaktorisierungsverfahren nutzen.

> **Merkmalsabstraktion (engl. feature abstraction):** Abstraktionsbasierte Modelle nutzen probabilistische oder Embedding-basierte Verfahren, um neue semantisch interpretierbare Konzepte direkt aus Daten zu extrahieren. Sie modellieren Themen durch komplexe statistische oder neuronale Prozesse und erzeugen Features, die nicht nur mathematische Transformationen, sondern bedeutungstragende Repräsentationen darstellen.

> **Merkmalsprojektion (engl. feature projection):** Projektionsbasierte Modelle nutzen algebraische Matrixfaktorisierungstechniken, um die Merkmalsmatrix in Faktoren zu zerlegen. Obwohl sie deterministische mathematische Transformationen verwenden, erzeugen sie dennoch interpretierbare latente Konzepte, die als Themen fungieren. Diese algebraischen Verfahren operieren typischerweise auf frequenzbasierten Repräsentationen.

 <ol type="1">
    <details>
      <summary>🟠 Wortraum Themenmodelle (engl. word-space topic models)</summary>
      <p><i>Wortraumbasierte Modelle operieren auf frequenzbasierten Eingabe-Repräsentation (z.B. BoW, TF-IDF). Sie nutzen einen diskreten Wort-Merkmals-Raum. Jedes Wort ist eine Dimension, die vom Modell über einen Merkmalsvektor (engl. feature vector) aufgerufen wird, da diese meist hochdimensional sind werden sie auch als dünnbesetzte Vektoren (engl. sparse vektors) bezeichnet (siehe oben).</i></p>
    <ul>
    <details>
        <summary>🟠 Merkmalsabstraktion (engl. feature abstraction)</summary>
        <ul>
        <li><ins>LDA (Latent Dirichlet Allocation)</li></ins>
        <p><i>Latent Dirichlet Allocation (LDA) ist ein probabilistisches Modell, das latente Themen aus der Merkmalsmatrix durch wahrscheinlichkeitsbasierte Themen-Wort-Verteilungen identifiziert. LDA erzeugt interpretierbare Themen mit probabilistischen Zuordnungen zu Dokumenten und Wörtern.</i></p>
          <div style="margin-left: 2em;">
            <code>gensim</code>&nbsp;<code>sklearn (LatentDirichletAllocation)</code><br><br>
          </div>
          <p><b>Output:</b> Themenmischung pro Dokument (α), Wort-Gewichte pro Thema (β), K latente Themen</p>
      </ul>
    </details>
    <details>
        <summary>🟠 Merkmalsprojektion (engl. feature projection)</summary>
    <ul>
    <li><ins>NMF (Non-Negative Matrix Factorization)</ins></li>
    <p><i>Non-Negative Matrix Factorization (NMF) ist ein algebraisches Verfahren, das die Merkmalsmatrix in zwei Faktormatrizen mit nicht-negativen Werten zerlegt. Im Gegensatz zu probabilistischen Modellen wie LDA erzeugt NMF deterministische Topic-Zuordnungen, die direkt aus der Matrixfaktorisierung hervorgehen.</i></p>  
        <div style="margin-left: 2em;">
          <code>sklearn (NMF)</code><br><br>
        </div>
        <p><b>Output:</b> Topic-Gewichte pro Dokument, Wort-Gewichte pro Topic, K Themen</p>
  </ul>
  <ul>
  <li><ins>LSA (Latent Semantic Analysis)</ins></li>
  <p><i>Latent Semantic Analysis (LSA) nutzt Singulärwertzerlegung (SVD), um latente semantische Dimensionen aus der Merkmalsmatrix zu extrahieren. LSA ist ein algebraisches Verfahren der Matrixfaktorisierung, das effektiv und effizient interpretierbare Themen für Topic Modeling erzeugt.</i></p>
    <div style="margin-left: 2em;">
      <code>sklearn (TruncatedSVD)</code><br><br>
    </div>
  <p><b>Output:</b> k latente Dimensionen, Singular Values, LSA-Komponenten</p>
</ul>
</details>
    </ul>
    </details>
    <details>
      <summary>🟠 Kontextraum Themenmodelle (engl. context-space topic models)</summary>
      <p><i>Kontextraumbasierte Modelle operieren auf semantischen Merkmalseinbettungen (engl. feature embeddings - z.B. BERT-Embeddings), die einen kontinuierlichen semantischen Raum bilden. Jedes Wort ist als dichter Vektor repräsentiert, siehe oben.</i></p>
    <ul>
    <details>
        <summary>🟠 Merkmalsabstraktion (engl. feature abstraction)</summary>
        <ul>
        <li><ins>BERTopic</li></ins>
        <p><i>BERTopic ist eine moderne Erweiterung klassischer Topic-Modeling-Methoden, die vortrainierte BERT-Embeddings mit Dimensionsreduktion (UMAP) und Clustering (HDBSCAN) kombiniert. Sie erzeugt interpretierbare und semantisch kohärente Themen direkt aus Embeddings, ohne dass eine separate Merkmalsmatrix nötig ist, und ist besonders effektiv bei großen Textsammlungen.</i></p>
        <div style="margin-left: 2em;">
          <code>bertopic</code>&nbsp;<code>sentence-transformers</code>&nbsp;<code>umap-learn</code><br><br>
        </div>
        <p><b>Output:</b> Topic-Label pro Dokument, Wort-Gewichte pro Topic, Cluster-Visualisierung</p>
        </ul>
    </details>
    </ul>
    </details>
</ol>
  </details>
</ul>
</ol>

###### Unterstützungsverfahren
<ol type="1">
  <details>
    <summary>⚫ Dimensionsreduktion (engl. dimensionality reduction)</summary>
    <p><i>Transformiert hochdimensionale Features auf neue mathematische Achsen für Visualisierung und Datenanalyse. Die neuen Dimensionen sind nicht semantisch interpretierbar, aber nützlich zur Strukturerkennung.</i></p>
    <ol type="1">
      <details>
        <summary>⚫ Lineare Projektionen (engl. linear projections)</summary>
        <p><i>Lineare Projektionen reduzieren Dimensionen durch orthogonale Transformationen, die Varianzrichtungen im Datenraum erfassen.</i></p>
        <ol type="1">
          <details>
            <summary>⚫ PCA (Principal Component Analysis)</summary>
          <p><i>Findet Hauptkomponenten (Richtungen maximaler Varianz) und projiziert Features darauf.</i></p>
          <div style="margin-left: 2em;">
            <code>sklearn (PCA)</code><br>
            <b>Output:</b> K Komponenten, Varianzanteil pro Komponente
          </div>
        </details>
      </details>
      <details>
        <summary>⚫ Nichtlineare Projektionen (engl. non-linear projections)</summary>
        <p><i>Nichtlineare Projektionen bewahren lokale oder globale Strukturen in den Daten besser, sind aber rechnerisch aufwendiger. Sie werden hauptsächlich für Visualisierung verwendet.
        </i></p>
        <ol type="1">
          <details>
            <summary>⚫ t-SNE (t-distributed Stochastic Neighbor Embedding)</summary>
            <p><i>Projiziert auf 2-3 Dimensionen, bewahrt lokale Nachbarschaften. Ideal für Cluster-Visualisierung.</i></p>
            T-SNE (t-distributed Stochastic Neighbor Embedding) „Die t-SNE- Projektion (Mitte) zeigt eine Verbesserung mit gut voneinander getrennten Clustern. Jede Farbe (die für eine andere Person steht) bildet eine eigene, kompakte Gruppe mit klaren Grenzen zwischen den verschiedenen Personen. Schau mal, wie t-SNE fast perfekte lokale Gruppierungen macht, bei denen Gesichter derselben Person ganz nah beieinander liegen und von anderen Gruppen weggeschoben werden. Das ist die Stärke von t-SNE: Es ist super darin, lokale Nachbarschaften zu erhalten und visuell unterschiedliche Cluster zu erstellen.“ (Thevapalan, 2025)
            nichtlineare probabilistische Technik zur Dimensionalitätsreduzierung „Eigene Embedding Spaces erstellen & visualisieren
            <div style="margin-left: 2em;">
              <code>sklearn (TSNE)</code><br>
              <b>Output:</b> 2-3D Koordinaten
            </div>
          </details>
          <details>
            <summary>⚫ UMAP (Uniform Manifold Approximation and Projection)</summary>
            <p><i>Moderne Alternative zu t-SNE, schneller und skalierbarer. Bewahrt lokale und globale Strukturen.</i></p>
            UMAP (Uniform Manifold Approximation and Projection) „Natürliche Sprachverarbeitung: Textdaten können, wenn sie in hochdimensionale Einbettungen umgewandelt werden, mit UMAP visualisiert werden, um semantische Beziehungen zu verstehen. Es wird oft benutzt, um Wort-Embeddings und Dokument-Cluster zu zeigen und Sprachmodelle zu debuggen, indem es zeigt, wie verschiedene Konzepte im Embedding-Raum miteinander zusammenhängen“ (Thevapalan, 2025).
            <div style="margin-left: 2em;">
              <code>umap-learn</code><br>
              <b>Output:</b> 2-3D Koordinaten
            </div>
            Output: Visualisierbar, aber nicht semantisch interpretierbar
          </details>
        </ol>
      </details>
    </ol>
  </details>
</ol>
  </details>
  <details>
    <summary>🟠<b> Modellbildung</b> (engl. model building)</summary>
     Modellbildung ist der Prozess, bei dem Modellarchitekturen durch Konfiguration, Initialisierung und Training optimiert werden, um optimale Features zu lernen. Dazu werden nicht-adaptive Konfigurationseinstellungen, sog. Hyperparameter und adaptive Einstellungen die Modellparameter bestimmt.
<ul>
  <details>
    <summary>🟠<b> Modellkonfiguration </b> (engl. model configuration)</summary>

  > Bei der Konfiguration werden Hyperparameter eines Modells festgelegt. Hierbei handelt es sich um nicht-adaptive Einstellungen eines Modells welche außerhalb liegen. Sie bestimmen das Trainingsverhalten und beeinflussen die Qualität der gelernten Features. Ihre Optimierung erfolgt typischerweise durch iterative Verfahren, Hyperparameter werden vor dem Training durch Abstimmung festgelegt (IBM Deutschland GmbH, 2025). Einige Hyperparameter bestimmen das Verhalten des Modells während des Trainings (z.B. Lernrate beim Gradientenabstieg oder die Anzahl der Epochen des Trainingsprozesses). Andere Hyperparameter sind für die Form und Struktur des Modells verantwortlich, wie z. B. Anzahl der Cluster im k-means Clustering oder der versteckten Schichten in einem neuronalen Netz (IBM Deutschland GmbH, 2025).
  
  </details>
  <details>
    <summary>🟠 <b> Modellinitialisierung </b> (engl. model initialization)</summary>

> In der Modellinitialisierung werden Modellparameter, modellinterne, adaptive Einstellungen für den Lernprozess des Modells mit Startwerten versehen. Je nach Initialisierungsstrategie können diese, mit zufälligen oder heuristisch begründeten Startwerten versehen werden.
  </details>
  <details>
    <summary>🟠 <b> Modelltraining</b> (engl. model training)</summary>

> Beim Modelltraining werden die Modellparameter vom Modell direkt oder über mehrere Iterationen des Lernprozesses als Reaktion auf die Trainingsdaten aktualisiert. Das Modell aktualisiert die Parameterwerte welche steuern, wie das Modell auf ungesehene Daten reagiert. Es handelt sich also um die gelernten Werte (Gewichtungen) innerhalb des maschinellen Lernmodells, die bestimmen, wie es Eingabedaten auf Ausgaben, wie z. B. eine vorhergesagte Klassifizierung oder ein Clusterung abbildet (IBM Deutschland GmbH, 2025). Die Anpassung der Parameter erfolgt bis zur Konvergenz oder zum Erreichen einer maximalen Anzahl von Iterationen. Je nachdem, ob und wie das Modell Eingaberepräsentationen verarbeitet, lassen sich zwei Hauptkategorien innerhalb des Modelltraining unterscheiden:<br>

<ul>
  <details>
    <summary>🟠 mit Merkmalsnutzung (engl. feature utilization)</summary>

> Bei der Merkmalsnutzung wird das Modelltraining mit fester Eingaberepräsentation durchgeführt. Die Eingaberepräsentationen bleiben während des gesamten Trainings unverändert, nur modellspezifische Parameter werden angepasst. Im Normalfall werden die Eingaberepräsentationen durch feste Kodierungsverfahren wie BoW oder TF-IDF direkt aus dem Projektkorpus erzeugt. Eine Sonderform liegt vor, wenn die festen Eingaberepräsentationen aus vortrainierten, im aktuellen Lernprozess eingefrorenen Encodern stammen und deren erzeugte Embeddings lediglich weiterverwendet werden.

> Innerhalb der Merkmalsnutzung kann nochmals zwischen zwei Untertypen unterschieden werden:

- *ohne Merkmalsprojektion:* Das Modell lernt Verteilungen oder Zuordnungen direkt im originalen Merkmalsraum, ohne diesen algebraisch zu transformieren. Der Merkmalsraum selbst bleibt unverändert. (z. B. LDA; BERTopic bei Nutzung fester SBERT-Embeddings als Eingaberepräsentation)<br><br>

> Im vorliegenden Anwendungsfall wurde SBERT als vortrainierter und eingefrorener Encoder zur Erzeugung fester Satz-Embeddings genutzt.

- *mit Merkmalsprojektion:* Das Modell projiziert die festen Eingabemerkmale algebraisch in einen neuen latenten Raum. Die Eingaberepräsentationen bleiben fest, der Merkmalsraum wird jedoch durch Matrixfaktorisierung transformiert. (z. B. NMF, LSA/TruncatedSVD)<br><br>
</details>
<details>
    <summary>🟠 mit Merkmalslernen (engl. feature learning)</summary>

> Bei Merkmalslernen wird das Modelltraining mit gelernten Repräsentationen durchgeführt. Daher spricht man in diesem Zusammenhang auch von Repräsentationslernen (engl. representation learning). Merkmalslernen tritt in überwachten, unüberwachten und selbstüberwachten Settings auf. Die Eingaberepräsentationen werden während des Trainings datengetrieben mitoptimiert, sodass Repräsentationsparameter und Modellparameter gemeinsam angepasst werden. Die gelernten Repräsentationen sind dabei stets relativ zur Zielfunktion, den Daten und den Evaluationsmetriken zu bewerten. (z. B. BERT Fine-Tuning)<br>
</details>
  </details>
</ul>
  </details>
  </ol>
  </details>
  <details>
    <summary>🟧<b> Modellabstimmung</b> (engl. model calibration)</summary>

> Bei der Modellabstimmung werden Algorithmus-Hyperparameter durch manuelle Suche (engl. search) oder systematische Suchverfahren wie Rastersuche (engl. grid search), Zufallssuche (engl. random search) oder bayesianische Optimierung (engl. Bayesian optimization) – iterativ optimiert, um die Modellperformance hinsichtlich einer definierten Zielfunktion (Metrik - siehe unten) zu maximieren, ohne dabei auf die Trainingsdaten zurückzugreifen, was als Hyperparameteroptimierung (engl. hyperparameter optimization – HPO) bezeichnet wird.<br>
    
  <ul>
    <details>
    <summary>🟠 Manuelle Suche (engl. search)</summary>
    
  > XXXXX
  </details>
  <details>
    <summary>🟠 Rastersuche (engl. grid search)</summary>
    Im vorliegenden Anwendungsfall wurde Rastersuche (engl. grid search) genutzt.
  </details>
  <details>
    <summary>🟠 Zufallssuche (engl. random search)</summary>
  </details>
  <details>
    <summary>🟠 Bayesianische Optimierung</b> (engl. bayesian optimization)</summary>
  </details>
  </ul>
  </details>
  <details>
    <summary>🟧<b> Modellbereitstellung</b> (engl. model deployment)</summary>
    Modellbereitstellung bezeichnet die technische Verfügbarmachung eines trainierten Modells, sodass es in einer Zielumgebung zuverlässig für Inferenz auf neuen Daten genutzt werden kann.
  </details>
  <details>
    <summary>🟧<b> Modellanwendung</b> (engl. model inference)</summary>
    Anwendung des trainierten Modells auf ungesehene Eingabedaten zur Erzeugung von Modellausgaben, die anschließend in die Datennachverarbeitung überführt werden.
    <br><ins>Modellausgaben (engl. output features)</ins>:
    <ul>
            <ul>
            <li>🟠 Rohwerte (engl. logits)<br>
            <li>🟠 Bewertungen (engl. scores)<br>
            <li>🟠 Klassenbezeichnungen (engl. labels)<br>
            <li>🟠 Themenzuordnungen (engl. topic assignments)<br>Clusterzuordnungen
            Als Themenzuordnungen werden die Zuordnung eines Dokuments (oder Satzes) zu einem Thema/Topic bezeichnet. Dabei kann zwischen Einzelzuordnungen (engl. single-assignment) und Mehrfachzuordnung (engl. multi-assignment) sowie harten und weichen (probalistischen) Zuordnungen diffenziert werden.
            </ul>
    Modell verarbeitet neue Daten und erzeugt rohe Ausgaben.
  </details>
</ol>

Das Ergebnis ist eine trainierte mathematische Funktion (das Modell), die spezifische NLP-Aufgaben wie eine Klassifikation (z.B. in der Sentiment Analysis) oder ein Clustering (z.B. im Topic Modeling) erfüllt, indem sie die Eingabedaten (Features) in Ausgabedaten (Klassifikationen/Cluster/Vorhersagen ect.) transformiert.

###### Pipeline Ausgabe (engl. pipeline output)

Die durch das finale Modell verarbeiteten Daten fließen in Form von Scores, Labels oder Logits die Datennachverarbeitung (engl. data post-processing) ein.

Input oder Outputfeatures in die Datennachvereitung ein
______________
### 🟦 Datennachverarbeitung (engl. data post-processing)
> Datennachverarbeitung (engl. post-processing) erfolgt nach der Modellausführung (Inference), um rohe Modellausgaben nutzbar zu machen.

<ol type="1">
  <details>
    <summary>🟦<b> Datenauswertung</b>  (engl. data evaluation)</summary>
    <p><i>Die Datenauswertung setzt sich aus der Datenzusammenfassung und der Datenanalyse zusammen.
    Merkmalsanalyse - Merkmalsauswertungen (engl. feature analysen) / Modellanalysen - Modellauswertung. 
    </i></p>
    <img src="docs/3 - Datenauswertung (engl. data evaluation).jpg" width="1200">
    <ol type="1">
      <details>
        <summary>🔵 Datenzusammenfassung (engl. data aggregation)</summary>
        <p><i>Die Datenzusammenfassung ist der Datenanalyse vorgelagert und fasst Daten zu kompakteren Zusammenfassungen zusammen. Dabei werden Modelleingaben (engl. input features), modellinterne Merkmale (engl. intermediate features) oder Modellausgaben (engl. output features) ein- oder mehrdimensional zusammengefasst. Die Aggregation kann dabei modellintern oder modellextern anhand unterschiedlicher Aggregationsstrategien erfolgen, wodurch die Datenmenge durch mathematische Operationen oder Gruppierungen (z. B. nach Token-Typ, Dokument oder Zeitraum) reduziert wird. Sie erfolgt vor der Visualisierung, um Überladung zu vermeiden, und ist rein datenverarbeitend ohne grafische Elemente.</i></p>
        <ul>
        <details> 
        <summary>Eingaben-Aggregation (engl. input-aggregation)</summary>
        </details>
        <details> 
        <summary>Modellinterne-Aggregation (engl. Intermediate aggregation)</summary>
        </details>
        <details> 
        <summary>Ausgaben-Aggregation (engl. output-aggregation)</summary>
      </details>
    </details>
      <details>
        <summary>🔵 Datenanalyse (engl. data analysis)</summary>
        <p><i>In der Datenanalyse werden Erkenntnisse über Merkmale und/oder die merkmalsverarbeitenden Modelle gewonnen.</i></p>
        <ul>
        <details> 
        <summary>🔵 Merkmalsanalyse (engl. feature analysis)</summary>
        <i>Merkmalsanalyse ist der analytische Prozess, bei dem bereits erstellte, ausgewählte oder gelernte Merkmale untersucht, beschrieben und interpretiert werden. Dies erfolgt durch Merkmalserkennung, um spezifische Muster und Strukturen in den Daten zu identifizieren.</i>
        -Der analytische Prozess, bei dem bereits erstellte oder vorhandene Merkmale untersucht, beschrieben und bewertet werden.
        Abhängig vom ML-Aufgabentyp erfolgt eine Modellbewertungen entweder anhand intrinsische oder extrinsische Metriken. Extrinsische Metriken werden bei überwachten Lernaufgaben intrinsische Metriken bei unüberwachten Lernaufgaben verwendet.
        „Intrinsic and extrinsic evaluators are distinct measures where intrinsic evaluators capture inherent properties and extrinsic evaluators assess performance in external contexts.“ (“Intrinsic and Extrinsic Evaluators”)</i>
      </details>
      <details>
        <summary>🔵 Modellanalyse (engl. model analysis)</summary>
        <p><i>Eine Modellanalyse erfolgt modellspezifisch und untersucht, wie ein Modell die Eingabemerkmale transformiert hat. Sie bewertet Modelle, weshalb sie auch als Modellbewertung (engl. model evaluation) bezeichnet wird und besteht aus der Modellqualitätsmessung und der Modellvalidierung.</i></p>
        <ul>
        <details>
          <summary>🔵 Modellqualitätsmessung (engl. model quality assessment)</summary>
          <p><i>Durch die Modellqualitätsmessung erfolgt eine Modellbewertung (engl. model scoring) durch Metriken. Dabei wird zwischen intrinsischen und extrinsischen Metriken unterschieden.</i></p>
          <ul>
          <details>
        <summary>🔵 Intrinsische Metriken (engl. intrinsic metrics)</summary>
          <p><i>Bewerten die Qualität gelernter Features basierend auf innerer Struktur, ohne externe Referenzen zu benötigen.</i></p>

> XXXXX Topicmodeling: Coherence Score / Topic diversity              
  <ol type="1">
            <details>
              <summary>🔘 Kohärenz (engl. coherence)</summary>
                <p><i>Misst semantische Konsistenz der Top-Wörter pro Thema. Ein höherer Wert deutet auf kohärente, interpretierbare Themen hin.</i></p>
                  <ul>
                    <li><b>u_mass</b>: Interne Kohärenz</li>
                    <li><b>c_v</b>: Externe Konsistenz (wird genutzt)</li>
                    <li><b>c_uci, c_npmi</b>: Alternative Berechnungsvarianten</li>
                  </ul>
              <div style="margin-left: 2em;">
                <code>gensim.models.CoherenceModel</code><br>
              <b>Bereich:</b> -1 bis 1 (höher = besser)
              </div>
              </details>
              <details>
                <summary>🔘 Themenvielfalt (engl. topic diversity)</summary>
                <p><i>Misst, inwieweit sich die Top-Wörter verschiedener Themen unterscheiden – verhindert redundante Themen.</i></p>
                <b>Bereich:</b> X bis X (X = X)
              </details>
              <details>
                <summary>🟡 Verwirrung (engl. perplexity)</summary>
                <p><i>Misst die durchschnittliche Vorhersageunsicherheit des Modells auf ungesehenen Daten. Niedrigere Werte deuten auf bessere Generalisierung hin.</i></p>
                <div style="margin-left: 2em;">
                  <code>gensim</code>&nbsp;<code>torchmetrics</code><br>
                <b>Bereich:</b> 0 bis ∞ (niedriger = besser)
                </div>
              </details>
      </details>
  <details>
    <summary>🔘 Extrinsische Metriken (engl. extrinsic metrics)</summary>
    <p><i>Bewerten Modellleistung durch Vergleich mit bekannten Labels in Downstream-Tasks (z.B. Klassifikation, Named Entity Recognition, ect.).</i></p>
    <ol type="1">
      <details>
        <summary>🟡 Klassifikationsmetriken</summary>
        <p><i>Bewerten die Modellleistung bei überwachten Aufgaben durch Vergleich von Vorhersagen mit bekannten Labels.</i></p>

  > „Bei der Entwicklung und dem Finetuning von Algorithmen müssen Metriken vorhanden sein, um zu beurteilen, wie gut ein Algorithmus im Vergleich zu anderen Systemen funktioniert. Bei einer binären Klassifizierungsaufgabe werden üblicherweise Accuracy (Genauigkeit), Precision (Relevanz), Recall (Trefferquote) und der F-Score für diesen Zweck verwendet.“ (iu., DLBDSEAIS01-01_D - 2025, p. 41)
  <ul>
          <li><b>Accuracy</b>: Anteil korrekt klassifizierter Instanzen</li>
          <li><b>Precision</b>: Anteil relevanter unter den als positiv klassifizierten Instanzen</li>
          <li><b>Recall</b>: Anteil erkannter relevanter Instanzen</li>
          <li><b>F1-Score</b>: Harmonisches Mittel aus Precision und Recall</li>
        </ul>
        <div style="margin-left: 2em;">
          <code>sklearn.metrics</code><br>
        </div>
      </details>
      <details>
        <summary>🔵 Clustering-Metriken</summary>
        <p><i>Bewerten Clustering-Ergebnisse durch Vergleich mit bekannten Referenzlabels. Relevant für Topic Modeling, wenn gelabelte Daten zur Validierung vorliegen.</i></p>
        <ul>
          <li><b>ARI (Adjusted Rand Index)</b>: Paarweise Übereinstimmung, zufallsbereinigt</li>
          <li><b>NMI (Normalized Mutual Information)</b>: Informationstheoretische Übereinstimmung</li>
          <li><b>V-Measure</b>: Harmonisches Mittel aus Homogeneity und Completeness</li>
          <li><b>(engl. homogeneity)</b>: XXX</li>
          <li><b>(engl. completeness)</b>: XXX</li>
          <li><b>(engl. purity)</b>: Anteil dominanter Labels pro Cluster</li>

  </ul>
        <div style="margin-left: 2em;">
          <code>sklearn.metrics (adjusted_rand_score, normalized_mutual_info_score, v_measure_score)</code><br>
        </div>
      </details>
      <details>
        <summary>🔵 Benchmark-Suiten (für Sprachmodelle)</summary>
        <p><i>Standardisierte Evaluierungsrahmen, die mehrere NLU-Aufgaben bündeln, um Sprachmodelle vergleichbar zu bewerten.</i></p>
        <ul>
          <li><b>GLUE</b>: General Language Understanding Evaluation – 9 Aufgaben (z.B. Sentiment, Textähnlichkeit, Inferenz)</li>
          <li><b>SuperGLUE</b>: Erweiterung von GLUE mit anspruchsvolleren Aufgaben (z.B. kausales Schließen, Wortsinn-Disambiguierung)</li>
        </ul>
      </details>
    </ol>
  </ul>
</details>
<details>
  <summary>🔵 Modellvalidierung (engl. model validation)</summary>
  <p><i>Bei der Modellvalidierung wird überprüft, ob das Modell zuverlässig und generalisierbar ist – also ob die gemessene Qualität belastbar ist. Hierbei kann zwischen quantitativer und qualitativer Validierung unterschieden werden.</i></p>
  <div style="margin-left: 2em;">
    <code>XXX</code>&nbsp;<code>XXXX</code>&nbsp;<code>XXXX</code><br><br>
  </div>
</details>
</ul>
</details>
    </ol>
  </details>
  <details>
    <summary>🟦<b> Datenkommunikation</b> (engl. data communication)</summary>
    <p><i>Durch die Datenkommunikation werden gewonnene Erkenntnisse aus der Datenanalyse textuell und/oder grafisch präsentiert, weshalb dieser Schritt auch als Datenpräsentation (engl. data presentation) oder Visualisierung (engl. visualization) bezeichnet wird.
    Visualisierung stellt aggregierte Daten grafisch dar, um Muster erkennbar zu machen.</i></p>
    <div style="margin-left: 2em;">
      <code>plotly</code>&nbsp;<code>matplotlib</code><br><br>
    </div> 
    <img src="docs/3 - Datenkommunikation (engl. data communication).jpg" width="1200">
    <ol type="1">
      <details>
        <summary>🔵 Merkmalsdarstellungen (engl. feature visualizations)</summary>
        <p><i>Die Darstellung von Merkmalen kann über textuelle, grafische oder hybride Darstellungsformen erfolgen.
          - Themenverteilungen; Top-Wörter pro Thema
          - BERTopic-Integrierte Visualisierung</i></p>
        <div style="margin-left: 2em;">
          <code>wordcloud</code>&nbsp;<code>BERTopic</code>&nbsp;<code>print() / display / </code><br><br>
        </div>
      </details>
      <details>
      <summary>🔵 Modelldarstellung (engl. model visualizations)</summary>
      <p><i>Bei der Modelldarstellung werden Parameter und Ergebnissen in textueller und visueller Form vorgenommen, um Funktionsweise, Qualität und Grenzen des Modells nachvollziehbar zu machen. Die Darstellung von Metriken erfolgt dabei tabellarisch und grafisch, um Modellgüte, Vergleichbarkeit und Optimierungspotenziale transparent zu bewerten.</i></p>
      </details>
    </ol>
  </details>
</ol>
</ul>

### 🟩 Datenverständnis (engl. data understanding)

> Das Datenverständnis bildet den letzten Schritt, in welchem die gewonnenen Analyseergebnisse domänenspezifisch interpretiert, eingeordnet und in belastbare Handlungsempfehlungen überführt werden.
______________
## Abhängigkeiten (engl. dependencies)
Ein Python Skript mit der Endung ".py" wird als Modul bezeichnet. Eine Sammlung von Modulen in einem Ordner, wird Paket (engl. package) genannt. Eine Sammlung von Paketen innerhalb eines größeren Projekts wird Bibliothek (engl. librarys) genannt. [^15]<br>

  | Bibliothek              | Website                                                                                                                                              |Verwendung              |
  |-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
  |[`torch`]                |Website: https://pypi.org/project/torch/                                  <br>Dokumentation: https://docs.pytorch.org/docs/stable/index.html          |KI-Detektor             |
  |[`transformers`]         |Website: https://pypi.org/project/transformers/                           <br>Dokumentation:                                                          |KI-Detektor             |
  |[`pandas`]               |Website: https://pandas.pydata.org                                        <br>Dokumentation: https://pandas.pydata.org/docs/                          |Datahandling            |
  |[`numpy`]                |Website: https://numpy.org                                                <br>Dokumentation: https://numpy.org/doc/stable/.                           |Datahandling            |
  |[`spacy`]                |Website: https://spacy.io                                                 <br>Dokumentation: https://spacy.io/api/doc/                                |NLP                     |
  |[`sentence-transformers`]|Website: https://huggingface.co/sentence-transformers                     <br>Dokumentation: https://www.sbert.net/index.html                         |NLP - Vektorisierung    |
  |[`sklearn`]              |Website: https://scikit-learn.org/stable/index.html                       <br>Dokumentation: https://scikit-learn.org/stable/user_guide.html          |NLP - Vektorisierung    |
  |[`gensim`]               |Website: https://pypi.org/project/gensim/                                 <br>Dokumentation: https://radimrehurek.com/gensim/apiref.html#api-reference|NLP - Themenmodellierung|
  |[`bertopic`]             |Website: https://maartengr.github.io/BERTopic/index.html                  <br>Dokumentation: https://maartengr.github.io/BERTopic/index.html#common   |NLP - Themenmodellierung|
  |[`matplotlib`]           |Website: https://matplotlib.org                                           <br>Dokumentation: https://matplotlib.org/stable/index.html                 |Visualisierung          |
  |[`seaborn`]              |Website: https://seaborn.pydata.org                                       <br>Dokumentation: https://seaborn.pydata.org/tutorial.html                 |Visualisierung          |
  |[`wordcloud`]            |Website: https://pypi.org/project/wordcloud/                              <br>Dokumentation: https://amueller.github.io/word_cloud/                   |Visualisierung          |
  |[`plotly`]               |Website: https://plotly.com/python/                                       <br>Dokumentation: https://docs.plotly.com                                  |Visualisierung          |
  |[`ipython`]              |Website: https://ipython.org                                              <br>Dokumentation: https://ipython.readthedocs.io/en/stable/index.html      |Visualisierung          |
  |[`scipy`]                |Website: https://scipy.org                                                <br>Dokumentation: https://docs.scipy.org/doc/scipy/                        |Visualisierung          |
  |[`stdlib - re`]          |Website: https://www.python.org                                           <br>Dokumentation: https://docs.python.org/3.9/library/re.html#module-re    |Datahandling            |
  |[`stdlib - csv`]         |Website: https://www.python.org                                           <br>Dokumentation: https://docs.python.org/3.9/library/csv.html#module-csv  |Datahandling            |

Als Rahmenwerk (engl. framework) werden große, grundlegende Bibliotheken mit vielen aufeinander aufbauenden oder voneinander abhängenden Paketen bezeichnet. Viele Funktionen sind Bestandteil von Bibliotheken und können über: 

```python
<paketname>.<funktionsname>(<funktionsargumente>)
```

aufgerufen werden.[^15]<br>

## Referenzen

###### Software
Jai Soorya N, K. (2023). AI-Text-Detector-python [Software]. https://github.com/Kishanjaisoorya/AI-Text-Detector-python<br>

###### Skripte
IU Internationale Hochschule. (2024). Advanced Data Analysis (DLBDSEDA01_D) [Lernskript]. 001-2024-1210.<br>
<br>IU Internationale Hochschule. (2023). Artificial Intelligence (K. Schaaff, Übers.; DLBDSEAIS01_D) [Studienskript]. 001-2023-1213.<br>
<br>IU Internationale Hochschule. (2025). Artificial Intelligence (K. Schaaff, Übers.) [Lernskript]. 001-2025-0805 (DLBDSEAIS01-01_D).<br>
<br>IU Internationale Hochschule. (2025). Data Analytics und Big Data (DLBINGDABD01) [Lernskript]. 002-2025-0108.

###### Abschlussarbeiten
Kruse, C. (2022). Vergleichende Evaluation von  Topic-Modellen für die  Analyse von  Softwareinzidenztickets [Masterarbeit, Technische Hochschule Ingolstadt]. https://opus4.kobv.de/opus4-haw/frontdoor/deliver/index/docId/3478/file/I001169705Abschlussarbeit.pdf<br>
<br>Steiner, D., & Zeneli, G. (2019). Texploration: Automatische Analyse von grossen Textsammlungen [Bachelorarbeit, Zürcher Hochschule für Angewandte Wissenschaften]. https://www.zhaw.ch/storage/engineering/institute-zentren/cai/BA19_Texploration_Steiner_Zeneli.pdf<br>

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
<br>Bonart, M., & Förstner, K. (o. J.). Python Pakete und Bibliotheken: Data librarian—Modul 3—Daten analysieren und darstellen. Data Librarian. Abgerufen 11. Oktober 2025, von https://bonartm.github.io/data-librarian/organisation/packages/
Choi, J. (2023). Choijin/NLP_Topic_Modeling [Jupyter Notebook]. https://github.com/choijin/NLP_Topic_Modeling (Ursprünglich erschienen 2023)

###### Anleitungen/Tutorials
`gensim` Řehůřek, R. (2024, August 10). LDA Model. Gensim. https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html<br>
<br>`gensim` Řehůřek, R. (2025). Gensim: Topic modelling for humans. Gemsim. https://radimrehurek.com/gensim/<br>
<br>`gensim` Řehůřek, R. (2024, August 10). LDA Model. Gensim. https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html<br>
<br>Sanchhaya Education Private Ltd. (2025, September 3). NLP Gensim Tutorial—Complete Guide For Beginners [Bildungsplattform]. GeeksforGeeks. https://www.geeksforgeeks.org/nlp/nlp-gensim-tutorial-complete-guide-for-beginners/<br>
<br>Sanchhaya Education Private Ltd. (2025, Juli 23). Normalizing Textual Data with Python [Bildungsplattform]. GeeksforGeeks. https://www.geeksforgeeks.org/python/normalizing-textual-data-with-python/<br>
Plotly Technologies Inc. (2026). Choropleth. Plotly. https://plotly.com/python/choropleth-maps/



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


Formatierung
🟥🟨🟦🟫⬜🟧🟩🟪◼️◻️🔶🔸🔘
:red_square:
<br>GitHub - https://docs.github.com/de/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax<br>
⚪
🟣🟢🟠🔴🔵🟡🟤⚫