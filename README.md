# Aufgabe 1.1: NLP-Techniken anwenden, um eine Textsammlung zu analyieren
Die linguistische Datenverarbeitung (LDV, engl. natural language processing – NLP) lässt sich in drei zentrale Bereiche unterteilen: die automatische Spracherkennung (engl. automatic speech recognition – ASR), das Verständnis natürlicher Sprache (engl. natural language understanding – NLU) sowie die natürliche Sprachgenerierung (engl. natural language generation – NLG). Dieses Projekt fokussiert sich auf NLU, also die Analyse und Interpretation von Textinhalten.

Ziel der Aufgabe ist es NLP-Techniken auf einem organisch entstandenen Datensatz mit Beschwerden anzuwenden und so die am häufigsten angesprochenen Themen aus den Beschwerdetexten zu extrahieren. Die hierdurch gewonnenen Informationen sollen im Anschluss für Entscheidungsträger (einer örtlichen Stadtverwaltung) aufbereitet werden.<br>

Das schriftliche Konzept hierzu soll die Schritte der NLP-Datenverarbeitung mit Python darlegen. Dabei sollen zwei Techniken zur Vektorisierung der Beschwerdetexte sowie zwei Ansätze zur Extraktion von Themen aus dem Datensatz genannt und die verwendeten Python-Bibliotheken kurz aufgeführt werden.

Durch einen Klick auf ► werden Erläuterungen und Unterschritte sichtbar. Die verwendeten Softwarebibliotheken werden in folgender Form hervorgehoben: 

<div style="margin-left: 2em;">
  <code>Bibliothek 1</code>&nbsp;<code>Bibliotek 2</code><br>
</div>

Bibiliotheken - Funktionen
<paketname>.<funktionsname>(<funktionsargumente>)

- Clustering von Beschwerden.

„Die Hyperparameter eines Modells liegen außerhalb des Modells und werden vor dem Training durch die Abstimmung der Hyperparameter festgelegt. Einige Hyperparameter bestimmen das Verhalten des Modells während des Trainings“ (IBM Deutschland GmbH, 2025)

## Konzeption
Die ausgearbeitete Konzeption lässt sich grob in 3 Phasen einteilen. Datenvorverarbeitung (engl. data pipeline), Sprachdatenverarbeitung (engl. NLP-Pipeline) sowie die Datennachverarbeitung (engl. data post-processing).

## ⚪ Datensatzverarbeitung (engl. dataset pipeline)
<img src="1 - Datensatzverarbeitung (engl. dataset pipeline).jpg" width="1200">

### Datensatzakquisition (engl. dataset acquisition)
In der Phase der Datenaquisition werden Datensätze gesucht und bewertungsbasiert ausgewählt. Hierzu wird ein trichterförmiger, vier stufiger Prozess bestehend aus Datensatzrecherche, –sammlung, –prüfung und –auswahl durchlaufen, um dem Korpus für die NLP-Pipeline zu bestimmen.<br>
<ol>
    <details>
      <summary>⚪ Datensatzrecherche (engl. dataset research)</b></summary>
      <i>Es wird eine Onlinerecherche auf verschiedenen Datenportalen (Kaggle, GitHub, GovData, MendeleyData, u.A.) durchgeführt und nach geeigneten deutschen und englischen Datensätzen gesucht.</i><br>
    </details>
</ol>
<ol>
    <details>
      <summary>⚪ Datensatzsammlung (engl. dataset collection)</summary>
      <i>Offensichtlich synthetisch erzeugte Datensätze werden ignoriert. Datenquellen mit vermutetem organischen Ursprung werden im CSV-Datenformat manuell oder per API heruntergeladen und lokal gespeichert.</i><br>
    </details>
</ol>
<ol>    
    <details>
      <summary>⚪ Datensatzprüfung (engl. dataset check)</summary>
      <i>Die gesammelten Datensätze werden anhand deines vortrainierten, extern entwickelten [KI-Detektors](https://github.com/Kishanjaisoorya/AI-Text-Detector-python) auf synthetisch erzeugte Instanzen (engl. samples) geprüft und mit Labels (REAL / FAKE / ERROR) getaggt. [genutze Technik - BERT ? Beschränkung der Zeichen] BERT-base
      </i><br><br>
      <div style="margin-left: 2em;">
       <code>transformers</code>&nbsp;<code>torch</code><br><br>
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

Es wird Datensatz Nr. 05[^05] *"complaints_data.csv"* gewählt da dieser ein Scoring von 82 % erreicht.

  </details>
</ol>

### Datensatzsichtung (engl. dataset inspection)
In der Phase der Datensatzsichtung wird eine Explorative Datenanalyse (engl. exploratory data analysis) durchgeführt, um Muster, Qualitätsprobleme und Strukturen des Datensatzes zu erkennen, damit diese zur Datensatzaufbereitung (engl. dataset preparation) und in den anschließenden Phasen berücksichtigt werden können. Die Datensatzsichtung wurde mittels des selbstgeschrieben Python-Skripts "Datensatzsichtung (engl. dataset inspection).ipynb" durchgeführt, um den gewählten Datensatz zur eine Datentrukturanalyse, sowie der Analyse der strukutierten und unstrukturierten besser zu verstehen.

<ol>   
  <details>
  <summary>⚪ Datenstrukturanalyse (engl. data structure analysis)</summary>
  <i>Die Datenstruktur des gewählten Datensatzes ist ein Spezialfall einer
  "Delimiter Separated Value"-Datei welche als Trennzeichen Komma (engl. comma)
  nutzt (Klein, 2023, p. 261-262).</i>
  <sup id="ref-12"><a href="#fn-12">[12]</a></sup><i> Diese sog. CSV-Datei verfügt im vorliegenden Fall über eine Header und
  5659 Zeilen, welche in 4 Spalten organisiert wurden.</i><br><br>

|author                             |posted_on                 |rating |text              |
|-----------------------------------|--------------------------|-------|------------------|
|`<Benutzername>`of`<US-Ortsangabe>`|`<Monat>`.`<Tag>`,`<Jahr>`|`<0-5>`|`<Beschwerdetext>`|

<i>Die in der Datei enthaltenen Daten lassen sich in <ins>struktierte Daten</ins> und <ins>unstrukturierte Daten</ins> unterteilen, wobei letztere als Input für die NLP-Pipeline genutzt wird.<br></i>
</ol>
  </details>
<ol>   
  <details>
  <summary>⚪ Explorative Datenanalyse (engl. exploratory data analysis)</summary>
  <i>In der EDA werden Textdaten untersucht, um Muster, Qualitätsprobleme und Strukturen zu erkennen.</i><br><br>
  <ul>
    <li><ins>EDA der struktierte Daten</ins></li>
    In strukturierter Form liegen die Spalten "author", "posted_on" und "rating" vor. Diesen Informationen ist gemein, dass sie ohne größere Vorverarbeitung direkt weiterverarbeitet werden können, da die Informationen meist in einheitlicher (normalisierter Form) vorliegen.<br><br>
    <ul>
        <li><ins> "author"</ins>

  > Die Zeilen der Spalte enthalten jeweils den alphanumerischen`<Benutzernamen>` des Beschwerdeverfassers sowie eine, durch ein "of" getrente, US-Ortsangabe welche im Format `<Ortsname "of" US-Bundesstaat>` vorliegt. 
  

  Die Datenexploration durch eine Ortsdatenanalyse zeigte, dass 3 US-Ortsangaben ['BC', 'ON', 'PE'] ungültig sind, aus 17 Bundesstaaten eine dreistellige Anzahl an Beschwerden zu verzeichnen ist, von deren auf ['FL: 778', 'CA: 554', 'GA: 414'] entfallen und aus 5 Bundesstaaten ['IA', 'MT', 'OK', 'RI', 'SD'] keine Beschwerden erfasst wurden.<br>
        <li><ins>"posted_on"</ins><br>

  > Die Zeilen der Spalte "posted_on" enhalten Datumsangaben mit alphabetisch abgekürzer Monatsangabe über einen Zeitraum von 16 Jahren im amerikanischem Format `<Monat>`.`<Tag>`,`<Jahr>`. 
  
   Im Rahmen der Datenexploration wurde eine Zeitdatenanlyse durchgeführt welche Muster in der (jährlichen, monatlichen, wöchentlichen) Verteilung der Beschwerden im Datensatz über den Zeitraum vom 31.07.2000 bis 22.11.2016 zeigte.

   <ins>jährliche Verteilung:</ins> Die EDA zeigt, dass die meisten Beschwerden im Jahr 2015 erfolgten sind. 
   
   <ins>monatliche Verteilung:</ins> Die EDA zeigte weiter, dass die meisten Beschwerden im August (540) und wie wenigsten Beschwerden im April (369) abgesetzt wurden, wobei der Datensatz ein saisonales Muster zeigt. 
   
   <ins>wöchentliche Verteilung:</ins> Die Verteilung der Beschwerden aufgeschlüsselt nach Wochentagen zeigt, dass die meisten Beschwerden mittwochs (993), dienstags (960) und donnerstags (861), gefolgt von montags (820) und freitags (802) abgesezt wurden, wohingegen an den Tagen der Wochenenden Samstag (659) und Sonntags (564) weniger Beschwerden zu verzeichnen sind. Dieses Muster deutet ebenfalls auf einen organischen Ursprung des Datensatzes hin.

  <li><ins>"rating"</ins><br>
  
  > Die Zeilen der Spalte "rating" enthalten Bewertungen auf einer Skala `<0-5>`.<br>

  Die Anzahl der Bewertungen nach "rating" ist wie folgt verteilt [rating: 0=1560 (27.57%); 1=3734 (65.98%); 2=260 (4.59%); 3=54 (0.95%); 4=19 (0.34%); 5=32 (0.57%)] was einen Überhang niedriger Bewertungen zeigt. Dies weist ebenfalls auf einen organischen Datensatz hin, da Beschwerden grundsätzlich negativ sind. 

  </ul>
</ul>

<ul>
  <li><ins>EDA der unstruktuierte Daten</ins></li>
  Unstrukturierte Daten sind Informationen, die in einer nicht identifizierbaren Datenstruktur vorliegen. Ein typisches Beispiel dafür sind natürlichsprachliche Texte wie sie in der Spalte "text" vorhanden sind.<br><br>
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
</ol>
  </details>

## Datensatzaufbereitung (engl. dataset preparation)
In der Phase der Datensatzbereinigung werden die in der EDA gewonnen Erkenntnisse genutzt, um den Datensatz für den Anwendungsfall vorzubereiten. Hierzu wird eine Datenbereinigung sowie eine Datenvalidierung durchgeführt, wodurch diejenigen Daten bestimmt werden, die weiter verarbeitet werden.
<ol>
    <details>
      <summary>⚪ Datensatzbereinigung (engl. dataset cleaning)</b></summary>
      <i>
      
###### Fehlwertbehandlung
Die Behandlung von Fehlwerten wie NaNs (Not a Number) oder NaTs (Not a Text) kann durch listenweisen Fallausschluss, durch welchen Zeilen ohne Text oder Text unter einer Mindeslänge entfernt wird oder Imputation, das Auffüllen oder Ersetzen fehlender oder unvollständiger Textelemente durch geschätzte Werte, damit der Datensatz für Modelltraining oder Analyse vollständig nutzbar bleibt.

###### Duplikatentfernung
Durch die Duplikatentfernung werden doppelte Zeilen im Datensatz entfernt, um Verzerrungen des NLP-Models zu vermeiden. </i><br>
    </details>
</ol>
<ol>
    <details>
      <summary>⚪ Datensatzvalidierung (engl. dataset validation)</b></summary>
      <i>Im Rahmen der Datensatzvalisierung werden fehlerhafte Daten korrigiert, verworfen oder speziell behandelt um Datenqualität und Aussagekraft zu sichern.</i><br>
    </details>
</ol>

## Datenverarbeitung (engl. data processing)
Im maschinellen Lernen stellen Merkmale (engl. features) kategorielle oder numerische Größen dar, anhand derer Algorithmen oder neuronale Netze Texte klassifizieren oder clustern können.[^16] Innerhalb von NLU dienen die Features als Brücke zwischen rohem Text und algorithmischer Verarbeitung: Sie extrahieren relevante linguistische Informationen auf lexikalischer, syntaktischer oder semantischer Ebene. 

###### Pipeline Eingabe (engl. pipeline input)
Die Sprachverarbeitung beginnt mit dem Import des aufbereiteten Datensatz *"complaints_data_cleaned.csv"*, genauer dem Import der Spalte `<text>` welche als als Korpus für die folgenden NLP-Schritte genutzt wird. Die Zeilen des Datensatzes werden auch als Dokumente bezeichnet. 

<div style="margin-left: 2em;">
  <code>pandas</code>&nbsp;<code>????</code><br>
</div>

### Datenvorverarbeiten (engl. data pre-processing)
> Während der Datenvorverarbeitung erfolgt die *Merkmalsvorbereitung (engl. feature preparation)* für nachfolgende Schritte in einem mehrstufigen Prozess, welcher sich grob in Textbereinigung (engl. text cleaning) und Merkmalsextraktion unterteilen lässt. 
<div style="margin-left: 2em;">
  <code>spaCy</code>&nbsp;<code>NLTK</code><br>
</div>

#### Textbereinigung (engl. text cleaning)
Im Rahmen der Textbereinigung werden Texte von Rauschen befreit und standardisiert.<br>
<img src="docs/2 - Textbereinigung (engl. text cleaning).jpg">

<ol type="1">
  <details>
    <summary>🔴 Rauschentfernung (engl. noise reduction)</summary>
    <p><i>Ziel der Rauschentfernung ist es irrelevante Token (Zeichen und Zeichenketten) für nachfolgende Prozesse zu identifizieren und zu löschen.</i></p>
    <ol type="1">
      </li>
      <li>Wortbereinigung (engl. word cleaning)</li>
        <i>XXX</i>
            <div style="margin-left: 2em;">
              <code>spaCy</code>&nbsp;<code>NLTK(stopwords)</code><br><br>
            </div>
            <ol type="2">
              </div>
      <li>Zeichenbereinigung ()</li>
            <li><ins>Satzzeichen (engl. punctuation marks)</ins></li>
              <div style="margin-left: 2em;">
                <code>spaCy</code>&nbsp;<code>regex</code><br><br>
              </div>
            <li><ins>Leerzeichen (engl. white space)</ins></li>
              <div style="margin-left: 2em;">
                <code>spaCy</code>&nbsp;<code>regex</code><br><br>
              </div>
            <li><ins>Sonderzeichen (engl. special character)</ins></li>
              <div style="margin-left: 2em;">
                <code>spaCy</code>&nbsp;<code>textnorm</code><br><br>
              </div>
        <li>Nummernbereinigung (engl. numbers cleaning)</li>
            <li><ins>Nummern (engl. removing numbers)</ins></li>
              <div style="margin-left: 2em;">
                <code>spaCy</code>&nbsp;<code>regex</code><br><br>
              </div>
      </ol>
  </details>
  <details>
    <summary>🔴 Standardisierung (engl. standardisation)</summary>
    <p><i>Durch Standardisierung werden relevante Token vereinheitlicht. Hierdurch wird vermieden, dass gleiche Inhalte nicht in mehreren leicht unterschiedlichen Varianten auftreten.</i></p>
          <div style="margin-left: 2em;">
            <code>???</code>&nbsp;<code>????</code><br><br>
          </div>
      <ol type="2">
            <li><ins>Normalisierung (engl. normalisation)</ins></li>
            Durch die Normalisierung wird Text in ein einheitliches Format, besser zu analysierendes Format gebracht.
            </div>
            <ul>
              <li>Kasusumwandlung (engl. case conversion)</li>
              In diesem Schritt erfolgt eine konsequente Kleinschreibung (engl. lowercasing) aller Wörter.
              <div style="margin-left: 2em;">
                <code>spaCy</code>&nbsp;<code>stdlib(.lower)</code><br><br>
              <li>Formatnormalisierungen (engl. format normalisations)</li>
              In der Formatnormalisierung erfolgt die Normalisierung von Schreibweisen (Datenformate oder Zahlenformaten) und Sonderformen (Emojis).
              <div style="margin-left: 2em;">
                <code>datetime</code><br><br>
              </div>
            </ul>
            <li><ins>Rechtschreibfehlerkorrektur (engl. spelling correction)</ins></li>
            Durch Rechtschreibkorrekrur werden Schreib- und Tippfehler korrigiert, um eine linguistisch korrekte Analyse gewährleisten.
              <div style="margin-left: 2em;">
                <code>pyspellchecker</code>&nbsp;<code>contextualSpellCheck</code>&nbsp;<code>PyEnchant</code><br><br>
              </div>
    </details>
      </ol>
    </ol>



#### Linguistische Analyse
Im Rahmen der lingusitischen Analyse erfolgt, je nach Anwendungsfall neben einer lexikalischen, eine syntaktische und/oder semantische Verarbeitung von bereinigten Texten zur Merkmalsvorbereitung.
<ol type="1">
    <details>
      <summary>🔴 lexikalische Verarbeitung (engl. lexical processing)</summary>
      <p><i>Im Rahmen der lexikalischen Analyse werden Texte tokenisiert, Token auf ihre Grundform reduziert und ein Vokabular aufgebaut.</i></p>
      <ol type="1">
        <li>Tokenisierung (engl. tokenization)<br>
          <i>Durch Tokenisierung wird der vorbereitete Text in Einzeltoken (Worte) oder N-Gramme (Phrasen) wie z.B. Sätze zerlegt. Tokenisierung zerlegt Text in Token (Wörter, Subwörter oder Zeichen), aus denen das Vokabular als Menge eindeutiger Token-IDs entsteht.</i><br>
          <div style="margin-left: 2em;">
            <code>SpaCy</code>&nbsp;<code>NLTK(word_tokenize; sent_tokenize)</code><br><br>
          </div>
        </li>
        <li>Grundformreduktion (engl. inflection reduction)<br>
          <i>Durch Grundformreduktion werden Wörter auf ihre Grundformen reduziert. Da Lemmatisierung (engl. lemmatization) genauer als Stammformreduktion (engl. stemming) ist, wird diese eingesetzt.</i><br>
          <div style="margin-left: 2em;">
            <code>spaCy</code>&nbsp;<code>NLTK (WordNetLemmatizer)</code><br><br>
          </div>
        </li>
        <li>Vokabularerstellung/Wortschatzaufbau (engl. vocabulary construction)<br>
          <i>Im Schritt des Wortschatzaufbaus wird aus dem tokenisierten Textkorpus ein endliches Vokabular erstellt, das allen Token eine eindeutige Token-ID zuweist. Das Vokabular stellt eine Menge eindeutiger Token-IDs dar.</i><br>
          <div style="margin-left: 2em;">
            <code>sklearn(CountVectorizer)</code><br><br>
          </div>
        </li>
        <li>lexikalisches POS-Tagging<br>
          <i>Durch lexikalisches Part-of-Speech Tagging können mittels Lookup-Tabellen grammatikalische Wortfunktionen und Kategorien zu einem gegebenen Text hinzugefügt werden. Das Vokabular wird dabei mit Sprachdatenannotationen (engl. linguistic annotations) versehen.</i><br>
          <div style="margin-left: 2em;">
            <code>XXXX</code><br><br>
          </div>
        </li>
      </ol>
    </details>
    <details>
      <summary>🔴 syntaktische Verarbeitung (engl. syntactic processing)</summary>
      <p><i>Im Rahmen der syntaktischen Analyse werden Satzstrukturen und grammatikalische Funktionen analysiert.</i></p>
      <ol type="1">
        <li>syntaktisches POS-Tagging<br>
          <i>Durch Wortart-Tagging (engl. Part-of-Speech Tagging) können durch Modelle (Hidden Markov Models - HMM) grammatikalische Wortfunktionen und Kategorien durch Tags-Sets zu einem gegebenen Text hinzugefügt werden und so das Vokabular sprach-/domänenspezifisch annotieren.</i><br>
          <div style="margin-left: 2em;">
            <code>spaCy</code>&nbsp;<code>NLTK</code><br><br>
          </div>
        </li>
        <li>syntaktisches Parsen (engl. syntax parsing)<br>
          <ol type="1">
            <li>Flaches Parsen (engl. shallow parsing/chunking)<br>
              <i>Beim flachen Parsen werden aufeinanderfolgende Wörter zu Satzgliedern (engl. chunks) gruppiert, ohne eine vollständige Satzstruktur zu analysieren. Dies ermöglicht eine schnelle und effiziente Extraktion von Nominalphrasen und Verbalphrasen.</i><br>
              <div style="margin-left: 2em;">
                <code>NLTK (ne_chunk)</code>&nbsp;<code>spaCy</code><br><br>
              </div>
            </li>
            <li>Tiefes Parsen (engl. deep parsing/dependency parsing)<br>
              <i>Beim tiefen Parsen wird die vollständige Satzstruktur durch Abhängigkeitsrelationen zwischen Wörtern analysiert. Dies ermöglicht ein tieferes Verständnis von Satzbeziehungen und grammatikalischen Strukturen.</i><br>
              <div style="margin-left: 2em;">
                <code>NLTK</code>&nbsp;<code>spaCy (dependency parser)</code><br><br>
              </div>
            </li>
          </ol>
        </li>
      </ol>
    </details>
    <details>
      <summary>🔴 semantische Verarbeitung (engl. semantic processing)</summary>
      <p><i>Im Rahmen der semantischen Verarbeitung werden Bedeutungen und Zusammenhänge im Text analysiert.</i></p>
      <ol type="1">
        <li>semantisches Parsen (engl. semantic parsing)<br>
          <i>Durch semantisches Parsen werden Bedeutungsstrukturen extrahiert, um tieferes Textverständnis zu ermöglichen.</i><br>
          <div style="margin-left: 2em;">
            <code>spaCy</code><br><br>
          </div>
          <ol type="1">
            <li>Eigennamenerkennung (engl. Named Entity Recognition - NER)<br>
              <i>Bei der der Erkennung benannter Entitäten werden Wörter in einem unstrukturierten Text als Kategorien klassifiziert (Namen, Orte, Zeit, Datum, Organisationen, Mengen).</i><br>
              <div style="margin-left: 2em;">
                <code>spaCy</code><br><br>
              </div>
            </li>
            <li>Koreferenzauflösung (engl. Coreference Resolution - CR)<br>
              <i>Koreferenzauflösung ist eine Methode die es Modellen ermöglicht, Referenzen auf dieselbe Entität oder dasselbe Konzept innerhalb eines Textes zu erkennen. Durch diese Technik können KI-Systeme besser verstehen, auf wen oder was sich Pronomen, Namen oder Nominalphrasen in einem Satz oder Absatz beziehen.</i><br>
              <div style="margin-left: 2em;">
                <code>Library1</code>&nbsp;<code>Library2</code><br><br>
              </div>
            </li>
            <li>Beziehungsextraktion (engl. Relationship Extraction - RE)<br>
              <i>Die Beziehungsextraktion identifiziert Beziehungen zwischen benannten Objekten in einem Text, beispielsweise zwischen Vater und Sohn oder zwischen Mutter und Tochter, ect.</i><br>
              <div style="margin-left: 2em;">
                <code>Library1</code>&nbsp;<code>Library2</code><br><br>
              </div>
            </li>
          </ol>
        </li>
      </ol>
    </details>
</ol>

### Datenvorbereitung (engl. data preparation)
> Im Rahmen der Datenverarbeitung werden Merkmale (engl. features) erzeugt und ausgewählt. Dies erfolgt durch  Merkmalsgenerierung (engl. feature generation/featurization) und Merkmalsauswahl (engl. feature selection). <br> Merkmalsgenerierung (engl. feature generation) bezeichnet in der NLP-Pipeline den Prozess, aus rohem oder vorverarbeitetem Text neue, informative Merkmale zu erzeugen, die Machine-Learning-Modelle effizient nutzen können. Sie wandelt unstrukturierte Daten in numerische oder kategorische Repräsentationen um, die syntaktische, semantische oder kontextuelle Aspekte einfangen. Dabei werden Attribute/Features in eine für die Modellierung adäquate Form überführt, weshalb von Merkmalsaufbereitung (engl. feature engineering) gesprochen wird (Baars und Kemper, 2021, p. 159). Dies kann mittels Merkmalskonstruktion, Merkmalsextraktion oder Merkmalsumwandlung erfolgen oder automatisch durch trainierte Modelle vorgenommen werden. In diesem Fall spricht man von Merkmalslernen (engl. feature learning / representation learning), wobei Merkmale direkt aus Rohtexten gewonnen werden. <br> Merkmalsauswahl (engl. feature selection) ist ein komplementärer Prozess, der aus einer großen Menge von erzeugten Merkmalen die relevantesten auswählt. Dies reduziert Dimensionalität, verbessert Modellperformance und verringert Rechenaufwand, indem irrelevante oder redundante Merkmale entfernt werden. <br>

feature - explizite / Abstrakte

<div style="margin-left: 2em;">
  <code>???</code>&nbsp;<code>???</code><br><br>
</div>

#### Merkmalsgenerierung (engl. feature generation/featurization)
Merkmalsgenerierung bezeichnet den Prozess, aus rohem oder vorverarbeitetem Text neue, informative Merkmale zu erzeugen. Unstrukturierte Daten werden durch Merkmalskodierung (engl. feature encoding) in numerische oder kategorische Repräsentationen überführt, die Machine-Learning-Modelle nutzen können.

<ol type="1">
  <details>
    <summary>🟡 Vektorisierung (engl. vectorization)</summary>
    <p><i>Als Vektorisierung wird die Merkmalskodierung (engl. feature encoding) von Textdaten bezeichnet. Die Token (Wörter, Subwörter oder Zeichen) aus dem Vokabular werden durch Vektorisierungstechniken in numerische Repräsentationen überführt, die als Merkmalsvektoren in einem n‑dimensionalen Merkmalsraum (enfl. feature space) dargestellt und zu Merkmalsmatrizen zusammengefasst werden. Vektorisierungstechniken nutzen Merkmalsextraktion, um Texte je nach Anwendungsfall auf Silben,- Wort-, Satz-, Segment‑ oder Dokumenten‑Ebene für Modelle aufzubereiten, um lexikalische, syntaktische oder kontextuelle Aspekte eines Textes einzufangen. - explizite Features</i></p>
    <ol type="1">
        <details>
          <summary>🟡 Merkmalsvektoren (engl. feature vectors)</summary>
          <p><i>Spannen keinen semantischen Merkmalsraum auf, sondern erzeugen dünn besetzte Vektoren (engl. sparse vectors) auf Basis von Tokenfrequenzen, was Modellen eine algebraische bzw. statistische Auswertung ermöglicht. Teils werden die Merkmalsvektoren auch als unsemantische oder häufigkeitsbasierte Embeddings (engl. frequency based embeddings) bezeichnet. Diese frequenzbasierten Methoden erzeugen dünn besetzte Merkmalsvektoren basierend auf Vokabularpositionen, wobei zwischen Methoden mit und ohne Informationsgewichtung diffrenziert wird.</i></p>
          <ul>
          <li><ins>BoX (Bag-of-X)</ins></li>
          Bei den Bag-of-X-Methoden erfolgt keine Informationsgewichtung, Token oder Tokensequenzen wird eine eigene Dimension zugewiesen.<br>
          <ul>
            <li><ins>BoX auf Einzeltoken</ins><br>
            Wird die Methode auf Wortebene durchgeführt, wird sie als Bag-of-Words (BoW) bezeichnet. Der „Bag-of-Words-Vektor hat für jedes Wort eine eigene Dimension. Wenn das Vokabular n Wörter umfasst, wird ein Dokument zu einem Punkt (Dokumentenvektor) in einem n-dimensionalen Raum.“ (Zheng und Casari, 2019, p. 41)
            <div style="margin-left: 2em;">
              <code>sklearn (CountVectorizer)</code>
            </div>
            <li><ins>BoX auf Tokensequenzen</ins><br>
            Wird die Methode mit einer Folge von n-Token durchgeführt, wird sie als Bag-of-N-Grams (BoN) bezeichnet, was eine lokal auf die Tokensequenz begrenzte Kontexterfassung ermöglicht. „Je größer n ist, desto reicher ist der Informationsgehalt und desto höher die Kosten“ für Berechnung, Speicherung und Modellierung (Zheng und Casari, 2019, p. 44). Was bedeutet, dass sich bei BoN ein viel größerer und dünner besetzten Merkmalsraum ergibt.
            <div style="margin-left: 2em;">
              <code>sklearn (CountVectorizer(ngram_range))</code><br><br>
            </div>
          </ul>
          </ul>
      </li>
          <ul>
            <li><ins>TF-IDF (term frequency times inverse document frequency)</ins><br>
            Bei der TF-IDF-Methode handelt es sich um eine statistische Erweiterung von BoX, durch welche eine Informationsgewichtung der Token bzw. Tokensequenzen vorgenommen wird.
              <ul>
              <li><ins>TF-IDF auf Einzeltoken</ins><br>
              Wird die TF-IDF-Methode auf Wortebene durchgeführt, werden Einzelwörter gewichtet, um ihre Relevanz im Dokument und im Korpus auszudrücken.
              <div style="margin-left: 2em;">
                <code>sklearn (TfidfVectorizer)</code>
              </div>
              <li><ins>TF-IDF auf Tokensequenzen</ins><br>
              Wird die TF-IDF-Methode mit einer Folge von n-Token durchgeführt, werden Wort-Paare oder längere Phrasen gewichtet, um ihre Relevanz auszudrücken.
              <div style="margin-left: 2em;">
                <code>sklearn (TfidfVectorizer(ngram_range))</code><br><br>
              </div>
              </ul>
            </li>
          </ul>
      </li>
        </details>
    </ol> 
      <ol type="1">
        <details>
          <summary>🟡 Merkmalseinbettungen (engl. feature embeddings)</summary>
          <p><i>Spannen einen semantischen Merkmalsraum auf und liefern dichtbesetzte Vektoren (engl. dense vectors), was Modellen eine Auswertung von semantischen Ähnlichkeiten über Abstände bzw. Ähnlichkeitsmaße ermöglicht. Einbettungen weisen jedem Merkmal einen dichten Vektor im semantischen Raum zu und erfassen so die Bedeutungsdimension mittels vorhersage- oder kontextbasierten Verfahren. Worteinbettungen = / Satzeinbettungen = Entstehung duch vortrainierte Modelle.
          Semi-explizite Features sind ein Mittelding zwischen explizit und latent. Sie entstehen vor allem bei Word Embeddings (Word2Vec, GloVe, FastText, BERT, ELMo):
          </i></p>
          <ul>
          <li><ins>vorhersagebasierte Wort-Einbettungen (engl. prediction based word embeddings)</ins></li>
            <ul>
              <li>GloVe (Global Vectors for Word Representation)</li>
              <li>Word2Vec</li>
              <li>FastText</li>
              - sind STATISCHE EMBEDDINGS „übersetzen jedes Wort zu einem festen Vektor – unabhängig vom Kontext, in dem es steht.“ (Wehner, 2026)„Das Training erfolgt auf riesigen Textkorpora. Die Modelle lernen, Wörter mit ähnlichen Kontexten auch im Vektorraum zusammenzubringen“ (Wehner, 2026)vSprachliche Vieldeutigkeiten (Polysemie) und Kontextänderungen werden hier noch nicht abgebildet.“ (Wehner, 2026)
            </ul>
            <div style="margin-left: 2em;">
              <code>gensim</code> <code>sentence-transformers</code><br>
            </div>
          <li><ins>kontextbasierte Wort-Einbettungen (engl. contextualized word embeddings)</ins></li>
            <ul>
              <li>ELMo (Embeddings from Language Models)</li>
              <li>BERT (Bidirectional Encoder Representations from Transformers)</li>
              „Semantische und syntaktische Unterschiede werden so erstmals maschinell berücksichtigt.“ (Wehner, 2026)„Jedes Wort, jede Phrase, jeder Satz bekommt situativ angepasste Embedding-Vektoren – in Abhängigkeit vom umgebenden Kontext.“ (Wehner, 2026)
              <li>GPT (Generative Pre-trained Transformer)</li>
            </ul>
            <div style="margin-left: 2em;">
              <code>transformers</code> <code>sentence-transformers</code><br>
            </div>
          <li><ins>Satz-Einbettungen (engl. sentence embeddings)</ins></li>
            <ul>
              <li>Contextualized Sentence Embedding</li>
              <li>SBERT Embedding</li>
            </ul>
            <div style="margin-left: 2em;">
              <code>sentence-transformers</code><br>
            </div>
    </ul>
      </li>
        </details>
    </ol>
  </details>
</ol>

#### Merkmalsauswahl (engl. feature selection)
<p><i>Merkmalsauswahl ist ein komplementärer Prozess zur Merkmalsgenerierung. Nach der Erzeugung von Features werden die relevantesten Merkmale aus dem bestehenden Merkmalsraum ausgewählt, um Redundanz zu reduzieren, Overfitting zu vermeiden und die Modellleistung zu optimieren.</i></p>
      <ol type="1">
        <details>
          <summary>🟡 Filtermethoden</summary>
          <p><i>wählen Features basierend auf statistischen Eigenschaften (z.B. Korrelation, Chi-Quadrat)</i></p>
        </details>
        <details>
          <summary>🟡 Wrapper-Methoden</summary>
          <p><i>evaluieren Feature-Subsets durch Modelltraining</i></p>
        </details>
        <details>
          <summary>🟡 Eingebettete Methoden</summary>
          <p><i>wählen Features während des Modelltrainings aus (z.B. Lasso, Tree-based)</i></p>
        </details>
        <div style="margin-left: 2em;">
          <code>sklearn (SelectKBest, SelectPercentile, RFE)</code><br><br>
        </div> 
      </ol>
Repräsenationen: Merkmalsvektoren / Merkmalsmatrix (numerische Daten)
„Ein Vektor ist nichts weiter als eine Sammlung von n Zahlen" (Zheng und Casari, 2019, p. 40).


#### Modellbildung (engl. model training/building)
<p><i>Modellbildung ist der Prozess, in dem die Merkmalslernen-Modelle mit den vorbereiteten Daten trainiert werden. Dabei werden die Modellparameter iterativ optimiert, um optimale Features zu lernen. Der Trainingsprozess umfasst die Festlegung von Hyperparametern (z.B. Anzahl der Themen K, Lernrate, Iterationen), die Initialisierung des Modells und die iterative Anpassung der Parameter bis zur Konvergenz oder zum Erreichen einer maximalen Anzahl von Iterationen.</i></p>

#### Merkmalslernen (engl. feature learning / representation learning)
<p><i>Merkmalslernen ist ein automatisierter Prozess, bei dem ein Modell selbst neue informative Merkmale aus den vorhandenen oder rohen Features lernt und entdeckt. Im Gegensatz zu manuellem Feature Engineering werden die Merkmale nicht von Menschen definiert, sondern vom Modell während des Trainings durch Algorithmen erlernt. Dabei wird eine Merkmalsumwandlung (engl. feature transformation) durch Modelle durchgeführt. Neuen Features entstehen dabei entweder durch semantische Abstraktion (neue interpretierbare Konzepte), Merkmalsabstraktion (engl. feature abstraction) oder mathematische Projektion (neue Achsen), Merkmalsprojektion (engl. feature projection).</i></p>

<ol type="1">
  <details>
    <summary>🟡 Merkmalsabstraktion (engl. feature abstraction)</summary>
    <p><i>Merkmalsabstraktion bedeutet, dass Modelle neue Konzepte oder Bedeutungen direkt aus den Daten herausfinden und als neue Features repräsentieren. Die neuen Features sind semantisch interpretierbar und nicht nur mathematische Transformationen.</i></p>
    <b>Merkmalsabstraktion:</b> Modelle extrahieren neue interpretierbare Konzepte aus Features (z.B. Themen, Embeddings)</li>
    <ol type="1">
      <details>
        <summary>🟡 Themenmodellierung (engl. topic modeling)</summary>
        <p><i>Themenmodellierung identifiziert unüberwacht latente abstrakte Themen in Textsammlungen. Diese neuen Merkmale (Themen) sind nicht explizit im Text vorhanden, sondern werden durch mathematische Modelle aus den bestehenden Merkmalen automatisch extrahiert oder transformiert. Topic-Modelle unterscheiden sich je nachdem, ob sie auf Merkmalsabstraktion oder Merkmalsprojektion basieren.</i></p>
        <ol type="1">
          <details>
            <summary>🟡 Abstraktion-basierte Modelle</summary>
            <p><i>Abstraktion-basierte Topic-Modelle erzeugen neue interpretierbare Konzepte direkt aus Merkmalsmatrizen oder Embeddings. Sie transformieren die Input-Features semantisch und ergeben neue Bedeutungsdimensionen.</i></p>
            <ol type="1">
              <details>
                <summary>🟡 LDA (Latent Dirichlet Allocation)</summary>
                <p><i>Latent Dirichlet Allocation (LDA) extrahiert direkt latente Themen aus der Merkmalsmatrix, indem es wahrscheinlichkeitsbasierte Themen-Wort-Verteilungen identifiziert. LDA ist eine probabilistische Merkmalsabstraktion, die interpretierbare Themen erzeugt.</i></p>
                <div style="margin-left: 2em;">
                  <code>gensim</code>&nbsp;<code>sklearn (LatentDirichletAllocation)</code><br><br>
                </div>
                <p><b>Output:</b> Themenmischung pro Dokument (α), Wort-Gewichte pro Thema (β), K latente Themen</p>
              </details>
              <details>
                <summary>🟡 NMF (Non-Negative Matrix Factorization)</summary>
                <p><i>Non-Negative Matrix Factorization (NMF) zerlegt die Merkmalsmatrix in zwei Faktormatrizen mit nicht-negativen Werten. Jedes Dokument ist eine Kombination von Basistopics, ähnlich LDA, aber deterministisch statt probabilistisch.</i></p>
                <div style="margin-left: 2em;">
                  <code>sklearn (NMF)</code><br><br>
                </div>
                <p><b>Output:</b> Topic-Gewichte pro Dokument, Wort-Gewichte pro Topic, K Themen</p>
              </details>
              <details>
                <summary>🟡 BERTopic</summary>
                <p><i>BERTopic ist eine moderne Erweiterung klassischer Topic-Modeling-Methoden, die vortrainierte BERT-Embeddings mit Dimensionsreduktion (UMAP) und Clustering (HDBSCAN) kombiniert. Sie erzeugt interpretierbare und semantisch kohärente Themen direkt aus Embeddings, ohne dass eine separate Merkmalsmatrix nötig ist, und ist besonders effektiv bei großen Textsammlungen.</i></p>
                <div style="margin-left: 2em;">
                  <code>bertopic</code>&nbsp;<code>sentence-transformers</code>&nbsp;<code>umap-learn</code><br><br>
                </div>
                <p><b>Output:</b> Topic-Label pro Dokument, Wort-Gewichte pro Topic, Cluster-Visualisierung</p>
              </details>
            </ol>
          </details>
          <details>
            <summary>🟡 Projektion-basierte Modelle</summary>
            <p><i>Projektion-basierte Topic-Modelle nutzen mathematische Projektionen (wie SVD), um neue Dimensionen aus der Merkmalsmatrix zu extrahieren. Obwohl sie durch mathematische Transformationen arbeiten, erzeugen sie dennoch interpretierbare latente Konzepte, die als Themen fungieren.</i></p>
            <ol type="1">
              <details>
                <summary>🟡 LSA (Latent Semantic Analysis)</summary>
                <p><i>Latent Semantic Analysis (LSA) nutzt Singulärwertzerlegung (SVD), um latente semantische Dimensionen aus der Merkmalsmatrix zu extrahieren. LSA erzeugt interpretierbare Themen ähnlich LDA und NMF, funktioniert aber durch mathematische Projektion. Dies macht LSA effektiv und effizient für Topic Modeling bei Textsammlungen.</i></p>
                <div style="margin-left: 2em;">
                  <code>sklearn (TruncatedSVD)</code><br><br>
                </div>
                <p><b>Output:</b> K latente Dimensionen, Singular Values, K LSA-Komponenten</p>
              </details>
            </ol>
          </details>
        </ol>
      </details>
      <details>
        <summary>🟡 Worteinbettungen (engl. word embeddings)</summary>
        <p><i>Worteinbettungen sind dichte Vektoren, die semantische Bedeutungen von Wörtern oder Sätzen repräsentieren. Sie entstehen durch das Training von Modellen auf Textdaten und erfassen semantische und syntaktische Beziehungen.</i></p>
        <ol type="1">
          <details>
            <summary>🟡 Vorhersagebasierte Wort-Einbettungen</summary>
            <p><i>Diese Embeddings werden durch Vorhersage von Wörtern basierend auf ihrem Kontext trainiert.</i></p>
            <ul>
              <li>Word2Vec (Skip-gram, CBOW)</li>
              <li>GloVe (Global Vectors for Word Representation)</li>
              <li>FastText</li>
            </ul>
            <div style="margin-left: 2em;">
              <code>gensim</code>&nbsp;<code>sentence-transformers</code><br><br>
            </div>
          </details>
          <details>
            <summary>🟡 Kontextbasierte Wort-Einbettungen</summary>
            <p><i>Diese Embeddings werden durch große Sprachmodelle erzeugt, die Kontext bidirektional nutzen.</i></p>
            <ul>
              <li>ELMo (Embeddings from Language Models)</li>
              <li>BERT (Bidirectional Encoder Representations from Transformers)</li>
              <li>GPT (Generative Pre-trained Transformer)</li>
            </ul>
            <div style="margin-left: 2em;">
              <code>transformers</code>&nbsp;<code>sentence-transformers</code><br><br>
            </div>
          </details>
        </ol>
      </details>
    </ol>
  </details>
</ol>

<ol type="1">
  <details>
    <summary>🟡 Merkmalsprojektion (engl. feature projection)</summary>
    <p><i>Merkmalsprojektion transformiert hochdimensionale Features auf neue mathematische Achsen. Die neuen Dimensionen sind mathematische Kombinationen der originalen Features, nicht semantisch interpretierbar, aber nützlich für Visualisierung oder Dimensionsreduktion.</i></p>
    <b>Merkmalsprojektion:</b> Modelle transformieren Features auf neue mathematische Achsen (z.B. PCA, T-SNE)</li>
    <ol type="1">
      <details>
        <summary>🟡 Lineare Projektionen</summary>
        <p><i>Lineare Projektionen reduzieren Dimensionen durch orthogonale Transformationen, die Varianzrichtungen im Datenraum erfassen.</i></p>
        <ol type="1">
          <details>
            <summary>🟡 PCA (Principal Component Analysis)</summary>
            <p><i>PCA findet die Hauptkomponenten (Richtungen maximaler Varianz) in den Daten und projiziert Features auf diese Achsen.
              PCA-Modell (Hauptkomponentenanalyse)
              „Die PCA-Projektion (links) zeigt eine dichte Cloud, die sich um den Ursprung dreht, mit vielen Überschneidungen zwischen den verschiedenfarbigen Punkten. Genau das erwarten wir von PCA – sie erfasst die Richtungen der maximalen Varianz in den Daten, schafft es aber nicht, die verschiedenen Personen effektiv voneinander zu trennen. Wir können sehen, dass Gesichter von verschiedenen Leuten (unterschiedliche Farben) komplett durcheinander sind, sodass es fast unmöglich ist, bestimmte Gruppen zu erkennen.“ (Thevapalan, 2025)
            </i></p>
            <div style="margin-left: 2em;">
              <code>sklearn (PCA)</code><br><br>
            </div>
            <p><b>Output:</b> K Hauptkomponenten, Varianzanteil pro Komponente</p>
          </details>
        </ol>
      </details>
      <details>
        <summary>🟡 Nichtlineare Projektionen</summary>
        <p><i>Nichtlineare Projektionen bewahren lokale oder globale Strukturen in den Daten besser, sind aber rechnerisch aufwendiger. Sie werden hauptsächlich für Visualisierung verwendet.
        UMAP (Uniform Manifold Approximation and Projection)
        „Natürliche Sprachverarbeitung: Textdaten können, wenn sie in hochdimensionale Einbettungen umgewandelt werden, mit UMAP visualisiert werden, um semantische Beziehungen zu verstehen. Es wird oft benutzt, um Wort-Embeddings und Dokument-Cluster zu zeigen und Sprachmodelle zu debuggen, indem es zeigt, wie verschiedene Konzepte im Embedding-Raum miteinander zusammenhängen.“ (Thevapalan, 2025)
        T-SNE (t-distributed Stochastic Neighbor Embedding)
        „Die t-SNE- Projektion (Mitte) zeigt eine Verbesserung mit gut voneinander getrennten Clustern. Jede Farbe (die für eine andere Person steht) bildet eine eigene, kompakte Gruppe mit klaren Grenzen zwischen den verschiedenen Personen. Schau mal, wie t-SNE fast perfekte lokale Gruppierungen macht, bei denen Gesichter derselben Person ganz nah beieinander liegen und von anderen Gruppen weggeschoben werden. Das ist die Stärke von t-SNE: Es ist super darin, lokale Nachbarschaften zu erhalten und visuell unterschiedliche Cluster zu erstellen.“ (Thevapalan, 2025)
        nichtlineare probabilistische Technik zur Dimensionalitätsreduzierung
        „Eigene Embedding Spaces erstellen & visualisieren  
        Zur Visualisierung bietet sich die t-SNE-Plot-Implementierung in sklearn an:  
        from sklearn.manifold import TSNE  
        import matplotlib.pyplot as plt  
        tsne = TSNE(n_components=2, random_state=42)  
        points_2d = tsne.fit_transform(embeddings)  
        plt.scatter(points_2d[:,0], points_2d[:,1])  
        plt.title("Embedding Space Visualisierung")  
        plt.show()“ (Wehner, 2026)
        </i></p>
        <ol type="1">
          <details>
            <summary>🟡 t-SNE (t-distributed Stochastic Neighbor Embedding)</summary>
            <p><i>t-SNE projiziert hochdimensionale Daten auf 2-3 Dimensionen und bewahrt dabei lokale Nachbarschaften. Ideal zur Visualisierung von Clustern und Gruppen.</i></p>
            <div style="margin-left: 2em;">
              <code>sklearn (TSNE)</code><br><br>
            </div>
            <p><b>Output:</b> 2-3D Koordinaten pro Datenpunkt zur Visualisierung</p>
          </details>
          <details>
            <summary>🟡 UMAP (Uniform Manifold Approximation and Projection)</summary>
            <p><i>UMAP ist eine moderne Alternative zu t-SNE, die schneller und skalierbarer ist. Es bewahrt sowohl lokale als auch globale Strukturen besser.
            </i></p>
            <div style="margin-left: 2em;">
              <code>umap-learn</code><br><br>
            </div>
            <p><b>Output:</b> 2-3D Koordinaten pro Datenpunkt zur Visualisierung</p>
          </details>
        </ol>
      </details>
    </ol>
  </details>
</ol>


  

#### Modellbewertung ()
Kohärenz/Perplexity (bewertet die Features) bewertung der erzeuugten Features.
    <i>Modellauswertung (engl. model evaluation)<i>
              <ul>
            <li>Kohärenz (engl. coherence)</li>
            „The Gensim library provides a class that implements the four most famous coherence models: _u_mass, c_v, c_uci, cnpmi. So, let’s break them into fundamental pieces.“ (Pedro, 2022)
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br><br>
              </div>
            <li>Perplexität (engl. perplexity)</li>
            xxx
              <div style="margin-left: 2em;">
                <code>???</code>&nbsp;<code>????</code><br><br>
              </div>
          </ul>
  </details>
</ol>


#### Merkmalsanalyse (engl. feature analysis)
<p><i>Merkmalsanalyse ist der analytische Prozess, bei dem bereits erstellte, ausgewählte oder gelernte Merkmale untersucht, beschrieben und interpretiert werden. Dies erfolgt durch Merkmalserkennung, um spezifische Muster und Strukturen in den Daten zu identifizieren.</i></p>

<ol type="1">
  <details>
    <summary>🟡 Merkmalserkennung (engl. feature recognition) – Pattern-Erkennung</summary>
    <p><i>Merkmalserkennung identifiziert spezifische Muster, Anomalien oder Strukturen in bereits erzeugten oder gelernten Features durch regelbasierte oder lernbasierte Verfahren.</i></p>
    <ol type="1">
      <li><ins>Regelbasierte Erkennung</ins><br>
      <i>Verwendung von vordefinierten Regeln und Heuristiken zur Mustererkennung.</i>
        <div style="margin-left: 2em;">
          <code>NLTK (pattern matching)</code>&nbsp;<code>regex</code><br><br>
        </div>
      </li>
      <li><ins>Lernbasierte Erkennung</ins><br>
      <i>Verwendung von trainierten Modellen zur automatischen Mustererkennung und Klassifikation.</i>
        <ol type="1">
          <details>
            <summary>🟡 KeyBERT (Keyword/Term Extraction)</summary>
            <p><i>KeyBERT extrahiert automatisch interpretierbare Schlüsselwörter aus Dokumenten durch semantische Ähnlichkeit von BERT-Embeddings und ermöglicht so die schnelle Identifikation dominanter Begriffe in Textsammlungen.</i></p>
            <div style="margin-left: 2em;">
              <code>keybert</code>&nbsp;<code>sentence-transformers</code><br><br>
            </div>
          </details>
        </ol>
        <div style="margin-left: 2em;">
          <code>sklearn (Clustering, Classification)</code>&nbsp;<code>K-Means, DBSCAN</code><br><br>
        </div>
      </li>
    </ol>
  </details>
</ol>

###### Pipeline Ausgabe (engl. pipeline output)

Die verarbeiteten Daten fließen in die Datenkonsolidierung ein.
Scores, Labels, Logits und .
______________
### 🔵 Datennachverarbeitung (engl. data post-processing)
Post-Processing macht Daten nutzbar.

Datennachverarbeitung (engl. post-processing) erfolgt nach der Modellausführung (Inference), um rohe Modellausgaben nutzbar zu machen.
<img src="3 - Datennachverarbeitung (engl. data post-processing).jpg" width="1200">

### 🔵 Merkmalszusammenfassung (engl. feature aggregation)
Im Rahmen der Merkmalszusammenfassung erfolgt eine Konsolidierung der Modellausgaben, in der Merkmalsanalysen (engl. feature analysis) durchgeführt und letztlich als Datenpräsentation (engl. data presentation) aufbereitet werden.

**Merkmalsanalyse** untersucht und beschreibt die in der Merkmalsaufbereitung erstellten Merkmale. Während die **Merkmalsaufbereitung** (Phase Datenverarbeitung) konstruktiv arbeitet und Merkmale schafft, arbeitet die **Merkmalsanalyse** (Phase Datennachverarbeitung) analytisch und interpretiert diese Merkmale für Entscheidungsträger.

- **Merkmalsanalyse (engl. feature analysis)**: Der analytische Prozess, bei dem bereits erstellte oder vorhandene Merkmale untersucht, beschrieben und bewertet werden. Merkmalsanalyse basiert auf Merkmalsbeschreibungen und erfolgt typischerweise in der Datennachverarbeitung.


Datenauswertung (engl. data analysis)
Datenpräsentation  (engl. data presentation)
<div style="margin-left: 2em;">
  <code>???</code>&nbsp;<code>???</code><br>
</div>

#### 🔵 Merkmalsauswertungen (engl. feature )

<ol>
    <details>
      <summary>🔵 Aggregation (engl. aggregation)</summary>
      <p><i>alphanumerische Darstellungen - Aggregation reduziert die Datenmenge durch mathematische Operationen wie Summe, Mittelwert, Zählung oder Maximum über Gruppierungen (z. B. nach Token-Typ, Dokument oder Zeitraum). In NLP könnte dies die Häufigkeitsverteilung von n-Grammen pro Domäne oder die durchschnittliche Embedding-Distanz pro Klasse bedeuten. Sie erfolgt vor der Visualisierung, um Überladung zu vermeiden, und ist rein datenverarbeitend ohne grafische Elemente. Aggregation fasst Rohdaten zu kompakteren Zusammenfassungen zusammen.</i></p>
      <div style="margin-left: 2em;">
        <code>???</code>&nbsp;<code>????</code><br><br>
      </div>
    <ol>
    </details>
    <details>
      <summary>🔵 Visualisierung (engl. visualization)</summary>
      <p><i>grafische Darstellung - Visualisierung stellt die aggregierten Daten grafisch dar, um Muster erkennbar zu machen.</i></p>

      Themenverteilungen; Top-Wörter pro Thema


      <div style="margin-left: 2em;">
        <code>???</code>&nbsp;<code>????</code><br><br>
      </div>
    </details>
  </li>
</ol>

Topic Diversity:
`library 3`

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
├─────────────── / complaints_data.csv
├─────────────── / complaints_data_cleaned.csv
├── src/       # Python-Module
├─────────────── / Datensatzvorverarbeitung (engl. dataset pipeline)
├─────────────── / Datenverarbeitung (engl. data processing)
├─────────────── / Datennachverarbeitung (engl. data post-processing)
├── docs/      # Übersichten (Datensatzauswertungen, Pipeline, Installation)
├─────────────── / 1 -
├─────────────── / 2 - 
├─────────────── / 3 - 
├─────────────── / X - 
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

