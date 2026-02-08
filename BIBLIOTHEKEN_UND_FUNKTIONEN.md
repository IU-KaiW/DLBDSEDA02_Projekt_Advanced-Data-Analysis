# Übersicht der im Projekt verwendeten Bibliotheken und Funktionen

## Zusammenfassung
Dieses Dokument enthält eine systematische Auflistung aller im Projekt "Advanced Data Analysis (ADA)" verwendeten Python-Bibliotheken und ihre konkret genutzten Funktionen.

---

## 1. Datenverarbeitung & Strukturen

### 1.1 **pandas** (≥1.3.0)
**Zweck:** Datenmanipulation, DataFrame-Operationen, CSV-Verarbeitung

| Funktion | Verwendung |
|----------|-----------|
| `pd.read_csv()` | Datensatz aus CSV-Dateien laden |
| `df.head()` | Erste Zeilen anzeigen |
| `df.tail()` | Letzte Zeilen anzeigen |
| `df.shape` | Dimensionen (Zeilen, Spalten) ermitteln |
| `df.columns` | Spalten auflisten |
| `df[['col1', 'col2']]` | Spalten selektieren |
| `df['col'].str.extract()` | Regex-basiertes Extrahieren aus Strings |
| `df['col'].str.len()` | Textlänge berechnen |
| `df['col'].str.lower()` | Text in Kleinbuchstaben |
| `df.dropna()` | Zeilen mit NaN entfernen |
| `df[df.isna()]` | Fehlwerte filtern |
| `df['col'].isin()` | Membership-Test |
| `df['col'].value_counts()` | Häufigkeitsverteilung |
| `df['col'].unique()` | Eindeutige Werte |
| `pd.to_datetime()` | String zu DateTime konvertieren |
| `df['date'].dt.year` | Jahr aus DateTime extrahieren |
| `df['date'].dt.month` | Monat aus DateTime extrahieren |
| `df['date'].dt.day` | Tag aus DateTime extrahieren |
| `df['date'].dt.month_name()` | Monatsname |
| `df['date'].dt.day_name()` | Wochentag |
| `df['date'].dt.to_period()` | Zeitraum extrahieren |
| `pd.Categorical()` | Kategoriale Variablen erstellen |
| `df.sort_values()` | Sortieren |
| `df.reset_index(drop=True)` | Index zurücksetzen |
| `df.astype()` | Datentyp konvertieren |
| `df.to_string()` | DataFrame zu String konvertieren |
| `pd.Series()` | Serie aus Liste erstellen |

---

### 1.2 **numpy** (≥1.21.0)
**Zweck:** Numerische Berechnungen und Array-Operationen

| Funktion | Verwendung |
|----------|-----------|
| `np.array()` | NumPy-Arrays erstellen |
| `np.zeros()` | Array mit Nullen |
| `np.ones()` | Array mit Einsen |
| `np.arange()` | Array mit Wertebereich |
| `np.linspace()` | Linear verteilte Werte |
| `np.mean()` | Durchschnitt berechnen |
| `np.std()` | Standardabweichung |
| `np.min() / np.max()` | Min/Max-Werte |
| `np.reshape()` | Array umformen |
| `np.where()` | Bedingte Indizierung |

---

## 2. Visualisierung

### 2.1 **matplotlib** (≥3.4.0)
**Zweck:** Statische 2D-Visualisierungen und Plots

| Funktion | Verwendung |
|----------|-----------|
| `plt.figure()` | Neue Figure erstellen |
| `plt.plot()` | Linienplot |
| `plt.bar()` / `plt.barh()` | Balkendiagramm |
| `plt.hist()` | Histogramm |
| `plt.scatter()` | Streudiagramm |
| `plt.title()` | Titel hinzufügen |
| `plt.xlabel()` | X-Achsen-Label |
| `plt.ylabel()` | Y-Achsen-Label |
| `plt.legend()` | Legende anzeigen |
| `plt.show()` | Plot anzeigen |
| `plt.subplots()` | Mehrere Subplots |
| `plt.savefig()` | Plot speichern |

---

### 2.2 **seaborn** (≥0.11.0)
**Zweck:** Statistische Visualisierungen und Themes

| Funktion | Verwendung |
|----------|-----------|
| `sns.set_style()` | Plot-Stil setzen |
| `sns.set_palette()` | Farbpalette |
| `sns.barplot()` | Statistisches Balkendiagramm |
| `sns.boxplot()` | Box-Plot |
| `sns.heatmap()` | Heatmap-Visualisierung |
| `sns.histplot()` | Histogramm |
| `sns.countplot()` | Häufigkeitsplot |
| `sns.scatterplot()` | Streudiagramm |
| `sns.lineplot()` | Linienplot |
| `sns.pairplot()` | Paar-Plot Matrix |

---

### 2.3 **plotly** (≥4.7.0)
**Zweck:** Interaktive 3D- und Choroplethenkarten

| Funktion | Verwendung |
|----------|-----------|
| `plotly.graph_objects.Figure()` | Interaktive Figure |
| `go.Choropleth()` | Choroplethenkarte (geografisch) |
| `go.Scatter()` | Interaktiver Scatter-Plot |
| `go.Bar()` | Interaktives Balkendiagramm |
| `fig.show()` | Interaktive Visualisierung anzeigen |
| `fig.to_html()` | In HTML exportieren |

---

### 2.4 **wordcloud** (≥1.8.0)
**Zweck:** Wortwolken-Visualisierung

| Funktion | Verwendung |
|----------|-----------|
| `WordCloud()` | Wortwolke erstellen |
| `wc.generate()` | Text in Wortwolke konvertieren |
| `wc.to_image()` | Als Bild exportieren |

---

### 2.5 **cartopy** (≥0.20.0)
**Zweck:** Geografische Kartendarstellung

| Funktion | Verwendung |
|----------|-----------|
| `cartopy.crs` | Kartenprojektionen |
| `ax.coastlines()` | Küstenlinien zeichnen |
| `ax.stock_img()` | Hintergrund-Kartenbild |
| `ax.add_feature()` | Geografische Features |

---

### 2.6 **pyLDAvis** (≥3.3.0)
**Zweck:** Interaktive LDA-Visualisierung

| Funktion | Verwendung |
|----------|-----------|
| `pyLDAvis.gensim()` | LDA-Modell visualisieren |
| `pyLDAvis.display()` | Anzeigen |

---

## 3. Machine Learning & NLP

### 3.1 **scikit-learn** (≥1.0.0)
**Zweck:** Machine Learning Algorithmen und Metriken

| Funktion/Klasse | Verwendung |
|----------|-----------|
| **Text-Vektorisierung** | |
| `CountVectorizer()` | Bag-of-Words Vektorisierung |
| `TfidfVectorizer()` | TF-IDF Vektorisierung |
| `vectorizer.fit_transform()` | Fit & Transform |
| `vectorizer.get_feature_names_out()` | Vokabular extrahieren |
| **Dimensionalitätsreduktion** | |
| `PCA()` | Principal Component Analysis |
| `TSNE()` | t-Distributed Stochastic Neighbor Embedding |
| `fit_transform()` | PCA/tSNE trasformieren |
| **Clustering** | |
| `KMeans()` | K-Means Clustering |
| `fit()` | Modell fitting |
| `labels_` | Cluster-Labels |
| `cluster_centers_` | Clusterzentren |
| **Metriken** | |
| `silhouette_score()` | Silhouette-Koeffizient |
| `davies_bouldin_score()` | Davies-Bouldin Index |
| `calinski_harabasz_score()` | Calinski-Harabasz Index |

---

### 3.2 **spaCy** (≥3.7.0)
**Zweck:** NLP-Pipeline mit Tokenisierung und Lemmatisierung

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `spacy.load()` | Sprachmodell laden (z.B. "en_core_web_sm") |
| `nlp()` | Text verarbeiten |
| `nlp.pipe()` | Batch-Processing (schneller) |
| `doc` | Verarbeitete Texte |
| `token.lemma_` | Lemma (Grundform) |
| `token.text` | Token-Text |
| `token.is_stop` | Stopwort-Flag |
| `token.is_punct` | Satzzeichen-Flag |
| `token.like_num` | Nummern-Flag |
| `token.is_ascii` | ASCII-Flag |
| `token.pos_` | Part-of-Speech Tag |

**Modelle:**
- `en_core_web_sm-3.7.1` - Kleines englisches Modell

---

### 3.3 **gensim** (≥4.1.0)
**Zweck:** Topic-Modeling (LDA, Word2Vec)

| Funktion/Klasse | Verwendung |
|----------|-----------|
| **LDA (Latent Dirichlet Allocation)** | |
| `Dictionary()` | Vokabular-Dictionary |
| `Dictionary.doc2bow()` | Text zu Bag-of-Words |
| `LdaModel()` | LDA-Modell erstellen |
| `lda_model.get_topics()` | Topics extrahieren |
| `lda_model.get_document_topics()` | Dokument-Topic-Verteilung |
| **Word2Vec** | |
| `Word2Vec()` | Word2Vec Embeddings |
| `model.wv['word']` | Word-Vector |

---

### 3.4 **BERTopic** (≥0.14.0)
**Zweck:** Transformer-basiertes Topic-Modeling

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `BERTopic()` | BERTopic-Modell initialisieren |
| `fit_transform()` | Modell trainieren & Topics extrahieren |
| `get_topic_info()` | Topic-Informationen |
| `visualize_topics()` | Topics visualisieren |
| `get_document_info()` | Dokument-Informationen |

---

### 3.5 **Top2Vec** (≥1.0.0)
**Zweck:** Automatisches Topic Discovery

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `Top2Vec()` | Top2Vec-Modell |
| `fit()` | Modell trainieren |
| `get_topics()` | Topics extrahieren |
| `search_documents_by_topic()` | Dokumente nach Topic suchen |

---

### 3.6 **nltk** (≥3.6.0)
**Zweck:** Klassische NLP-Tools (Tokenisierung, POS-Tagging)

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `nltk.download()` | Ressourcen herunterladen |
| `nltk.tokenize.word_tokenize()` | Wort-Tokenisierung |
| `nltk.tokenize.sent_tokenize()` | Satz-Tokenisierung |
| `nltk.pos_tag()` | Part-of-Speech Tagging |
| `nltk.corpus.stopwords.words()` | Stopwort-Liste |
| `PorterStemmer()` | Porter Stemming |
| `WordNetLemmatizer()` | Lemmatisierung |

---

### 3.7 **transformers** (≥4.20.0)
**Zweck:** State-of-the-Art Sprachmodelle (BERT, GPT, etc.)

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `AutoTokenizer.from_pretrained()` | Tokenizer laden |
| `AutoModel.from_pretrained()` | Transformer-Modell laden |
| `pipeline()` | NLP-Task Pipelines |
| `TextClassificationPipeline()` | Text-Klassifikation |

---

### 3.8 **torch / PyTorch** (≥1.10.0)
**Zweck:** Deep Learning Framework

| Komponente | Verwendung |
|----------|-----------|
| `torch.Tensor` | Tensor-Datenstruktur |
| `torch.nn` | Neuronale Netze |
| `torch.optim` | Optimierer |
| `torch.cuda` | GPU-Unterstützung |
| `torchvision` | Computer Vision |
| `torchaudio` | Audio Processing |

---

### 3.9 **sentence-transformers** (≥0.4.1)
**Zweck:** Sätze zu Embeddings konvertieren

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `SentenceTransformer()` | Modell laden |
| `encode()` | Sätze encodieren |
| `similarity()` | Ähnlichkeit berechnen |

---

### 3.10 **umap-learn** (≥0.5.0)
**Zweck:** Dimensionalitätsreduktion (Alternative zu t-SNE)

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `UMAP()` | UMAP-Reduzierer |
| `fit_transform()` | Fit & Reduktion |

---

### 3.11 **hdbscan** (≥0.8.29)
**Zweck:** Hierarchisches Density-Based Clustering

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `HDBSCAN()` | Clustering-Algorithmus |
| `fit_predict()` | Cluster-Labels |

---

## 4. Geografische Daten

### 4.1 **geopandas** (≥0.10.0)
**Zweck:** Geografische DataFrames

| Funktion/Klasse | Verwendung |
|----------|-----------|
| `gpd.GeoDataFrame()` | Geografischer DataFrame |
| `gpd.read_file()` | Shape-Dateien laden |
| `GeoDataFrame.plot()` | Geografische Visualisierung |

---

## 5. Hilfsbibliotheken

### 5.1 **IPython** (≥7.0.0)
**Zweck:** Interactive Python Shell & Jupyter-Integration

| Funktion | Verwendung |
|----------|-----------|
| `%pip install` | Magic: Paket-Installation |
| `%matplotlib inline` | Magic: Matplotlib inline Plots |
| `display()` | Jupyter-Ausgabe |

---

### 5.2 **regex (re)** - Python Standard
**Zweck:** Reguläre Ausdrücke

| Funktion | Verwendung |
|----------|-----------|
| `re.search()` | Muster suchen |
| `re.findall()` | Alle Matches finden |
| `re.sub()` | Muster ersetzen |
| `re.match()` | Anfang des Strings |
| `pattern.extract()` | pandas + Regex |

---

### 5.3 **csv** - Python Standard
**Zweck:** CSV-Dateiverarbeitung

| Funktion | Verwendung |
|----------|-----------|
| `csv.reader()` | CSV lesen |
| `csv.DictReader()` | CSV als Dictionary |

---

### 5.4 **Sonstige**
| Bibliothek | Verwendung |
|----------|-----------|
| `venv` | Virtual Environment Management |
| `tqdm` | Progress Bars |
| `tenacity` | Retry-Mechanismen |
| `packaging` | Package-Versioning |
| `threadpoolctl` | Thread-Pool Kontrolle |

---

## 6. Projektstruktur & Verwendung

### Notebooks nach Verwendung:

| Notebook | Hauptbibliotheken |
|----------|----------|
| **0 - Installationen.ipynb** | pandas, numpy, matplotlib, seaborn, spacy, nltk, sklearn, gensim, transformers, torch |
| **1.1 - Datensatzakquisition** | pandas, requests, urllib |
| **1.2 - Datensatzsichtung** | pandas, matplotlib, seaborn, re, plotly |
| **2 - Datenverarbeitung** | pandas, spacy, sklearn, gensim, bertopic, top2vec, nltk, transformers |
| **3 - Datennachverarbeitung** | pandas, matplotlib, plotly, wordcloud, cartopy, geopandas |

---

## 7. Wichtige Funktionskombinationen

### Text-Verarbeitung Pipeline:
1. `pd.read_csv()` - Daten laden
2. `nlp.pipe()` - Spacy-Verarbeitung
3. `df['cleaned']` - Bereinigte Tokens
4. `CountVectorizer()` oder `TfidfVectorizer()` - Vektorisierung
5. `LdaModel()` / `BERTopic()` - Topic-Modeling
6. `pyLDAvis.gensim()` - Visualisierung

### Visualisierungs-Pipeline:
1. `df['col'].value_counts()` - Häufigkeiten
2. `plt.bar()` / `sns.barplot()` - Plotting
3. `plt.title(), plt.xlabel()` - Labels
4. `plt.show()` - Anzeige

---

## 8. Abhängigkeiten & Versionen

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
transformers>=4.20.0
torch>=1.10.0
spacy>=3.7.0
gensim>=4.1.0
bertopic>=0.14.0
top2vec>=1.0.0
nltk>=3.6.0
wordcloud>=1.8.0
cartopy>=0.20.0
pyLDAvis>=3.3.0
geopandas>=0.10.0
plotly>=4.7.0
umap-learn>=0.5.0
hdbscan>=0.8.29
sentence-transformers>=0.4.1
IPython>=7.0.0
```

---

**Dokumentation erstellt:** Februar 8, 2026  
**Projekt:** DLBDSEDA02 Advanced Data Analysis (ADA)
