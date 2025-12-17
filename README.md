# Projekt: Advanced Data Analysis (DLBDSEDA02)
Projekt: Advanced Data Analysis (DLBDSEDA02)
Zunächst wird manuell nach Datensätzen in deutscher und englischer Sprache recherchiert und offensichtlich synthetisch erzeugte Varianten aussortiert. 

## # Vorprüfung
Anhand eines KI-Detektors werden die übrigen Datensätze auf synthetisch erzeugte Instanzen (engl. samples) geprüft und bewertet. Anhand von Kriterien wie dem vermuteten Anteil realer oder synthetischer Samples sowie den nicht durch das Modell verarbeitbaren Samples (Real/Fake/Error) wird der Datensatz mit dem geringsten Real/Fake-Quotienten gewählt, da hier die Wahrscheinlichkeit eines organischen Ursprungs am höchsten erscheint. Der so identifizierte Datensatz geht in die NLP-Pipeline, welche mit der Textvorverarbeitung (engl. text pre-processing) des Datensatzes beginnt. Übersteigt die Anzahl der Samples eine Schwelle von 2000 wird der Datensatz zunächst aus Performancegründen gesplittet.
