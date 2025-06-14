# SentenceDiagrammer

A web-based tool to create dependency diagrams and Reed-Kellogg diagrams of sentences.

## Requirements

Install GraphViz from their website: https://graphviz.org/download/

After installing the dependencies from `requirements.txt`, use the following command to install the language database:
```
pip install $(spacy info en_core_web_sm --url)
```
