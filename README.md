# SentenceDiagrammer

SentenceDiagrammer is a web-based tool for generating **dependency diagrams** and **Reed-Kellogg diagrams** of sentences. It leverages natural language processing (NLP) with spaCy and benepar to parse sentences and visualize their grammatical structure using GraphViz and Matplotlib, making it a valuable resource for linguists, educators, and language enthusiasts.

This app was developed using Grok's automatic code generation and manual review and corrections. A full transcript of the prompts and replies can be found [here](https://grok.com/share/c2hhcmQtMg%3D%3D_af5ea4ca-65d7-4f06-8e07-ee5a016a3f5f).

## Current Features
- **Dependency Diagrams**: Illustrate syntactic dependencies between words in a sentence.
- **Reed-Kellogg Diagrams**: Display traditional sentence diagrams showing grammatical components.
- Built with Flask for an accessible web interface.
- Supports declarative sentence structures with adjective and adverb modifiers, prepositional phrases, and partial support for coordinated and dependent clauses and coordinated nouns and verbs.

## Future Enhancements
- Additional sentence types/structures, including imperative, interrogative, and exclamatory sentences.
- Better spacing and structure for the generated graphs.
- More complete handling of all coordinating conjunction structures.
- Better error handling and sentence validation.

## Installation

To set up SentenceDiagrammer, you need to install GraphViz, Python dependencies, and the spaCy language model.

### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Steps
1. **Install GraphViz**:
   - Download and install GraphViz from the official website: [https://graphviz.org/download/](https://graphviz.org/download/)
   - Add GraphViz to your system's PATH during installation to enable diagram rendering.

2. **Clone the Repository**:
   - Clone or download the project:
     ```
     git clone https://github.com/Jed-Schaaf/SentenceDiagrammer.git
     cd SentenceDiagrammer
     ```

3. **Install Python Dependencies**:
   - Install the required packages listed in `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```
   - Install the spaCy English language model:
     ```
     pip install $(spacy info en_core_web_sm --url)
     ```

4. **Install Benepar Model**:
   - The project uses benepar for parsing. If not already installed via `requirements.txt`, download it:
     ```
     python -m benepar.download benepar_en3
     ```

## Usage

### Running the Application
1. Start the Flask app from the project directory:
   ```
   python app.py
   ```
2. Open a web browser and navigate to `http://127.0.0.1:5000/`.

### Generating Diagrams
- **Input**: Enter a sentence in the provided text area (e.g., "The quick brown fox jumps over the lazy dog").
- **Style**: Select a diagram style:
  - "dependency" for a dependency diagram.
  - "reed-kellogg" for a Reed-Kellogg diagram.
- **Action**: Click "Generate Diagrams" to view the output.
- **Output**: The tool processes the sentence and displays the diagram as an SVG image.

### Notes
- Empty or invalid input will return an error message: "Please enter at least one valid sentence."
- Errors during diagram generation (e.g., parsing issues) will be displayed per sentence.

## Examples

- **Dependency Diagram**:
  - Input: "She baked me a cake."
  - Output: A graph showing "baked" as the root, with dependencies like "She" (subject), "me" (indirect object), and "cake" (direct object).
- **Reed-Kellogg Diagram**:
  - Input: "The very big red cat quickly chased the small gray mouse."
  - Output: A traditional diagram with "cat" and "chased" on the main line, modifiers like "very big red" slanting below "cat," and "mouse" as the object with its modifiers.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE). See the [LICENSE](LICENSE) file for full details.

## Contributing

We welcome contributions to enhance SentenceDiagrammer! To contribute:
- **Report Issues**: Submit bugs or suggestions via the [GitHub Issues page](https://github.com/Jed-Schaaf/SentenceDiagrammer/issues).
- **Submit Pull Requests**: Fork the repository, make improvements, and submit a pull request to [https://github.com/Jed-Schaaf/SentenceDiagrammer](https://github.com/Jed-Schaaf/SentenceDiagrammer).
- **Guidelines**: Ensure code follows Python PEP 8 standards and include comments for clarity.

### Development Notes
- The project uses `nlp_module.py` for sentence parsing with spaCy and benepar.
- `diagrammer.py` handles diagram generation, with extensive logic for Reed-Kellogg diagrams.
- Test your app changes locally with `python app.py` before submitting.
  - Check diagramming updates with `python diagrammer.py`. Example sentences are included at the bottom of the file along with test code.

## Troubleshooting
- **GraphViz Not Found**: Ensure GraphViz is installed and in your PATH.
- **Module Errors**: Verify all dependencies are installed correctly from `requirements.txt`.
- **Diagram Issues**: Complex sentences may require further development or debugging in `diagrammer.py`.