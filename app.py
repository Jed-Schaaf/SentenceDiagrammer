from flask import Flask, render_template, request
from nlp_module import split_sentences
from diagrammer import generate_diagram
from error_handling import check_input

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form.get('sentence')
        style = request.form.get('style', 'dependency')  # Default to dependency tree
        # Validate input
        error = check_input(input_text)
        if error:
            return render_template('index.html', results=[{'sentence': input_text, 'error': error}])
        sentences = split_sentences(input_text)
        results = []
        for sentence in sentences:
            try:
                diagram = generate_diagram(sentence, style)
                results.append({'sentence': sentence, 'diagram': diagram})
            except Exception as e:
                results.append({'sentence': sentence, 'error': str(e)})
        return render_template('index.html', results=results, input_text=input_text, style=style)
    # Render the form for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)