from flask import Flask, render_template, request
from diagrammer import generate_diagram
from error_handling import check_input

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form.get('sentence').strip()
        style = request.form.get('style', 'dependency')  # Default to dependency tree

        # Validate input
        error = check_input(sentence)
        if error:
            return render_template('index.html', error=error)

        try:
            # Parse sentence and generate diagram
            diagram_svg = generate_diagram(sentence, style)
            return render_template('index.html', diagram=diagram_svg, sentence=sentence, style=style)
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    # Render the form for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)