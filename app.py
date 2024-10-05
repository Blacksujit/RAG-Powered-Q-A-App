from flask import Flask, render_template, request, jsonify
from qa_system import get_answer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    if question:
        answer, documents = get_answer(question)
        return jsonify({'answer': answer, 'documents': documents})
    return jsonify({'error': 'No question provided!'})

if __name__ == "__main__":
    app.run(debug=True)
