from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile
import re
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import heapq

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')

# Download NLTK resources (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# In-memory storage (in a real app, you'd use a database)
documents = {}
current_document_id = None


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/upload', methods=['POST'])
def upload_document():
    global current_document_id

    if 'document' not in request.files:
        return jsonify({"error": "No document part"}), 400

    file = request.files['document']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Create a unique ID for this document
        document_id = str(len(documents) + 1)
        current_document_id = document_id

        # Process the document based on file extension
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        try:
            # Extract text based on file type
            if file_extension == 'pdf':
                text = extract_text_from_pdf(file)
            elif file_extension == 'docx':
                text = extract_text_from_docx(file)
            elif file_extension == 'txt':
                text = file.read().decode('utf-8', errors='replace')
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            # Generate summary with fallback method
            try:
                summary = generate_summary(text)
            except Exception as e:
                # Fallback to simple summarization if advanced method fails
                print(f"Advanced summarization failed: {str(e)}")
                summary = simple_summarize(text)

            # Store document data
            documents[document_id] = {
                "filename": filename,
                "text": text,
                "summary": summary
            }

            return jsonify({
                "document_id": document_id,
                "filename": filename,
                "summary": summary
            })

        except Exception as e:
            print(f"Document processing error: {str(e)}")
            return jsonify({"error": f"File processing failed: {str(e)}"}), 500

    return jsonify({"error": "File processing failed"}), 500


@app.route('/query', methods=['POST'])
def query_document():
    global current_document_id

    if not current_document_id or current_document_id not in documents:
        return jsonify({"error": "No document has been uploaded yet"}), 400

    try:
        data = request.json
        if 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data['query']
        document_text = documents[current_document_id]['text']

        # Process the query and generate a response
        response = answer_query(query, document_text)

        return jsonify({"response": response})
    except Exception as e:
        print(f"Query processing error: {str(e)}")
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500


def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
        file.save(temp.name)

    text = ""
    try:
        with open(temp.name, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:  # Check if extract_text() returned something
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {str(e)}")
        raise
    finally:
        # Clean up temp file
        try:
            os.unlink(temp.name)
        except:
            pass

    # If no text was extracted, raise an error
    if not text.strip():
        raise Exception("No text could be extracted from the PDF file")

    return text


def extract_text_from_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp:
        file.save(temp.name)

    text = ""
    try:
        doc = docx.Document(temp.name)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
    except Exception as e:
        print(f"DOCX extraction error: {str(e)}")
        raise
    finally:
        # Clean up temp file
        try:
            os.unlink(temp.name)
        except:
            pass

    # If no text was extracted, raise an error
    if not text.strip():
        raise Exception("No text could be extracted from the DOCX file")

    return text


def simple_summarize(text, num_sentences=5):
    """A simple fallback summarization method that extracts the first few sentences"""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    return " ".join(sentences[:num_sentences])


def generate_summary(text, num_sentences=5):
    # Check if text is empty or too short
    if not text or len(text) < 100:
        return text

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # If there are fewer sentences than requested summary length, return all sentences
    if len(sentences) <= num_sentences:
        return text

    # Remove stop words and calculate word frequencies
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word.isalnum() and word not in stop_words:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    # Normalize word frequencies
    if word_frequencies:  # Check if any valid words were found
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = word_frequencies[word]
                else:
                    sentence_scores[i] += word_frequencies[word]

    # Get top N sentences with highest scores
    if not sentence_scores:  # If no sentences were scored
        return simple_summarize(text, num_sentences)

    summary_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_indices.sort()  # Sort to maintain original order

    summary = " ".join([sentences[i] for i in summary_indices])
    return summary


def answer_query(query, document_text):
    # Check if document text is empty
    if not document_text or not document_text.strip():
        return "The document appears to be empty or contains no readable text."

    # Convert query to lowercase and remove punctuation
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)

    # Tokenize document text into sentences
    sentences = sent_tokenize(document_text)

    # Calculate relevance scores for each sentence
    scores = []
    query_words = [word.lower() for word in word_tokenize(query) if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    query_words = [word for word in query_words if word not in stop_words]

    # If no valid query words after filtering, use all words
    if not query_words:
        query_words = [word.lower() for word in word_tokenize(query) if word.isalnum()]

    for sentence in sentences:
        sentence_words = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]

        # Count matching words
        match_count = sum(1 for word in query_words if word in sentence_words)

        # Calculate score as percentage of query words found in sentence
        if query_words:
            score = match_count / len(query_words)
        else:
            score = 0

        scores.append((score, sentence))

    # Sort sentences by score
    sorted_sentences = sorted(scores, key=lambda x: x[0], reverse=True)

    # Get top 3 most relevant sentences
    top_sentences = [sentence for _, sentence in sorted_sentences[:3] if _ > 0]

    if not top_sentences:
        # If no relevant sentences found, provide a helpful message
        return "I couldn't find specific information related to your query in the document. Try asking a different question or rewording your query."

    # Format response
    response = " ".join(top_sentences)

    return response


if __name__ == '__main__':
    app.run(debug=True)