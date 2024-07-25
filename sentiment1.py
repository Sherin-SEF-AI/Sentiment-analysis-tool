import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Initialize sentiment analysis
analyzer = SentimentIntensityAnalyzer()

# Sentiment Analysis Function
def analyze_sentiment_vader(text):
    try:
        scores = analyzer.polarity_scores(text)
        return scores
    except Exception as e:
        return None, str(e)

def analyze_sentiment_textblob(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment.polarity, sentiment.subjectivity
    except Exception as e:
        return None, str(e)

# Function to process file content based on file type
def process_file(file_path, file_type):
    content = ""
    if file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    elif file_type == "csv":
        df = pd.read_csv(file_path)
        content = df.to_string()
    elif file_type == "pdf":
        pdf_document = fitz.open(file_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            content += page.get_text()
    return content

# Function to visualize sentiment analysis results
class SentimentVisualizer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, vader_scores, polarity, subjectivity):
        labels = ['Positive', 'Neutral', 'Negative', 'Polarity', 'Subjectivity']
        vader_data = [vader_scores['pos'], vader_scores['neu'], vader_scores['neg'], polarity, subjectivity]

        self.ax.clear()
        self.ax.bar(labels, vader_data, color=['green', 'blue', 'red', 'purple', 'orange'])
        self.ax.set_ylabel('Scores')
        self.ax.set_title('Sentiment Analysis Results')
        self.ax.set_ylim(0, 1)
        self.draw()

# GUI Application
class SentimentAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.results = ""
        self.vader_scores = None
        self.polarity = None
        self.subjectivity = None
        self.file_content = ""

    def initUI(self):
        self.setWindowTitle("Sentiment Analysis")

        self.text_entry = QTextEdit(self)
        self.text_entry.setPlaceholderText("Enter text or upload a file to analyze sentiment")
        
        self.result_label = QLabel("", self)
        
        self.analyze_button = QPushButton("Analyze Sentiment", self)
        self.analyze_button.clicked.connect(self.perform_analysis)
        
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_text)
        
        self.upload_button = QPushButton("Upload File", self)
        self.upload_button.clicked.connect(self.upload_file)
        
        self.save_button = QPushButton("Save Results", self)
        self.save_button.clicked.connect(self.save_results)
        
        self.visualizer = SentimentVisualizer(self)

        layout = QVBoxLayout()
        layout.addWidget(self.text_entry)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.visualizer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def perform_analysis(self):
        text = self.text_entry.toPlainText().strip() or self.file_content
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text or upload a file to analyze.")
            return

        # Perform VADER sentiment analysis
        self.vader_scores = analyze_sentiment_vader(text)
        if self.vader_scores:
            overall_sentiment = max(self.vader_scores, key=self.vader_scores.get)
            result_text = f"Overall Sentiment (VADER): {overall_sentiment}\n"
            result_text += f"Scores: {self.vader_scores}\n"
        else:
            result_text = "Error analyzing sentiment with VADER.\n"

        # Perform TextBlob sentiment analysis
        self.polarity, self.subjectivity = analyze_sentiment_textblob(text)
        if self.polarity is not None:
            sentiment = "Positive" if self.polarity > 0 else "Negative" if self.polarity < 0 else "Neutral"
            result_text += f"Overall Sentiment (TextBlob): {sentiment}\n"
            result_text += f"Polarity: {self.polarity:.4f}\nSubjectivity: {self.subjectivity:.4f}"
        else:
            result_text += "Error analyzing sentiment with TextBlob."

        self.results = result_text
        self.result_label.setText(result_text)

        # Update visualization
        self.visualizer.plot(self.vader_scores, self.polarity, self.subjectivity)

    def clear_text(self):
        self.text_entry.clear()
        self.result_label.setText("")
        self.results = ""
        self.vader_scores = None
        self.polarity = None
        self.subjectivity = None
        self.file_content = ""
        self.visualizer.ax.clear()
        self.visualizer.draw()

    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "Text Files (*.txt);;CSV Files (*.csv);;PDF Files (*.pdf)", options=options)
        if file_path:
            file_type = file_path.split('.')[-1]
            self.file_content = process_file(file_path, file_type)
            self.text_entry.setPlainText(self.file_content)

    def save_results(self):
        if self.results:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Text Files (*.txt)", options=options)
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.results)
                QMessageBox.information(self, "Save Results", "Results saved successfully!")
        else:
            QMessageBox.warning(self, "No Results", "No results to save. Perform analysis first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = SentimentAnalysisApp()
    mainWin.show()
    sys.exit(app.exec_())

