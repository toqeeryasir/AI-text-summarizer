from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import gradio as gr
from time import time
import re

class SummarizerTxt:
    def __init__(self):
        model_name = "google/pegasus-xsum"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.chunk_size = 512
        self.max_length = 256

    def split_into_sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text):
        if not text.strip():
            return []

        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer(sentence, return_tensors="pt", truncation=False)["input_ids"][0]
            sentence_length = len(sentence_tokens)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def summarize(self, text):
        if not text.strip():
            return "Please enter some text to summarize."

        chunks = self.chunk_text(text)
        
        if not chunks:
            return "Text is too short to summarize."

        summaries = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                truncation=True,
                padding="longest",
                max_length=self.chunk_size,
                return_tensors="pt"
            ).to(self.device)

            summary_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=4,
                max_length=self.max_length,
                min_length=32,
                early_stopping=True,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        return " ".join(summaries)

summarizer = SummarizerTxt()

def summarize_text(input_text):
    start = time()
    result = summarizer.summarize(input_text)
    end = time()
    time_taken = end - start
    return result, f"{time_taken:.0f} seconds"
    
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter text to summarize...",
        lines=5
    ),
    outputs=[
        gr.Textbox(label="Summary", lines=3),
        gr.Textbox(label="Time taken",
            placeholder="time...")
    ],
    title="Text Summarizer",
    description="Summarize long texts using Google's Pegasus model",
    examples=[
        ["The Apollo program was NASA's third human spaceflight program, which successfully landed humans on the Moon between 1969 and 1972."],
        ["Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."]
    ]
)

if __name__ == "__main__":
    iface.launch(debug=True)
