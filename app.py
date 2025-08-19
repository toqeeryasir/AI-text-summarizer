from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import gradio as gr

class SummarizerTxt:
    def __init__(self):
        model_name = "google/pegasus-xsum"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.chunk_size = 512

    def summarize(self, text):
        if not text.strip():
            return "Please enter some text to summarize."
            
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(self.tokenizer(current_chunk + sentence)["input_ids"]) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

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
                **inputs,
                num_beams=4,
                max_length=100,
                early_stopping=True
            )
            summaries.append(self.tokenizer.decode(summary_ids[0], skip_special_tokens=True))

        return " ".join(summaries)

summarizer = SummarizerTxt()

def summarize_text(input_text):
    return summarizer.summarize(input_text)

iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        label="Input Text", 
        placeholder="Enter text to summarize...",
        lines=5
    ),
    outputs=gr.Textbox(
        label="Summary",
        lines=3
    ),
    title="Text Summarizer",
    description="Summarize long texts using Google's Pegasus model",
    examples=[
        ["The Apollo program was NASA's third human spaceflight program, which successfully landed humans on the Moon between 1969 and 1972."],
        ["Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."]
    ]
)

iface.launch()