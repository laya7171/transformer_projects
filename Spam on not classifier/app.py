import gradio as gr
import torch

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Spam" if prediction == 1 else "Ham"

iface = gr.Interface(fn=classify, inputs="text", outputs="text", title="Spam Classifier")
iface.launch()
