import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Model Configuration
# Ensure you have downloaded/saved your model files into this folder
MODEL_PATH = "./model_weights" 

try:
    # Load the optimized model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    print(f"Error: Could not load model from {MODEL_PATH}. Check your folder path.")
    print(f"Details: {e}")

label_map = {0: "Negative 🔴", 1: "Neutral 🟡", 2: "Positive 🟢"}

# 2. Inference Logic
def predict_service(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    idx = np.argmax(probs)
    conf = float(probs[idx])
    
    # Calculate Margin (Top 1 vs Top 2)
    sorted_p = sorted(probs, reverse=True)
    margin = sorted_p[0] - sorted_p[1]

    # Logic for Borderline Detection
    is_borderline = margin < 0.23 or conf < 0.60
    status = "⚠️ Borderline - Review Needed" if is_borderline else "✅ High Confidence"

    # Formatting output for Gradio
    res_dict = {label_map[i]: float(probs[i]) for i in range(3)}
    
    # Custom Markdown Report
    report = (
        f"### Analysis Result\n"
        f"- **Primary Decision**: {label_map[idx]}\n"
        f"- **System Status**: {status}\n"
        f"- **Certainty Margin**: {margin:.2f} (Confidence: {conf:.2%})"
    )
    return res_dict, report

# 3. Gradio Interface Design
demo = gr.Interface(
    fn=predict_service,
    inputs=gr.Textbox(
        lines=3, 
        label="Financial News Input", 
        placeholder="Type or paste financial news here..."
    ),
    outputs=[
        gr.Label(label="Sentiment Probability Distribution"), 
        gr.Markdown(label="Decision Report")
    ],
    title="🏦 FinSense-TAPT: Calibrated Sentiment Engine",
    description=(
        "This system utilizes a **TAPT-optimized BERT** model specifically trained for financial semantics. "
        "It includes a **Confidence-Margin** system to flag ambiguous or 'hedged' statements for manual review."
    ),
    examples=[
        ["The tech giant reported a 10% increase in quarterly revenue, but warned that global supply chain disruptions could significantly impact profit margins in the coming months."],
        ["NVIDIA reports record-breaking revenue fueled by massive AI infrastructure spending."],
        ["The company filed for bankruptcy after a catastrophic fraud investigation."],
        ["Following the completion of the merger, the new entity will be headquartered in Singapore."]
    ],
    theme=gr.themes.Soft()
)

# 4. Launch
if __name__ == "__main__":
    # share=True provides a temporary public URL for interviews
    demo.launch(share=True)
