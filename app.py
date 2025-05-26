from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def medical_chatbot(question):
    prompt = (
        "You are a knowledgeable and helpful medical assistant. "
        "Answer the user's medical question clearly and accurately based on reliable medical knowledge.\n"
        f"Question: {question}\nAnswer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()

interface = gr.Interface(
    fn=medical_chatbot,
    inputs="text",
    outputs="text",
    title="Medical Chatbot (GPT-2)",
    description="Ask a medical question and receive a helpful answer based on general medical knowledge."
)

interface.launch()