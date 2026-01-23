import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "siddharth-magesh/Tiny_Lllama-sqlQuries-fine_tuned"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")  # Ensure model is on CPU
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

def chat_template(question, context):
    template = f"""\
    <|im_start|>user
    Given the context, generate an SQL query for the following question
    context:{context}
    question:{question}
    <|im_end|>
    <|im_start|>assistant 
    """
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template

def generate_response(context, user_input):
    prompt = chat_template(user_input,context) 
    inputs = tokenizer(prompt, return_tensors="pt").to('cpu') 
    output = model.generate(**inputs, max_new_tokens=512)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# Streamlit app
st.title("Streamlit App with Context and Input")

# Input fields
context = st.text_area("Context", "Enter the context here...")
user_input = st.text_area("Input", "Enter the input here...")

# Button to generate response
if st.button("Generate Response"):
    response = generate_response(context, user_input)
    st.write("Response:")
    st.write(response)
