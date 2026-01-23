import streamlit as st
from transformers import AutoTokenizer , AutoModelForCausalLM
import torch

model_id = "siddharth-magesh/Tiny-Llama-alpaca-fine-tuning"
model = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Define your model function that takes a text input and returns a text response
def model_function(input_text):
    # Example: Tokenize the input text
    input_text = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:" + input_text + " ### Response : "

    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**input_ids,max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Streamlit app setup
st.title("Chatbot Interface")

# Create an input text box
user_input = st.text_input("Enter your question here:")

# Create a button that, when clicked, will call the model function
if st.button("Get Response"):
    if user_input:
        # Tokenize the input and get the model's response
        response = model_function(user_input)

        # Display the response
        st.write("Response from the model:")
        st.write(response)
    else:
        st.warning("Please enter a question before clicking the button.")
