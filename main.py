import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("2003achu/ai_poet")
model = AutoModelForSeq2SeqLM.from_pretrained("2003achu/ai_poet")

# Function to generate AI poem
def generate(prompt):
    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_length=150, num_return_sequences=3, temperature=0.9)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]

# About Section
st.sidebar.title('About')
st.sidebar.info(
    "This AI Poem Generator takes a word or phrase as input and generates AI-generated poems based on it. "
    "Select a prompt example below to see how it works."
    "\n\nPrompt Examples:"
    "\n- Tree"
    "\n- Nature"
    "\n- Winter"
)

# Main content
st.title('AI Poem Generator')
input_text = st.text_input("Enter your prompt here")

# Generate poems upon button click
if st.button("Generate Poem"):
    if input_text:
        # Display spinner while generating poems
        with st.spinner("Generating poems..."):
            output = generate(input_text)
            st.text_area(f"Poem :", value=output, height=5)

# Footer
st.markdown("---")
st.markdown("This app takes an input word or phrase and converts it into AI-generated poems.")