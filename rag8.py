import subprocess
import streamlit as st
from llama_index.llms.ollama import Ollama

# Function to get installed Ollama models using the 'ollama list' command
def get_ollama_models():
    try:
        # Execute the 'ollama list' command and capture the output
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        
        # Parse the output (assuming the models are listed line by line)
        models = result.stdout.splitlines()
        
        # Filter out empty lines or unnecessary information (if needed)
        model_names = []
        for model in models:
            model = model.strip()
            if model:  # Skip empty lines
                # Split by the first space and take the part before the first space
                model_name = model.split()[0]
                model_names.append(model_name)

        return model_names
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while trying to fetch models: {e}")
        return []

# Get the available models using 'ollama list'
available_models = get_ollama_models()

# Check if models are available
if available_models:
    # Dropdown to select the model
    selected_model = st.selectbox("Select an Ollama Model", available_models)
    st.write(f"You selected: {selected_model}")

    # Set up the LLM model based on the selected model
    llm = Ollama(model=selected_model, request_timeout=300.0)

else:
    st.write("No models found. Please ensure you have models installed using 'ollama list'.")

# Example usage of selected model for a query (if needed):
if st.button("Ask"):
    # Example query
    query = "What is the capital of France?"
    response = llm.query(query)  # Make sure `llm.query()` is the right method
    st.write(f"Response from model: {response}")
