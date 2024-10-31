import torch
from unixcoder import UniXcoder
import gradio as gr

# Set up the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the UniXcoder model
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

# Define the function for code autocompletion
def code_autocomplete(context):
    # Tokenize the input context
    tokens_ids = model.tokenize([context], max_length=512, mode="<decoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    
    # Generate predictions
    prediction_ids = model.generate(
        source_ids,
        decoder_only=True,
        beam_size=3,
        max_length=128
    )
    
    # Decode the predictions
    predictions = model.decode(prediction_ids)
    
    # Return the original context concatenated with the generated code
    return context + predictions[0][0]

# Create the Gradio interface
iface = gr.Interface(
    fn=code_autocomplete,
    inputs=gr.Textbox(lines=15, placeholder="Enter your code here..."),
    outputs=gr.Textbox(label="Autocompleted Code"),
    title="Code Autocompletion with UniXcoder",
    description="Enter a code snippet, and the model will generate autocompleted code based on your input.",
    examples=[
        ["def add_numbers(a, b):\n    # Add two numbers and return the result"],
        ["public class HelloWorld {\n    public static void main(String[] args) {"],
        ["def fibonacci(n):\n    # Calculate the nth Fibonacci number"],
    ],
)

# Launch the interface
iface.launch()
