import gradio as gr

def greet(name):
    return "Hello, " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# This is the variable Azure looks for by default
app = demo.app

# For local development
if __name__ == "__main__":
    demo.launch()
