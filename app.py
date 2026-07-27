"""Hugging Face Spaces entrypoint. Spaces looks for `app.py` at the repo
root by default; the real app lives in `app/gradio_app.py` alongside the
rest of the pipeline code, so this just re-exports its `demo` object.
"""
from app.gradio_app import demo

if __name__ == "__main__":
    demo.launch()
