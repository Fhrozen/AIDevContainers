# HTML content for the header section

header_title = """
Generate about an hour of audio per minute on the Kokoro-82M TTS model, with unexpected quality
""".strip()

time_button = """
⏱️ Small requests/Initial chunks can be slower due to warm-up
"""

header_html = f"""
<div>
    <!-- Top badges bar -->
    <div style="display: flex; justify-content: flex-end; padding: 4px; gap: 8px; height: 32px; align-items: center;">
        <div style="height: 28px; display: flex; align-items: center; margin-top: 3px;">
            <a class="github-button" href="https://github.com/Fhrozen/AIDevContainers" data-color-scheme="no-preference: dark; light: dark; dark: dark;" data-size="large" data-show-count="true">AIDevContainers</a>
        </div>
        <a href="https://huggingface.co/hexgrad/Kokoro-82M" target="_blank" style="height: 28px; display: flex; align-items: center;">
            <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Model on HF" style="height: 100%;">
        </a>
    </div>

    <div style="text-align: center; margin-bottom: 1rem;">
        <h1 style="font-size: 1.75rem; font-weight: bold; color: #ffffff; margin-bottom: 0.5rem;">Kokoro TTS Demo</h1>
        <p style="color: #d1d5db;">{header_title}</p>
    </div>
        
    <div style="display: flex; gap: 1rem;">
        <div style="flex: 1; background: rgba(30, 58, 138, 0.3); border: 1px solid rgba(59, 130, 246, 0.3); padding: 0.5rem 1rem; border-radius: 6px; display: flex; align-items: center; justify-content: center;">
            <span style="font-weight: 500; color: #60a5fa; text-align: center;">{time_button}</span>
        </div>
    </div>
</div>
<script async defer src="https://buttons.github.io/buttons.js"></script>
"""

# Markdown content for demo text info
demo_text_info = """
All input text was sourced as public domain.
"""

styling = """
    .equal-height {
        min-height: 400px;
        display: flex;
        flex-direction: column;
    }
    .token-label {
        font-size: 1rem;
        margin-bottom: 0.3rem;
        text-align: center;
        padding: 0.2rem 0;
    }
    .token-count {
        color: #4169e1;
    }
    #gradio-accordion > .label-wrap {
        background: radial-gradient(circle, rgba(30,58,138,0.8) 0%, rgba(167,71,254,0.6) 100%);
        padding: 0.8rem 1rem;
        font-size: 1rem;
        color: #000000;
        border-radius: 4px;
    }
"""
