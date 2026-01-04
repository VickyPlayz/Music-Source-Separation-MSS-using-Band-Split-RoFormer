import gradio as gr
import torch
import os
import subprocess
import soundfile as sf

# Wrapper to call inference.py
def separate_audio(audio_path):
    # Fixed checkpoint path for demo (User should update)
    checkpoint = "models/checkpoints/checkpoint_ep1.pt"
    if not os.path.exists(checkpoint):
        return [None]*4
        
    output_dir = "outputs/demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # We call python script or import directly
    # Call script to isolate env/args
    cmd = [
        "venv/Scripts/python", "src/inference.py", 
        audio_path, 
        "--checkpoint", checkpoint,
        "--output_dir", output_dir,
        "--dim", "64", # Matches our dry run
        "--depth", "1",
        "--heads", "4"
    ]
    
    subprocess.run(cmd, check=True)
    
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    stems = ["vocals", "drums", "bass", "other"]
    results = []
    for s in stems:
        path = os.path.join(output_dir, f"{basename}_{s}.wav")
        results.append(path)
        
    return results

if __name__ == "__main__":
    iface = gr.Interface(
        fn=separate_audio,
        inputs=gr.Audio(type="filepath", label="Input Mix"),
        outputs=[
            gr.Audio(label="Vocals"),
            gr.Audio(label="Drums"),
            gr.Audio(label="Bass"),
            gr.Audio(label="Other")
        ],
        title="Band-Split RoFormer MSS Demo",
        description="Upload a song to separate into 4 stems."
    )
    iface.launch()
