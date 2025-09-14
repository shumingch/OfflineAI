import torch
from diffusers import DiffusionPipeline

# --- Configuration ---
# Model ID from Hugging Face. "runwayml/stable-diffusion-v1-5" is a popular choice.
# You can find other models at https://huggingface.co/models?pipeline_tag=text-to-image
MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPT = "a photograph of an astronaut riding a horse, with only one arm visible"
OUTPUT_FILENAME = "generated_image.png"

def generate_image(pipe, prompt, output_filename, num_inference_steps=20):
    """
    Uses a pre-loaded Stable Diffusion pipeline to generate an image from a text prompt.
    """
    print(f"\nGenerating image for prompt: '{prompt}'")
    print("This can take a moment...")
    try:
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        image.save(output_filename)
        print(f"\nSuccessfully saved image as '{output_filename}'")
    except Exception as e:
        print(f"\nAn error occurred during image generation: {e}")
        print("This could be due to memory constraints. If you are running out of GPU memory,")
        print("consider using a smaller model or reducing the image resolution if possible.")

if __name__ == "__main__":
    # --- Device Setup ---
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. This will be very slow on the CPU.")
        device = "cpu"
        torch_dtype = torch.float32
    else:
        device = "cuda"
        torch_dtype = torch.float16

    # --- Model Loading ---
    print(f"Loading model: {MODEL_ID}")
    print("This may take a few minutes and require a good internet connection on the first run.")

    try:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            local_files_only=True
        )
        print("Model loaded from local cache.")
    except EnvironmentError:
        print("Could not find model in local cache. Downloading from Hugging Face Hub...")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            local_files_only=False
        )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # Enable attention slicing for speed/memory

    # --- Image Generation ---
    try:
        generate_image(pipe, PROMPT, OUTPUT_FILENAME, num_inference_steps=20)
        generate_image(pipe, PROMPT, "generated_image_2.png", num_inference_steps=20)
    except Exception as e:
        print(f"\nAn error occurred during image generation: {e}")
        print("This could be due to memory constraints. If you are running out of GPU memory,")
        print("consider using a smaller model or reducing the image resolution if possible.")
