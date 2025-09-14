import torch
from diffusers import StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline

# --- Configuration ---
PROMPT = "a photograph of an astronaut riding a horse"
OUTPUT_FILENAME = "generated_image_upscaled.png"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
UPSCALER_MODEL_ID = "stabilityai/sd-x2-latent-upscaler"

def generate_upscaled_image():
    """
    Generates a high-quality base image and then uses an AI upscaler
    to increase its resolution.
    """
    # --- Device Setup ---
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
        return
    device = "cuda"
    torch_dtype = torch.float16

    # --- Load Base Pipeline ---
    print(f"Loading base model: {BASE_MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch_dtype
    ).to(device)

    # --- Load Upscaler Pipeline ---
    print(f"Loading upscaler model: {UPSCALER_MODEL_ID}")
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        UPSCALER_MODEL_ID, torch_dtype=torch_dtype
    ).to(device)

    # --- Stage 1: Generate Base Image ---
    print("\nStage 1: Generating 768x768 base image (40 steps)...")
    # We generate in latent space to pass directly to the upscaler
    base_latents = pipe(
        prompt=PROMPT,
        height=768,
        width=768,
        num_inference_steps=40,
        output_type="latent",
    ).images[0]

    # --- Stage 2: Upscale Image ---
    print("Stage 2: Upscaling image to 1536x1536 (20 steps)...")
    upscaled_image = upscaler(
        prompt=PROMPT,
        image=base_latents,
        num_inference_steps=20,
        guidance_scale=0, # This should be 0 for the latent upscaler
    ).images[0]

    # --- Save Final Image ---
    print(f"\nSaving upscaled image as '{OUTPUT_FILENAME}'")
    upscaled_image.save(OUTPUT_FILENAME)
    print("Done!")

if __name__ == "__main__":
    generate_upscaled_image()