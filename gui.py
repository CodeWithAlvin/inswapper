"""
Gradio Interface for Face Swapping (Images and Videos)
Supports both single image and video face swapping with restoration options.
"""

import gradio as gr
import os
import tempfile
import shutil
from PIL import Image
import cv2
import numpy as np
from restoration import *

# Import the face swap functions
from swapper import process as process_image
from video import process_video


def swap_faces_image(source_images, target_image, source_indexes, target_indexes, 
                    face_restore, background_enhance, face_upsample, upscale, codeformer_fidelity):
    """
    Process image face swapping through Gradio interface
    """
    try:
        if not source_images or not target_image:
            return None, "Please provide both source and target images."
        
        # Convert source_images to list if it's a single image
        if not isinstance(source_images, list):
            source_images = [source_images]
        
        # Convert PIL images if needed
        source_pil = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in source_images]
        target_pil = Image.fromarray(target_image) if isinstance(target_image, np.ndarray) else target_image
        
        # Set default model path
        model = "./checkpoints/inswapper_128.onnx"
        
        # Check if model exists
        if not os.path.exists(model):
            return None, f"Model not found at {model}. Please download inswapper_128.onnx from HuggingFace."
        
        # Process the image
        result_image = process_image(
            source_img=source_pil,
            target_img=target_pil,
            source_indexes=source_indexes,
            target_indexes=target_indexes,
            model=model
        )
        
        # Apply face restoration if requested
        if face_restore:
            try:
                import torch
                
                check_ckpts()
                upsampler = set_realesrgan()
                device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                
                codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(device)
                
                ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
                checkpoint = torch.load(ckpt_path)["params_ema"]
                codeformer_net.load_state_dict(checkpoint)
                codeformer_net.eval()
                
                result_array = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                result_array = face_restoration(
                    result_array,
                    background_enhance,
                    face_upsample,
                    upscale,
                    codeformer_fidelity,
                    upsampler,
                    codeformer_net,
                    device
                )
                result_image = Image.fromarray(result_array)
            except Exception as e:
                print(f"Face restoration failed: {e}")
        
        return result_image, "Face swap completed successfully!"
        
    except Exception as e:
        return None, f"Error during face swap: {str(e)}"


def swap_faces_video(source_images, target_video, source_indexes, target_indexes,
                    face_restore, background_enhance, face_upsample, upscale, codeformer_fidelity):
    """
    Process video face swapping through Gradio interface
    """
    try:
        if not source_images or not target_video:
            return None, "Please provide both source images and target video."
        
        # Convert source_images to list if it's a single image
        if not isinstance(source_images, list):
            source_images = [source_images]
        
        # Convert to PIL images
        source_pil = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in source_images]
        
        # Set default model path
        model = "./checkpoints/inswapper_128.onnx"
        
        # Check if model exists
        if not os.path.exists(model):
            return None, f"Model not found at {model}. Please download inswapper_128.onnx from HuggingFace."
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Process the video
        result_video = process_video(
            source_img=source_pil,
            target_video_path=target_video,
            output_video_path=output_path,
            source_indexes=source_indexes,
            target_indexes=target_indexes,
            model=model,
            face_restore=face_restore,
            background_enhance=background_enhance,
            face_upsample=face_upsample,
            upscale=upscale,
            codeformer_fidelity=codeformer_fidelity
        )
        
        return result_video, "Video face swap completed successfully!"
        
    except Exception as e:
        return None, f"Error during video face swap: {str(e)}"


# Create Gradio interface
def create_interface():
    """
    Create and configure the Gradio interface
    """
    
    with gr.Blocks(title="Face Swap Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎭 Face Swap Tool")
        gr.Markdown("Swap faces in images and videos using InsightFace and InSwapper")
        
        with gr.Tabs():
            # Image Face Swap Tab
            with gr.TabItem("📸 Image Face Swap"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Source Images")
                        source_images_img = gr.File(
                            label="Upload Source Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        gr.Markdown("### Target Image")
                        target_image_img = gr.Image(
                            label="Upload Target Image",
                            type="pil"
                        )
                        
                        with gr.Row():
                            source_indexes_img = gr.Textbox(
                                label="Source Face Indexes",
                                value="-1",
                                placeholder="e.g., 0,1,2 or -1 for all faces"
                            )
                            target_indexes_img = gr.Textbox(
                                label="Target Face Indexes", 
                                value="-1",
                                placeholder="e.g., 0,1,2 or -1 for all faces"
                            )
                    
                    with gr.Column():
                        gr.Markdown("### Enhancement Options")
                        face_restore_img = gr.Checkbox(label="Face Restoration", value=False)
                        background_enhance_img = gr.Checkbox(label="Background Enhancement", value=False)
                        face_upsample_img = gr.Checkbox(label="Face Upsample", value=False)
                        
                        with gr.Row():
                            upscale_img = gr.Slider(
                                label="Upscale Factor",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1
                            )
                            codeformer_fidelity_img = gr.Slider(
                                label="CodeFormer Fidelity",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1
                            )
                        
                        swap_button_img = gr.Button("🔄 Swap Faces", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        result_image = gr.Image(label="Result", type="pil")
                    with gr.Column():
                        status_img = gr.Textbox(label="Status", lines=3)
            
            # Video Face Swap Tab
            with gr.TabItem("🎥 Video Face Swap"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Source Images")
                        source_images_vid = gr.File(
                            label="Upload Source Images",
                            file_count="multiple", 
                            file_types=["image"]
                        )
                        
                        gr.Markdown("### Target Video")
                        target_video_vid = gr.Video(
                            label="Upload Target Video"
                        )
                        
                        with gr.Row():
                            source_indexes_vid = gr.Textbox(
                                label="Source Face Indexes",
                                value="-1",
                                placeholder="e.g., 0,1,2 or -1 for all faces"
                            )
                            target_indexes_vid = gr.Textbox(
                                label="Target Face Indexes",
                                value="-1", 
                                placeholder="e.g., 0,1,2 or -1 for all faces"
                            )
                    
                    with gr.Column():
                        gr.Markdown("### Enhancement Options")
                        face_restore_vid = gr.Checkbox(label="Face Restoration", value=False)
                        background_enhance_vid = gr.Checkbox(label="Background Enhancement", value=False)
                        face_upsample_vid = gr.Checkbox(label="Face Upsample", value=False)
                        
                        with gr.Row():
                            upscale_vid = gr.Slider(
                                label="Upscale Factor",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1
                            )
                            codeformer_fidelity_vid = gr.Slider(
                                label="CodeFormer Fidelity",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1
                            )
                        
                        swap_button_vid = gr.Button("🔄 Swap Faces", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        result_video = gr.Video(label="Result Video")
                    with gr.Column():
                        status_vid = gr.Textbox(label="Status", lines=3)
        
        # Setup event handlers
        def process_source_images(files):
            if not files:
                return []
            images = []
            for file in files:
                try:
                    img = Image.open(file.name)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {file.name}: {e}")
            return images
        
        # Image processing
        swap_button_img.click(
            fn=lambda source_files, target, s_idx, t_idx, restore, bg_enh, up_face, upsc, fid: 
                swap_faces_image(
                    process_source_images(source_files) if source_files else None,
                    target, s_idx, t_idx, restore, bg_enh, up_face, upsc, fid
                ),
            inputs=[
                source_images_img, target_image_img, source_indexes_img, target_indexes_img,
                face_restore_img, background_enhance_img, face_upsample_img, upscale_img, codeformer_fidelity_img
            ],
            outputs=[result_image, status_img]
        )
        
        # Video processing  
        swap_button_vid.click(
            fn=lambda source_files, target, s_idx, t_idx, restore, bg_enh, up_face, upsc, fid:
                swap_faces_video(
                    process_source_images(source_files) if source_files else None,
                    target, s_idx, t_idx, restore, bg_enh, up_face, upsc, fid
                ),
            inputs=[
                source_images_vid, target_video_vid, source_indexes_vid, target_indexes_vid,
                face_restore_vid, background_enhance_vid, face_upsample_vid, upscale_vid, codeformer_fidelity_vid
            ],
            outputs=[result_video, status_vid]
        )
        
        # Add information section
        with gr.Accordion("ℹ️ Usage Instructions", open=False):
            gr.Markdown("""
            ### How to use:
            
            **For Images:**
            1. Upload one or more source images containing the faces you want to use
            2. Upload a target image where you want to replace faces
            3. Specify face indexes (optional): 
               - Use `-1` to swap all faces
               - Use specific indexes like `0,1,2` to swap particular faces (left to right order)
            4. Enable enhancement options if desired
            5. Click "Swap Faces"
            
            **For Videos:**
            1. Upload source images containing the faces you want to use
            2. Upload a target video where you want to replace faces
            3. Configure indexes and enhancement options
            4. Click "Swap Faces" (this may take a while depending on video length)
            
            ### Requirements:
            - Download `inswapper_128.onnx` from [HuggingFace](https://huggingface.co/deepinsight/inswapper/tree/main)
            - Place it in `./checkpoints/inswapper_128.onnx`
            - For face restoration: Download CodeFormer weights
            
            ### Face Indexes:
            - Faces are detected and numbered from left to right (0, 1, 2, ...)
            - Source indexes: which faces from source images to use
            - Target indexes: which faces in target to replace
            - Use `-1` to process all detected faces
            """)
    
    return app


if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True
    )