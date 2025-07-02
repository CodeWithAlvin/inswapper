"""
Video Face Swapper based on inswapper_128
This project extends the single frame face swap to video processing.
Built on top of insightface, sd-webui-roop and CodeFormer.
"""

import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
import tempfile
from tqdm import tqdm
from restoration import *


def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser, frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

    
def get_many_faces(face_analyser, frame: np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    """
    paste source_face on target image
    """
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(face_swapper, face_analyser, source_faces, frame, source_indexes, target_indexes):
    """
    Process a single frame for face swapping
    """
    # Detect faces in the current frame
    target_faces = get_many_faces(face_analyser, frame)
    
    if target_faces is None:
        return frame
    
    num_target_faces = len(target_faces)
    num_source_faces = len(source_faces)
    temp_frame = copy.deepcopy(frame)
    
    if target_indexes == "-1":
        if num_source_faces == 1:
            num_iterations = num_target_faces
        elif num_source_faces < num_target_faces:
            num_iterations = num_source_faces
        elif num_target_faces < num_source_faces:
            num_iterations = num_target_faces
        else:
            num_iterations = num_target_faces

        for i in range(num_iterations):
            source_index = 0 if num_source_faces == 1 else i
            target_index = i
            temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame)
    else:
        if source_indexes == "-1":
            source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))
        
        source_indexes = source_indexes.split(',')
        target_indexes = target_indexes.split(',')
        num_source_faces_to_swap = len(source_indexes)
        num_target_faces_to_swap = len(target_indexes)
        
        if num_source_faces_to_swap > num_source_faces:
            return frame
        
        if num_target_faces_to_swap > num_target_faces:
            return frame
        
        num_iterations = min(num_source_faces_to_swap, num_target_faces_to_swap)
        
        for index in range(num_iterations):
            source_index = int(source_indexes[index])
            target_index = int(target_indexes[index])
            
            if source_index > num_source_faces - 1 or target_index > num_target_faces - 1:
                continue
                
            temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame)
    
    return temp_frame


def process_video(source_img: Union[Image.Image, List],
                  target_video_path: str,
                  output_video_path: str,
                  source_indexes: str,
                  target_indexes: str,
                  model: str,
                  face_restore: bool = False,
                  background_enhance: bool = False,
                  face_upsample: bool = False,
                  upscale: int = 1,
                  codeformer_fidelity: float = 0.5):
    
    # Load machine default available providers
    providers = onnxruntime.get_available_providers()
    
    # Load face_analyser
    face_analyser = getFaceAnalyser(model, providers)
    
    # Load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)
    
    # Process source images to get faces
    if isinstance(source_img, list):
        source_faces_list = []
        for img in source_img:
            source_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            faces = get_many_faces(face_analyser, source_frame)
            if faces:
                source_faces_list.extend(faces)
        source_faces = source_faces_list
    else:
        source_frame = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        source_faces = get_many_faces(face_analyser, source_frame)
    
    if not source_faces:
        raise Exception("No source faces found!")
    
    # Open video
    cap = cv2.VideoCapture(target_video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"VIDEO DETAILS - {fps} FPS {width}X{height} {total_frames} FRAMES")
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize face restoration if needed
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
        except ImportError:
            print("Face restoration dependencies not found. Skipping face restoration.")
            face_restore = False
    
    # Process video frame by frame
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = process_frame(face_swapper, face_analyser, source_faces, frame, source_indexes, target_indexes)
        
        # Apply face restoration if enabled
        if face_restore:
            try:
                processed_frame = face_restoration(
                    processed_frame,
                    background_enhance,
                    face_upsample,
                    upscale,
                    codeformer_fidelity,
                    upsampler,
                    codeformer_net,
                    device
                )
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                processed_frame = cv2.resize(processed_frame, (width, height))
            except:
                pass  # Continue without restoration if it fails
        
        # Write frame
        out.write(processed_frame)
        frame_count += 1
        pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    pbar.close()
    
    print(f'Video processing completed: {output_video_path}')
    return output_video_path


def parse_args():
    parser = argparse.ArgumentParser(description="Video face swap.")
    parser.add_argument("--source_img", type=str, required=True, 
                       help="The path of source image(s), can be multiple images separated by semicolon: dir1;dir2;dir3")
    parser.add_argument("--target_video", type=str, required=True, 
                       help="The path of target video.")
    parser.add_argument("--output_video", type=str, required=False, default="result.mp4", 
                       help="The path and filename of output video.")
    parser.add_argument("--source_indexes", type=str, required=False, default="-1", 
                       help="Comma separated list of face indexes to use in source image(s), starting at 0 (-1 uses all faces)")
    parser.add_argument("--target_indexes", type=str, required=False, default="-1", 
                       help="Comma separated list of face indexes to swap in target video, starting at 0 (-1 swaps all faces)")
    parser.add_argument("--face_restore", action="store_true", 
                       help="Enable face restoration.")
    parser.add_argument("--background_enhance", action="store_true", 
                       help="Enable background enhancement.")
    parser.add_argument("--face_upsample", action="store_true", 
                       help="Enable face upsampling.")
    parser.add_argument("--upscale", type=int, default=1, 
                       help="Upscale factor, up to 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, 
                       help="CodeFormer fidelity value.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Parse source images
    source_img_paths = args.source_img.split(';')
    print("Source image paths:", source_img_paths)
    
    # Load source images
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    
    # Model path - download from https://huggingface.co/deepinsight/inswapper/tree/main
    model = "./checkpoints/inswapper_128.onnx"
    
    # Process video
    result_video = process_video(
        source_img=source_img,
        target_video_path=args.target_video,
        output_video_path=args.output_video,
        source_indexes=args.source_indexes,
        target_indexes=args.target_indexes,
        model=model,
        face_restore=args.face_restore,
        background_enhance=args.background_enhance,
        face_upsample=args.face_upsample,
        upscale=args.upscale,
        codeformer_fidelity=args.codeformer_fidelity
    )
    
    print(f'Video face swap completed: {result_video}')