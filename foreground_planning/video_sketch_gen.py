from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time 
import copy 
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import requests
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import re 
import json
from tqdm import tqdm 
import os 
import argparse



def parse_frames(input_text):
    frames = []
    
    input_text = input_text.replace("“", '"').replace("”", '"')

    # Modify the pattern to handle frames without newlines
    pattern = re.compile(
                r'Frame_(\d+):\s*(\[\[.*?\]\])\s*,\s*caption:\s*(.*?)(?=\n?Frame_\d+:|\Z)',

        re.DOTALL
    )
    matches = pattern.finditer(input_text)
    for match in matches:
        frame_number = int(match.group(1))
        objects_list_str = match.group(2)
        caption = match.group(3).strip()
        if "Reasoning:" in caption:
            caption = caption.split("Reasoning:")[0].strip("\n")

        # Parse the objects list string into a Python list
        try:
            objects_list = json.loads(objects_list_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in frame {frame_number}: {e}")
            continue

        frames.append({
            'frame_number': frame_number,
            'objects': objects_list,
            'caption': caption
        })
    return frames


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
        



def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]


    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f"{label}: {score:.2f}"
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x", yref="y",
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
                xref="x", yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="up",
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show plot
    fig.show()


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks
     
     
def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections
     

def load_pipeline(generator_type: str, cache_dir: str):
    if generator_type == "flux":
        return DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            cache_dir=cache_dir
        ).to("cuda")
    elif generator_type == "sdxl":
        return AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")
    else:
        raise ValueError("Unsupported generator type. Choose 'flux' or 'sdxl'.")

def main(args):
    with open(args.plan_path, "r") as f:
        plan1 = json.load(f)
    f.close()

    img_pipeline = load_pipeline(args.generator, args.cache_dir)
    output_path = args.output_path

    threshold = 0.3

    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam-vit-base"

    input_path = args.input_background_video_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for k, v in plan1.items():
        skill_name = k.split(".")[0]
        if not os.path.exists(f"{input_path}/{skill_name}/{index:04d}/{i}.jpg"):
            continue 
        if not os.path.exists(f"{output_path}/{skill_name}"):
            os.mkdir(f"{output_path}/{skill_name}")

        for i, plan in enumerate(tqdm(v)):
            index = i + 1
            
            if "[]" in plan or "[ ]" in plan:
                print("Error processing frames for plan", i)
                continue 
            parsed_frames = parse_frames(plan)
            if parsed_frames == []:
                print("Error processing frames for plan", i)
                continue 
            all_objs = set()
            # Define threshold for movement detection
            MOVEMENT_THRESHOLD = 0.05

            # Collect moving objects
            for frame in parsed_frames:
                for obj in frame['objects']:
                    obj_name = obj[0]
                    while isinstance(obj_name, list):
                        obj_name = obj_name[0]
                    all_objs.add(obj_name)
        
            final_images = []
            for i in range(49):
                final_images.append(Image.open(f"{input_path}/{skill_name}/{index:04d}/{i}.jpg").resize((1024, 1024)))

            for obj in all_objs:
                generator = torch.Generator("cuda").manual_seed(13)
                obj_image = img_pipeline(f"A photo of a {obj}", generator=generator).images[0]
                obj_image = obj_image.resize((1024, 1024))
                try:
                    # SAM to extract the obj in the transparent background
                    image_array, detections = grounded_segmentation(
                            image=obj_image,
                            labels=[obj],
                            threshold=threshold,
                            polygon_refinement=True,
                            detector_id=detector_id,
                            segmenter_id=segmenter_id
                        )

                    # # Load the image and assume mask is already a uint8 array
                    img_np = np.array(obj_image)

                    # Convert RGB to BGR for OpenCV compatibility
                    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    mask = detections[0].mask

                    # Ensure the image has 4 channels (RGBA)
                    if img.shape[2] != 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                    # Apply the mask to the alpha channel
                    img[:, :, 3] = mask

                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
                    x1, y1, x2, y2 = detections[0].box.xmin, detections[0].box.ymin, detections[0].box.xmax, detections[0].box.ymax
                    resized_img = np.array(pil_img)[y1:y2, x1:x2]
                    
                    # Define the boxes
                    box1 = np.array([detections[0].box.xmin, detections[0].box.ymin, detections[0].box.xmax, detections[0].box.ymax], dtype=float)
                    # Normalize box1 by 1024
                    box1 /= 1024

                    # For each frame, paste the image onto the background
            
                    for j, frame in enumerate(parsed_frames):
                        for obj_ in frame["objects"]:
                            if obj_[0] == obj:
                                box2 = np.array(obj_[1])
                                break 
                        
                        # Calculate width and height of both boxes
                        box1_width = box1[2] - box1[0]
                        box1_height = box1[3] - box1[1]
                        box2_width = box2[2] - box2[0]
                        box2_height = box2[3] - box2[1]

                        # Identify the longer side
                        longer_side = max(box2_width, box2_height)

                        # Scale factor to resize the longer side to 0.15
                        if longer_side < 0.15:
                            scale_factor = 0.15 / longer_side
                        else:
                            scale_factor = 1 

                        # Compute new width and height
                        box2_width = box2_width * scale_factor
                        box2_height = box2_height * scale_factor
                        
                        # Determine scaling factor to fit longest edge
                        scale_factor = min(box2_width / box1_width, box2_height / box1_height)

                        # Calculate new size using scale_factor
                        new_width = int(resized_img.shape[1] * scale_factor)
                        new_height = int(resized_img.shape[0] * scale_factor)
                        
                        # Resize the image
                        scaled_img = Image.fromarray(resized_img).resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Load the background image
                        for m in range(2):
                            background = copy.deepcopy(final_images[j*2+m])
                            
                            # Calculate the center of the box2
                            center_x = (box2[0] + box2[2]) / 2
                            center_y = (box2[1] + box2[3]) / 2

                            # Calculate the target width and height in the background
                            bg_width = int(box2_width * 1024)
                            bg_height = int(box2_height * 1024)

                            # Adjust x_start and y_start to ensure the center of box2 remains fixed
                            x_start = int(center_x * 1024 - bg_width / 2)
                            y_start = int(center_y * 1024 - bg_height / 2)
                            
                            # Resize the scaled ball if needed to fit the target box
                            scaled_img = scaled_img.resize((bg_width, bg_height), Image.Resampling.LANCZOS)
                            
                            background.paste(scaled_img, (x_start, y_start), scaled_img)
                            final_images[j*2+m] = background
                except:
                    continue

            if not os.path.exists(f"{output_path}/{skill_name}/{index:04d}"):
                os.mkdir(f"{output_path}/{skill_name}/{index:04d}")
            for k, img in enumerate(final_images):
                img.save(f"{output_path}/{skill_name}/{index:04d}/{k}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=str, required=True, help="Path to the object planning JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output background images directory")
    parser.add_argument("--input_background_video_path", type=str, required=True, help="Path to background videos")
    parser.add_argument("--generator", type=str, choices=["flux", "sdxl"], default="flux", help="Image generation pipeline to use")
    parser.add_argument("--seed", type=int, default=31, help="Seed for image generation")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache pretrained models")
    args = parser.parse_args()

    main(args)