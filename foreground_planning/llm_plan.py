import base64
import requests
import json 
import argparse
from tqdm import tqdm 
import os 
import re 



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def normalize_number(number: str) -> str:
    return re.sub(r'\d+', lambda x: x.group().zfill(4), number)

def main(args):
    # OpenAI API Key
    api_key = args.api_key 
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    output_file = args.output_file
    result = dict()
    
    system_prompt = '''Prompt: Assuming the frame size is normalized to the range 0-1, you need to give a possible 25-frame layout with bounding boxes of the entities of a given event. 

        The background image is provided. Besides, the objects in the background image are also annotated with bounding box, and these objects are static objects across the 25 frames. You need to generate the moving object at the correct location based on the input background image and bounding box annotation. You don't need to include the box plan of the objects that already in the background. 

        You need to generate plan for all the key objects in the prompt (e.g., Generate both objA and objB for prompt objA is left to objB). There should be enough distance between the two objects so that each object can be generated clearly. 

        The object name should contain its attributes (e.g., color, shape). 

        Number the objects if there the multiple same objects in the prompt (e.g., 2 birds). 

        Each object in the image is one rectangle or square box in the layout and size of boxes should be as large as possible. You need to generate layouts from the close up camera view of the event. 

        The layout difference between two adjacent frames must be small, considering the small interval. 

        You need to generate a caption that best describes the image for each frame. After generating all frames, add reasoning to your design. 

        Please strictly follow the user format here to generate the plan. Do not include any code or calculation before generating the plan. Add all the reasoning process after generating the plan. 

        Use format: 
        Frame_1: [[object1, [left, top, right, bottom]], [object2, [left, top, right, bottom]], ..., [object_n, [left, top, right, bottom]]], caption:...
        Frame_2: [[object1, [left, top, right, bottom]], [object2, [left, top, right, bottom]], ..., [object_n, [left, top, right, bottom]]], caption:...
        ...
        Frame_25: [[object1, [left, top, right, bottom]], [object2, [left, top, right, bottom]], ..., [object_n, [left, top, right, bottom]]], caption:...

        Reasoning:...' 

        Here's one example: 
        User: Provide bounding box coordinates for the prompt: A cat walks from left to right on a table. 

        Background box annotation: [ {"label": "window", "box": [0.6983, 0.0095, 0.9983, 0.4673]}, {"label": "floor", "box": [0.0013, 0.6519, 0.9981, 0.9975]}, {"label": "lamp", "box": [0.4194, 0.0015, 0.5276, 0.3123]}, {"label": "plant", "box": [0.0011, 0.1719, 0.2060, 0.6134]}, {"label": "table", "box": [0.2776, 0.4650, 0.9937, 0.8847]}, {"label": "lamp", "box": [0.5457, 0.0013, 0.6814, 0.2483]}, {"label": "radiator", "box": [0.7764, 0.5358, 0.9980, 0.7557]}, {"label": "room room", "box": [0.0014, 0.0030, 0.9983, 0.9960]}, {"label": "lamp", "box": [0.4183, 0.0012, 0.6815, 0.3127]}, {"label": "plant", "box": [0.6179, 0.3298, 0.6996, 0.4584]}, {"label": "stool", "box": [0.1339, 0.5109, 0.3203, 0.7738]}, {"label": "stool", "box": [0.0682, 0.4962, 0.2351, 0.7270]}, {"label": "chair", "box": [0.2132, 0.4343, 0.4240, 0.7143]} ] 

        Assistant: 
        Frame_1: [["cat", [0.278, 0.365, 0.378, 0.465]]], caption: The cat steps onto the table from the left, beginning its walk.
        Frame_2: [["cat", [0.303, 0.365, 0.403, 0.465]]], caption: The cat advances a little further on the table, heading right.
        Frame_3: [["cat", [0.329, 0.365, 0.429, 0.465]]], caption: The cat continues its steady walk to the right.
        Frame_4: [["cat", [0.355, 0.365, 0.455, 0.465]]], caption: The cat moves further along the table, walking confidently.
        Frame_5: [["cat", [0.38, 0.365, 0.48, 0.465]]], caption: The cat strolls closer to the middle of the table.
        Frame_6: [["cat", [0.406, 0.365, 0.506, 0.465]]], caption: The cat approaches the central area of the table, continuing its path.
        Frame_7: [["cat", [0.432, 0.365, 0.532, 0.465]]], caption: The cat walks steadily, nearing the halfway point.
        Frame_8: [["cat", [0.457, 0.365, 0.557, 0.465]]], caption: The cat passes the midpoint of the table.
        Frame_9: [["cat", [0.483, 0.365, 0.583, 0.465]]], caption: The cat continues towards the right-hand side.
        Frame_10: [["cat", [0.509, 0.365, 0.609, 0.465]]], caption: The cat walks steadily, enjoying its journey across the table.
        Frame_11: [["cat", [0.534, 0.365, 0.634, 0.465]]], caption: The cat inches closer to the right edge of the table.
        Frame_12: [["cat", [0.56, 0.365, 0.66, 0.465]]], caption: The cat maintains its steady pace, nearing the table's far end.
        Frame_13: [["cat", [0.586, 0.365, 0.686, 0.465]]], caption: The cat calmly continues to the right side of the table.
        Frame_14: [["cat", [0.611, 0.365, 0.711, 0.465]]], caption: The cat moves along the edge, nearing the final stretch.
        Frame_15: [["cat", [0.637, 0.365, 0.737, 0.465]]], caption: The cat continues, with its destination now in sight.
        Frame_16: [["cat", [0.663, 0.365, 0.763, 0.465]]], caption: The cat nears the table's far-right edge, still walking steadily.
        Frame_17: [["cat", [0.688, 0.365, 0.788, 0.465]]], caption: The cat's journey across the table approaches completion.
        Frame_18: [["cat", [0.714, 0.365, 0.814, 0.465]]], caption: The cat steps closer to the end of the table, walking steadily.
        Frame_19: [["cat", [0.74, 0.365, 0.84, 0.465]]], caption: The cat is almost at the right edge of the table.
        Frame_20: [["cat", [0.765, 0.365, 0.865, 0.465]]], caption: The cat carefully approaches the end of the table.
        Frame_21: [["cat", [0.791, 0.365, 0.891, 0.465]]], caption: The cat reaches the edge, preparing to step off.
        Frame_22: [["cat", [0.817, 0.365, 0.917, 0.465]]], caption: The cat looks ahead, nearing the final step on the table.
        Frame_23: [["cat", [0.842, 0.365, 0.942, 0.465]]], caption: The cat takes one of its last steps on the table.
        Frame_24: [["cat", [0.868, 0.365, 0.968, 0.465]]], caption: The cat is at the edge, ready to step down.
        Frame_25: [["cat", [0.894, 0.365, 0.994, 0.465]]], caption: The cat completes its journey, stepping off the table.

        Reasoning:The cat is walking on the table, and the table's top boundary (y_0) is at 0.465 based on the background annotation. Thus, the bottom of the cat's bounding box (y_1) is fixed at 0.465 across all frames, ensuring the cat appears consistently on the table. The horizontal movement was calculated in small increments to illustrate smooth and natural walking motion. The vertical position was maintained at a fixed offset from the table's boundary for visual alignment.
        '''
    
    txt_files = os.listdir(args.prompt_path)
    background_image_annotation = args.background_image_annotation
    for file in tqdm(txt_files):
        if file[0] == '.':
            continue
        result[file] = []

        with open(os.path.join(args.prompt_path, file), "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f)):
                line = line.strip()

                caption = line #'A blue car drives past a white picket fence on a sunny day'

                anno_path = os.path.join(background_image_annotation, file.split('.')[0], normalize_number(str(i+1)), "label_processed.json") 
                json_data = json.load(open(anno_path))

                formatted_str = "[\n"
                formatted_str += ",\n".join(
                    [f'    {{"label": "{item["label"]}", "box": {item["box"]}}}' for item in json_data]
                )
                formatted_str += "\n]"
                
                user_prompt = f'''
                    User: Provide bounding box coordinates for the prompt: {caption}

                    Background box annotation: {formatted_str}
                    '''
                
                prompt = system_prompt + user_prompt
                
                image_path = os.path.join(background_image_annotation, file.split('.')[0], normalize_number(str(i+1)), "/raw_image.jpg")  
            
                base64_image = encode_image(image_path)
                
                payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 4096
                }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                output_text = response.json()["choices"][0]["message"]["content"]
                
                result[file].append(output_text)
                
            with open(output_file, "w") as f:
                json.dump(result, f)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--background_image_annotation", type=str, required=True, help="Path to the background image grounding annotation")
    args = parser.parse_args()

    main(args)
    
