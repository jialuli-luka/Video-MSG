import base64
import requests
import json 
import argparse
from tqdm import tqdm 
import os 



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def main(args):
    # OpenAI API Key
    api_key = args.api_key 
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    output_file = args.output_file
    result = dict()
    
    txt_files = os.listdir(args.prompt_path)
    for file in tqdm(txt_files):
        if file[0] == '.':
            continue
        result[file] = []

        with open(os.path.join(args.prompt_path, file), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                caption = line #'A blue car drives past a white picket fence on a sunny day'
        
                prompt = f'''
                    Prompt: You need to provide a detailed description of the background for a given event. The description should be about the environment, lighting, and setting, without including any moving objects or entities. The background description should not include any key objects that are mentioned in the prompt (e.g., don't include objA/objB if the prompt is like objA is left to objB, a white objA beside a green objB). The background description should not exceed 50 words. 

                    Example:

                    User: Provide a background description for the event: A ball is bouncing on the ground.

                    Assistant: Background Description:
                    The event takes place in a park with a flat, grassy ground. Trees line the horizon, and a blue sky with light clouds stretches above.
                    User: Provide a background description for the event: {caption}
                    Assistant:
                    '''
                    
                    
                payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt
                        }
                    ]
                    }
                ],
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
    args = parser.parse_args()

    main(args)
    
