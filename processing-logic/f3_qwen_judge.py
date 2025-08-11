from openai import OpenAI
import base64 
import re 
import json
from PIL import Image
import os
import ast
from config.openrouter_api_keys import API_KEY_PERSO, API_KEY_COIN, API_KEY_PRO

API_KEYS = [API_KEY_PERSO, API_KEY_COIN, API_KEY_PRO]

def api_key_cycle():
    while True:
        for key in API_KEYS:
            yield key
key_gen = api_key_cycle()


def qwen_judge(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # prompt = "Detect all plants in the image and return their locations in the form of coordinates. The format of output should be like {'bbox_2d': [x1, y1, x2, y2], 'label': plantType (jade, cactus, etc ...)"
    prompt = "Detect all plants in the image and return their locations in the form of coordinates. The format of output should be like {'bbox_2d': [x1, y1, x2, y2], 'label': plantType (jade, cactus, etc ...). The bbox generated must properly frame each plant so that this crop is presentable on an application for users."
    

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # we create sub folder for this image
        image_basename = os.path.splitext(image_name)[0]
        image_output_dir = os.path.join(output_dir, image_basename)
        os.makedirs(image_output_dir, exist_ok=True)

        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")


        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=next(key_gen),
        )

        completion = client.chat.completions.create(
        extra_body={},
        model="qwen/qwen2.5-vl-32b-instruct:free",
        messages=[
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
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
                }
            ]
            }
        ]
        )

        response_text = completion.choices[0].message.content
        print(response_text)



        try:
            # First try to extract JSON from code blocks
            match = re.search(r"```json\s*(\[.*?\])\s*```", response_text, re.DOTALL)
            if match:
                json_data = json.loads(match.group(1))
            else:
                # If no code block, try to find JSON array directly in the text
                match = re.search(r"(\[.*?\])", response_text, re.DOTALL)
                if match:
                    json_data = json.loads(match.group(1))
                else:
                    print(f"No JSON array found in response for {image_name}")
                    print("Response text:", response_text)
                    continue
                    
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for {image_name}: {e}")
            print("Matched text:", match.group(1) if match else "No match")
            continue
        except Exception as e:
            print(f"Unexpected error parsing response for {image_name}: {e}")
            continue





        # match = re.search(r"```json\s*(\[.*?\])\s*```", response_text, re.DOTALL)
        # if match:
        #     json_data = ast.literal_eval(match.group(1))

        # else:
        #     raise ValueError("No JSON array found in model response.")


        output_path = os.path.join(image_output_dir, "bboxes_types.json")
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=4)

        print(f"Saved in {output_path}")


        # crop the detections from the image then save them in the ouput dir
        img = Image.open(image_path)
        for i, det in enumerate(json_data):
            bbox = det.get("bbox_2d")
            label = det.get("label", "_unknown")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                crop = img.crop((x1, y1, x2, y2))
                crop_filename = f"plant_{i+1}_{label}.jpg"
                crop.save(os.path.join(image_output_dir, crop_filename))
                print(f"Saved crop: {crop_filename}")
            else:
                print(f"Skipping detection {i} -- invalid bbox")



if __name__ == '__main__':
    IMAGE_DIR = ""
    OUTPUT_DIR = "/home/lounes/turf3/output"
    qwen_judge(IMAGE_DIR, OUTPUT_DIR)