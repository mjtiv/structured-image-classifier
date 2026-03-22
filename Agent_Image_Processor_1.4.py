#!/usr/bin/env python3.10

import base64
import json
import mimetypes
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI


########################################
# PIPELINE OVERVIEW
#
# 1. Load image files from directory
# 2. Encode images to base64
# 3. Send to OpenAI Vision API
# 4. Enforce structured JSON output
# 5. Collect predictions + confidence
# 6. Save results to JSON
########################################



#########################################################################################################################
################################################## Process Images  #####################################################
#########################################################################################################################



def image_file_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/png"

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def classify_image(client: OpenAI, image_path: Path, allowed_labels: list[str]) -> dict:
    image_data_url = image_file_to_data_url(image_path)

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are an image classifier. Return only strict JSON."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Classify the main animal in this image into one of these labels: "
                            + ", ".join(allowed_labels)
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "animal_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "predicted_label": {
                            "type": "string",
                            "enum": allowed_labels,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["predicted_label", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
    )

    data = json.loads(response.output_text)
    data["filename"] = image_path.name

    if response.usage:
        data["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return data


def analyze_images(location_images, allowed_labels, output_json, client, results):

    """
    Loops over the folder of images and labels them

    """

    # Pathway and File Name for Output
    output_json_file = location_images / output_json


    # Monitor Tokens
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    # Loops over the images to label
    for image_path in sorted(location_images.glob("*.png")):
        try:
            result = classify_image(client, image_path, allowed_labels)
            results.append(result)

            if "usage" in result:
                total_tokens  += result["usage"]["total_tokens"]
                input_tokens  += result["usage"]["input_tokens"]
                output_tokens += result["usage"]["output_tokens"]

            print(f"{image_path.name} -> {result['predicted_label']} ({result['confidence']:.2f})")
        except Exception as exc:
            print(f"ERROR processing {image_path.name}: {exc}")

    output_json_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {output_json_file}")

    return total_tokens, input_tokens, output_tokens


#########################################################################################################################
############################################ Main Function ##############################################################
#########################################################################################################################

def main():

    ################################Parameters#############################
    
    # Folder of of Images
    location_images = Path(r"D:\Coding_Agents\Image_Agentic_Processing\labeled_images")

    # Labels for the AI to use
    allowed_labels = ["cat", "dog", "bear", "chicken", "fish", "iguana", "giraffe", "raccoon",
                     "octopus", "owl"]

    # Output File Name (will be placed with images)
    output_json = "classification_results.json"

    # Estimated token costs (03/21/2025)
    INPUT_COST_PER_TOKEN = 0.0000025
    OUTPUT_COST_PER_TOKEN = 0.000015

    #######################################################################

    print ("\n")
    print ("Starting Program")
    print ("\n")


    # Calling the API client
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not found")
    client = OpenAI(api_key=API_KEY)
    
    # List to Store Results
    results = []

    print ("Starting to Analyze Images")
    # Calls the main analysis function
    total_tokens, input_tokens, output_tokens = analyze_images(location_images, allowed_labels, output_json, client, results)
    print ("Done Analyzing Images")

    num_images = len(results)
    avg_tokens = total_tokens / num_images if num_images else 0
    input_cost  = input_tokens * INPUT_COST_PER_TOKEN
    output_cost = output_tokens * OUTPUT_COST_PER_TOKEN
    total_cost  = input_cost + output_cost

    print("\n--- Run Summary ---")
    print(f"Images processed: {num_images}")
    print(f"Total tokens: {total_tokens}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Avg tokens/image: {avg_tokens:.2f}")
    print(f"Input cost:  ${input_cost:.4f}")
    print(f"Output cost: ${output_cost:.4f}")
    print(f"Total cost:  ${total_cost:.4f}")

    print ("\n")
    print ("\n")
    print ("Done Running Program")

          
if __name__ == "__main__":
    main()   












