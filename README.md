# Structured Image Classifier with OpenAI Vision

A Python pipeline for batch image classification using OpenAI vision
models with:

-   strict JSON-schema output\
-   constrained label selection\
-   batch processing over a local image folder\
-   saved structured results\
-   token and cost tracking

This project demonstrates how to move from ad hoc image prompting to a
**controlled, reproducible vision pipeline**.

------------------------------------------------------------------------

## Example Run

![Run Example](Program_Running.png)

**Note:**  
`Fox_Image.png` was intentionally introduced as a *wildcard test case* and was **not included in the allowed label set**.  
As a result, the model mapped it to the closest available class (`dog`) with a lower confidence score (0.91).

This demonstrates an important behavior of constrained classification systems:
when the correct label is unavailable, the model will still attempt to select the nearest valid option rather than abstaining.

------------------------------------------------------------------------

## What this project does

The script:

1.  Loads `.png` images from a local folder\
2.  Converts each image to a base64 data URL\
3.  Sends each image to an OpenAI vision-capable model\
4.  Forces the model to return strict JSON\
5.  Restricts predictions to an allowed label set\
6.  Saves results to a `classification_results.json` file\
7.  Tracks token usage and estimated run cost

------------------------------------------------------------------------

## Why this repo exists

Most image API examples stop at "send one image and print a result."

This repo shows how to build a **structured pipeline**:

-   reproducible batch processing\
-   machine-readable outputs\
-   schema enforcement\
-   cost observability

------------------------------------------------------------------------

## Project structure

```text
structured-image-classifier/
├── README.md
├── LICENSE
├── .gitignore
├── Agent_Image_Processor_1.4.py
├── Program_Running.png
└── labeled_images/

------------------------------------------------------------------------

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/mjtiv/structured-image-classifier.git
cd structured-image-classifier

### 2. Create virtual environment

``` bash
python -m venv .venv
```

Activate:

**Windows**

``` bash
.venv\Scripts\activate
```

**Mac/Linux**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Add API key

Create `.env` file:

``` env
OPENAI_API_KEY=your_api_key_here
```

------------------------------------------------------------------------

## How to run

``` bash
python src/agent_image_processor.py
```

------------------------------------------------------------------------

## Example output

``` text
Fox_Image.png -> dog (0.91)
```

------------------------------------------------------------------------

## Example JSON output

``` json
[
  {
    "predicted_label": "dog",
    "confidence": 0.91,
    "filename": "Fox_Image.png"
  }
]
```

------------------------------------------------------------------------

## Key design choices

### Strict JSON schema

Ensures reliable parsing and downstream automation.

### Closed label set

Forces classification into a controlled set, exposing edge cases.

### Cost tracking

Tracks token usage to estimate run cost.

------------------------------------------------------------------------

## Limitations

-   PNG-only input\
-   Hardcoded label list\
-   No retry logic\
-   No evaluation metrics

------------------------------------------------------------------------

## Future improvements

-   CLI arguments for paths and labels\
-   Support JPG / WEBP\
-   CSV export\
-   Retry + logging\
-   Confusion matrix / evaluation

------------------------------------------------------------------------

## License

MIT

