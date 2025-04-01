import os
import time
import pickle
import openai
import dotenv
from openai import OpenAI

dotenv.load_dotenv()
# Make sure you've installed openai>=1.0.0:
#   pip install --upgrade openai
# Also ensure you have the environment variable set:
#   export OPENAI_API_KEY="sk-..."

PROMPTS_FILE = "prompts.pickle"
RESPONSES_FILE = "responses.pickle"

# Replace this with a valid model name (e.g. "o4-mini", "text-davinci-003", etc.)
MODEL_NAME = "gpt-o4-mini"

def main():
    # 1) Load the OpenAI key from environment
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # 2) Load prompts
    with open(PROMPTS_FILE, "rb") as f:
        prompts = pickle.load(f)[:5]
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

    # 3) Generate responses using the Completion endpoint
    responses = []
    print("\nGenerating responses...")
    for i, prompt_text in enumerate(prompts, start=1):
        try:
            # Create a completion. With openai>=1.0.0, use openai.Completion.
            completion = client.responses.create(
                model=MODEL_NAME,
                instructions="You are scRNA-seq expert that assists in generating textual descriptions of single cells based on their metadata",
                input=prompt_text,
            )

            # The generated text is in completion.choices[0].text
            response_text = completion.output_text.strip()

            print(f"[{i}/{len(prompts)}] Response length: {len(response_text)} chars")

            # Collect results (store original prompt plus the response)
            responses.append({
                "prompt": prompt_text,
                "response": response_text
            })

        except Exception as e:
            print(f"Error generating response for prompt {i}: {e}")
            responses.append({
                "prompt": prompt_text,
                "response": None,
                "error": str(e)
            })

        # Optional: Sleep to avoid hitting rate limits if you have many prompts
        time.sleep(0.2)

    # 4) Save the responses to a file
    with open(RESPONSES_FILE, "wb") as f:
        pickle.dump(responses, f)

    print(f"\nDone. Saved {len(responses)} responses to {RESPONSES_FILE}.")

if __name__ == "__main__":
    main()
