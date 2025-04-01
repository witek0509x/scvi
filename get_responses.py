import os
import time
import pickle
import openai

# Make sure you have `pip install openai`
# and that your OPENAI_API_KEY is set in the environment, e.g.:
#   export OPENAI_API_KEY="sk-..."

PROMPTS_FILE = "prompts.pickle"
RESPONSES_FILE = "responses.pickle"
MODEL_NAME = "o4-mini"  # or another model, if needed

def main():
    # 1) Load OpenAI key from env
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")

    # 2) Load prompts
    with open(PROMPTS_FILE, "rb") as f:
        prompts = pickle.load(f)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

    # 3) Generate responses
    responses = []
    print("Generating responses...")
    for i, prompt_text in enumerate(prompts, start=1):
        try:
            # Send the prompt to OpenAI
            completion = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=200
            )

            response_text = completion.choices[0].message["content"]
            print(f"[{i}/{len(prompts)}] Response length: {len(response_text)} chars")

            # Collect the result
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

        # OPTIONAL: Avoid hitting rate limits if you have many prompts
        time.sleep(0.5)

    # 4) Save all responses
    with open(RESPONSES_FILE, "wb") as f:
        pickle.dump(responses, f)

    print(f"Done. Saved {len(responses)} responses to {RESPONSES_FILE}")


if __name__ == "__main__":
    main()
