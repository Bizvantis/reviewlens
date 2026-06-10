import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_explanation(counterfactual_result: dict, user_reviews: list = None) -> str:
    original_asin = counterfactual_result["original_asin"]
    cf_asin = counterfactual_result["counterfactual_asin"]
    magnitude = counterfactual_result["perturbation_magnitude"]
    steps = counterfactual_result["steps_taken"]
    original_score = counterfactual_result["original_score"]
    new_score = counterfactual_result["new_score"]

    review_context = ""
    if user_reviews:
        review_context = f"""
The user has written these past reviews:
{chr(10).join([f'- "{r}"' for r in user_reviews[:3]])}
"""

    prompt = f"""
You are an AI system that explains product recommendations to Amazon customers in plain English.

A recommendation system currently recommends product {original_asin} to this user (confidence score: {original_score}).
The system found that a small shift in the user's preferences (magnitude: {magnitude}, found in {steps} steps)
would cause the system to recommend {cf_asin} instead (new score: {new_score}).
{review_context}

Write a clear, friendly 3-sentence explanation for the customer that:
1. Tells them WHY they are being recommended {original_asin} right now
2. Explains what small preference shift would lead to {cf_asin} being recommended instead
3. Ends with an empowering insight about their own taste

Rules:
- No jargon, no mention of embeddings, vectors, or scores
- Speak directly to the customer as "you"
- Keep it conversational and under 80 words
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    sample_result = {
        "user_idx": 0,
        "original_asin": "B015X3CXXA",
        "original_idx": 8264,
        "counterfactual_asin": "B01DPCONFM",
        "counterfactual_idx": 9069,
        "steps_taken": 27,
        "perturbation_magnitude": 0.0027,
        "original_score": 0.1215,
        "new_score": 0.1082
    }

    sample_reviews = [
        "Great guitar, perfect for beginners",
        "Amazing sound quality for the price",
        "Bought this as a gift, loved it"
    ]

    print("Generating explanation...\n")
    explanation = generate_explanation(sample_result, sample_reviews)
    print("--- Groq Explanation ---")
    print(explanation)