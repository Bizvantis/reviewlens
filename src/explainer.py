import os
from dotenv import load_dotenv
from groq import Groq
from src.product_lookup import get_product_info

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from src.product_lookup import get_product_info

def generate_explanation(counterfactual_result: dict, user_reviews: list = None) -> str:
    original_asin = counterfactual_result["original_asin"]
    cf_asin = counterfactual_result["counterfactual_asin"]
    magnitude = counterfactual_result["perturbation_magnitude"]
    steps = counterfactual_result["steps_taken"]
    original_score = counterfactual_result["original_score"]
    new_score = counterfactual_result["new_score"]

    # Fetch real product names
    orig_info = get_product_info(original_asin)
    cf_info = get_product_info(cf_asin)

    review_context = ""
    if user_reviews:
        review_context = f"""
The user has written these past reviews:
{chr(10).join([f'- "{r}"' for r in user_reviews[:3]])}
"""

    prompt = f"""
You are an AI system that explains product recommendations to Amazon customers in plain English.

The user is currently being recommended: "{orig_info['title']}" (ASIN: {original_asin})
This product has an average rating of {orig_info['avg_rating']} from {orig_info['rating_count']} reviews.

Our system found that a very small shift in this user's preferences would instead recommend:
"{cf_info['title']}" (ASIN: {cf_asin})
{review_context}

Write a clear, friendly 3-sentence explanation for the customer that:
1. Tells them WHY they are being recommended "{orig_info['title']}" right now based on their review history
2. Explains what small preference shift would lead to "{cf_info['title']}" being recommended instead
3. Ends with an empowering insight about what this reveals about their musical taste

Rules:
- No jargon, no mention of embeddings, vectors, or scores
- Speak directly to the customer as "you"
- Reference the actual product names, not ASINs
- Keep it conversational and under 100 words
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
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