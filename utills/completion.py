from google import genai
from google.genai import types

client = genai.Client()


def generate_completion(prompt):
    answer = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1000,
            temperature=0.3,
        )
    )
    return answer.text