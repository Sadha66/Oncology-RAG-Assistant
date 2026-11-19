
import os
from openai import OpenAI

api_key = os.getenv("NEXUS_API_KEY")
base_url = os.getenv("NEXUS_BASE_URL")

print("API Key available:", bool(api_key))
print("Base URL:", base_url)

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": "Say Nexus API connected successfully"
        }
    ],
    temperature=0
)

print("\nAPI Response:")
print(response.choices[0].message.content)
