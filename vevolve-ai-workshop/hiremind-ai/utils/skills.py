# utils/skills.py

import json
from openai import OpenAI

client = OpenAI()

def extract_skills(text):

    prompt = f"""
You are an AI skill extraction engine.

Extract:
- technical skills
- frameworks
- tools
- cloud technologies
- programming languages
- databases

from the following text.

TEXT:
{text}

Return ONLY valid JSON list.

Example:
["python", "aws", "docker"]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    try:
        skills = json.loads(content)

    except:
        skills = []

    # clean duplicates

    skills = list(set([
        s.lower().strip()
        for s in skills
    ]))

    return skills