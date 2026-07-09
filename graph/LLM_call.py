from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def JSON_Model(model_name: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    return response.choices[0].message.content


def Model(model_name: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content