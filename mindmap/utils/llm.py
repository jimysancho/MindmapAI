import os
import httpx
import asyncio

from openai import AsyncClient, RateLimitError
from openai.types.chat import ChatCompletion


async def _acomplete_chat(
    system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini"
) -> ChatCompletion:
    
    client = AsyncClient(api_key=os.environ['OPENAI_API_KEY'])
    
    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], 
            temperature=0, 
            model=model
        )
    except RateLimitError as e:
        print("RateLimitError -> ", e)
        await asyncio.sleep(0.5)
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], 
            temperature=0, 
            model=model
        )

    return response


async def _acomplete_vision(
    prompt: str, encoded_image: str, temperature: float=0.0, model: str = 'gpt-4o', 
) -> ChatCompletion:
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    payload = {
        "model": model, 
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}", 
                        }, 
                    }
                ]
            }
        ],
        "max_tokens": 500, 
        "temperature": temperature
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
    return ChatCompletion(**response.json())
