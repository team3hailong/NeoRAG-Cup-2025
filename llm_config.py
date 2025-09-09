"""
Simple LLM Configuration
Cấu hình đơn giản cho NVIDIA và GROQ
"""

import os
import time
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "writer/palmyra-med-70b")  
NVIDIA_ENDPOINT = os.getenv("NVIDIA_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

# Initialize client
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"✅ GROQ client ready: {GROQ_MODEL}")
    except ImportError:
        print("⚠️ GROQ library not installed")

if NVIDIA_API_KEY:
    print(f"✅ NVIDIA client ready: {NVIDIA_MODEL}")

def get_llm_response(messages: List[Dict], temperature: float = 0.0, max_tokens: int = 300, max_retries: int = 3) -> str:
    """
    Đơn giản gọi LLM - tự động chọn provider có sẵn
    """
    providers = []
    delay = 1
    if LLM_PROVIDER == "nvidia" and NVIDIA_API_KEY:
        providers.append(("NVIDIA", _call_nvidia))
    if groq_client:
        providers.append(("GROQ", _call_groq))
    for name, func in providers:
        for attempt in range(max_retries):
            try:
                return func(messages, temperature, max_tokens)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"{name} failed after {max_retries} attempts: {e}")
                else:
                    time.sleep(delay)
                    delay *= 2
    print("❌ No LLM provider available")
    return ""

def _call_nvidia(messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """Call NVIDIA API"""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    response = requests.post(NVIDIA_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def _call_groq(messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """Call GROQ API"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens
    )
    return response.choices[0].message.content

def get_config_info() -> Dict:
    """Get current config info"""
    return {
        "provider": LLM_PROVIDER,
        "model": NVIDIA_MODEL if LLM_PROVIDER == "nvidia" else GROQ_MODEL,
        "api_key_available": bool(NVIDIA_API_KEY if LLM_PROVIDER == "nvidia" else GROQ_API_KEY),
        "client_initialized": bool(NVIDIA_API_KEY or groq_client)
    }
