"""
Translation utilities for Vietnamese to English
"""
from googletrans import Translator
import asyncio

def translate_vi_to_en(text: str) -> str:
    """
    Translate Vietnamese text to English
    
    Args:
        text: Vietnamese text
        
    Returns:
        English translation
    """
    try:
        translator = Translator()
        result = translator.translate(text, src='vi', dest='en')
        
        # Handle coroutine if returned
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(result)
            loop.close()
        
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        # Fallback to original text if translation fails
        return text


def translate_en_to_vi(text: str) -> str:
    """
    Translate English text to Vietnamese
    
    Args:
        text: English text
        
    Returns:
        Vietnamese translation
    """
    try:
        translator = Translator()
        result = translator.translate(text, src='en', dest='vi')
        
        # Handle coroutine if returned
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(result)
            loop.close()
        
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        # Fallback to original text if translation fails
        return text
