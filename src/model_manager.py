from groq import Groq
import os

class HFModelManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def chat_completion(self, messages, max_tokens=2048, temperature=0.7, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

def get_planner_model():
    return HFModelManager("llama-3.3-70b-versatile")

def get_research_model():
    return HFModelManager("llama-3.3-70b-versatile")

def get_writer_model():
    return HFModelManager("llama-3.3-70b-versatile")

def get_editor_model():
    return HFModelManager("llama-3.3-70b-versatile")
