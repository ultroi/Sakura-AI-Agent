from groq import Groq
from config import load_settings

def main():
    print("Loading settings...")
    settings = load_settings()
    client = Groq(api_key=settings.groq_api_key)

    print("\nFetching available models from Groq...")
    models_response = client.models.list()
    available_models = [model.id for model in models_response.data]

    print("\n--- ALL AVAILABLE MODELS FOR YOUR API KEY ---")
    for model_id in available_models:
        print(f"- {model_id}")

    # A ranked list of the best Groq models for Sakura's Tool Calling
    ideal_backups = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]

    # Check which of our ideal backups your API key actually has access to
    valid_backups = [m for m in ideal_backups if m in available_models]

    print("\n\n" + "="*50)
    print(" SUCCESS! COPY AND PASTE THIS INTO YOUR agent.py")
    print("="*50)
    print("        # --- NEW: Smart Fallback Router ---")
    print("        fallback_models = [")
    print('            "openai/gpt-oss-20b",  # 1. Primary')
    
    # Add the top 3 best available backup models to the code
    for count, backup in enumerate(valid_backups[:3], start=2):
        print(f'            "{backup}",  # {count}. Fallback')
        
    print("        ]")
    print("="*50)

if __name__ == "__main__":
    main()