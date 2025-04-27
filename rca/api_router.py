import json
import os
import yaml
import time

from botocore.exceptions import ClientError


def load_config(config_path="rca/api_config.yaml"):
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs

configs = load_config()

def OpenAI_chat_completion(messages, temperature):    
    from openai import OpenAI
    client = OpenAI(
        api_key=configs["API_KEY"]
    )
    return client.chat.completions.create(
        model = configs["MODEL"],
        messages = messages,
        temperature = temperature,
    ).choices[0].message.content

def Google_chat_completion(messages, temperature):
    import google.generativeai as genai
    genai.configure(
        api_key=configs["API_KEY"]
    )
    genai.GenerationConfig(temperature=temperature)
    system_instruction = messages[0]["content"] if messages[0]["role"] == "system" else None
    messages = [item for item in messages if item["role"] != "system"]
    messages = [{"role": "model" if item["role"] == "assistant" else item["role"], "parts": item["content"]} for item in messages]
    history = messages[:-1]
    message = messages[-1]
    return genai.GenerativeModel(
        model_name=configs["MODEL"],
        system_instruction=system_instruction
        ).start_chat(
            history=history if history != [] else None
            ).send_message(message).text

def Anthropic_chat_completion(messages, temperature):
    import anthropic
    client = anthropic.Anthropic(
        api_key=configs["API_KEY"]
    )
    return client.messages.create(
        model=configs["MODEL"],
        messages=messages,
        temperature=temperature
    ).content


#  aws bedrock
def AWS_bedrock_completion(messages, temperature):
    import boto3
    session = boto3.session.Session(profile_name=configs["AWS_PROFILE"])
    client = session.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )

    bedrock_message = []
    for msg in messages:
        content = msg['content']
        role = msg['role']
        bedrock_message.append({
            "role": 'assistant' if role == 'system' else role,
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    try:
        response = client.invoke_model(
            modelId=configs["MODEL"],
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10000,
                "temperature": temperature,
                "messages": bedrock_message
            })
        )
        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the response text.
        response_text = model_response["content"][0]["text"]
        return response_text
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke, Reason: {e}")
        raise e


# for 3-rd party API which is compatible with OpenAI API (with different 'API_BASE')
def AI_chat_completion(messages, temperature):    
    from openai import OpenAI
    client = OpenAI(
        api_key=configs["API_KEY"],
        base_url=configs["API_BASE"]
    )
    return client.chat.completions.create(
        model = configs["MODEL"],
        messages = messages,
        temperature = temperature,
    ).choices[0].message.content

def get_chat_completion(messages, temperature=0.0):
    def send_request():
        if configs["SOURCE"] == "AI":
            return AI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "OpenAI":
            return OpenAI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Google":
            return Google_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Anthropic":
            return Anthropic_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Bedrock":
            return AWS_bedrock_completion(messages, temperature)
        else:
            raise ValueError("Invalid SOURCE in api_config file.")
    
    for i in range(3):
        try:
            return send_request()
        except Exception as e:
            print(e)
            if '429' in str(e):
                print("Rate limit exceeded. Waiting for 1 second.")
                time.sleep(1)
                continue
            else:
                raise e