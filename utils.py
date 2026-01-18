import json
import requests
from typing import Union, List, Dict
import os
from contextlib import redirect_stdout, redirect_stderr
import logging
import httpx
from openai import OpenAI, DefaultHttpxClient
from together import Together

# Custom transport to bypass SSL verification
transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=False)

# Create a DefaultHttpxClient instance with the custom transport
http_client = DefaultHttpxClient(transport=transport)


def print_object_properties(obj: Union[dict, list]) -> None:
    t = ''
    if isinstance(obj, dict):
        keys = list(obj.keys())
        keys.sort()
        for x in keys:
            y = obj[x]
            if x == 'article_content':
                t += f'{x}: {y[:100]}...(truncated)\n'
            elif x == 'main_vector':
                t+= f'{x}: {y[:30]}...(truncated)\n'
            elif x == 'chunk':
                t+= f'{x}: {y[:100]}...(truncated)\n'

            else:
                t+= f'{x}: {y}\n'
    else:
        for l in obj:
            print_object_properties(l)
        
    print(t)
    
    

# Define utility functions and classes
def generate_embedding(prompt: str, model: str = "BAAI/bge-base-en-v1.5", together_api_key = None, **kwargs):
    payload = {
        "model": model,
        "input": prompt,
        **kwargs
    }
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        client = OpenAI(
    api_key = '', # Set any as dlai proxy does not use it. Set the together api key if using the together endpoint
    base_url="http://proxy.dlai.link/coursera_proxy/together/", # If using together endpoint, add it here https://api.together.xyz/
   http_client=http_client, # ssl bypass to make it work via proxy calls, remove it if running with together.ai endpoint 
)
        try:
            json_dict = client.embeddings.create(**payload).model_dump()
            return json_dict['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key=together_api_key)
        try:
            json_dict = client.embeddings.create(**payload).model_dump()
            return json_dict['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}")


def generate_with_single_input(prompt: str, 
                               role: str = 'user', 
                               top_p: float = None, 
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str = "meta-llama/llama-3.2-3b-instruct",
                               api_key: str = None,
                               base_url: str = "https://openrouter.ai/api/v1",
                               **kwargs):
    """
    A provider-agnostic function for OpenRouter, Together, OpenAI, etc.
    """
    
    # 1. Resolve API Key: Priority: Argument > Environment Variable
    final_api_key = api_key or os.environ.get('OPENROUTER_API_KEY') or os.environ.get('TOGETHER_API_KEY')
    
    if not final_api_key:
        raise ValueError("No API key found. Pass it as an argument or set OPENROUTER_API_KEY environment variable.")

    # 2. Build the Payload (Only include non-None values to avoid Pydantic errors)
    payload = {
        "model": model,
        "messages": [{"role": role, "content": prompt}],
        "max_tokens": max_tokens,
        **kwargs
    }
    if top_p is not None: payload["top_p"] = top_p
    if temperature is not None: payload["temperature"] = temperature

    # 3. Set Headers (Includes OpenRouter specific headers which are ignored by others)
    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000", # Required for some OpenRouter free models
        "X-Title": "Dissertation Project"
    }

    # 4. Make the Request
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    response = requests.post(endpoint, json=payload, headers=headers)

    if not response.ok:
        raise Exception(f"API Error ({response.status_code}): {response.text}")

    try:
        json_dict = response.json()
        choice = json_dict['choices'][-1]['message']
        return {
            'role': choice['role'],
            'content': choice['content']
        }
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected response format: {json_dict}. Error: {e}")


def generate_with_multiple_input(messages: List[Dict], 
                                 top_p: float = None, 
                                 temperature: float = None,
                                 max_tokens: int = 500,
                                 model: str = "meta-llama/llama-3.2-3b-instruct", 
                                 api_key: str = None,
                                 base_url: str = "https://openrouter.ai/api/v1",
                                 **kwargs):
    """
    Universal LLM caller for multi-turn conversations (chat history).
    Compatible with OpenRouter, Together, OpenAI, etc.
    """
    
    # 1. Resolve API Key
    final_api_key = api_key or os.environ.get('OPENROUTER_API_KEY') or os.environ.get('TOGETHER_API_KEY')
    
    if not final_api_key:
        raise ValueError("No API key found. Pass 'api_key' or set OPENROUTER_API_KEY in environment.")

    # 2. Build the Payload
    # We only add top_p and temperature if they are explicitly provided as numbers.
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if temperature is not None:
        payload["temperature"] = float(temperature)

    # 3. Set Universal Headers
    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000", # Helps avoid blocks on OpenRouter free models
        "X-Title": "Dissertation Project"
    }

    # 4. Make the Request
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        
        if not response.ok:
            raise Exception(f"API Error ({response.status_code}): {response.text}")

        json_dict = response.json()
        
        # 5. Standardize Output Extraction
        # This works for any OpenAI-compatible API response
        choice = json_dict['choices'][-1]['message']
        return {
            'role': choice.get('role', 'assistant'),
            'content': choice.get('content', '')
        }
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error while calling LLM: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to parse LLM response. Error: {e}\nResponse: {response.text}")




def print_properties(item):
    print(
        json.dumps(
            item.properties,
            indent=2, sort_keys=True, default=str
        )
    )


import ipywidgets as widgets
from IPython.display import display, Markdown

def display_widget(llm_call_func, semantic_search_retrieve, bm25_retrieve, hybrid_retrieve, semantic_search_with_reranking):
    def on_button_click(b):
        query = query_input.value
        top_k = slider.value
        rerank_property = rerank_property_dropdown.value
        # Clear existing outputs
        for output in [output1, output2, output3, output4, output5]:
            output.clear_output()
        status_output.clear_output()
        # Display "Generating..." message
        status_output.append_stdout("Generating...\n")
        # Update outputs one by one
        retrievals = [
            (output1, llm_call_func, query, semantic_search_retrieve, True),
            (output4, llm_call_func, query, semantic_search_with_reranking, True, True, rerank_property),
            (output2, llm_call_func, query, bm25_retrieve, True),
            (output3, llm_call_func, query, hybrid_retrieve, True),
            (output5, llm_call_func, query, None, False)  # Without RAG
        ]
        for output, func, query, retriever, use_rag, *extra_info in retrievals:
            use_rerank = extra_info[0] if len(extra_info) > 0 else False
            if use_rerank:
                rr_property = extra_info[1]
            else:
                rr_property = None
            response = func(query=query, top_k=top_k, use_rag=use_rag, retrieve_function=retriever, use_rerank=use_rerank, rerank_property=rr_property)
            with output:
                display(Markdown(response))
        # Clear "Generating..." message
        status_output.clear_output()
    
    query_input = widgets.Text(
        description='',
        value="Tell me about United States and Brazil's relationship over the course of 2024. Provide links for the resources you use in the answer.",
        placeholder='Type your query here',
        layout=widgets.Layout(width='70%')  # Adjusted width to make room for label
    )
    
    query_label = widgets.Label(value="Query:", layout=widgets.Layout(width='10%', align='center'))
    query_box = widgets.HBox([query_label, query_input], layout=widgets.Layout(align_items='center'))  # Positioned side by side

    slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Top K:',
        style={'description_width': 'initial'}
    )
    
    rerank_property_dropdown = widgets.Dropdown(
        options=['question', 'answers'],
        value='question',
        description='Rerank Property:',
        style={'description_width': 'initial'}
    )
    
    output_style = {'border': '1px solid #ccc', 'width': '100%'}
    output1 = widgets.Output(layout=output_style)
    output2 = widgets.Output(layout=output_style)
    output3 = widgets.Output(layout=output_style)
    output4 = widgets.Output(layout=output_style)
    output5 = widgets.Output(layout=output_style)
    status_output = widgets.Output()
    
    submit_button = widgets.Button(
        description="Get Responses",
        style={'button_color': '#eee', 'font_color': 'black'}
    )
    submit_button.on_click(on_button_click)
    
    label1 = widgets.Label(value="Semantic Search")
    label2 = widgets.Label(value="BM25 Search")
    label3 = widgets.Label(value="Hybrid Search")
    label4 = widgets.Label(value="Semantic Search with Reranking")
    label5 = widgets.Label(value="Without RAG")
    
    display(widgets.HTML("""
    <style>
        .custom-output {
            background-color: #f9f9f9;
            color: black;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .widget-text, .widget-button {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ddd !important;
        }
        .widget-output {
            background-color: #f9f9f9 !important;
            color: black !important;
        }
        input[type="text"] {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ddd !important;
        }
    </style>
    """))
    
    display(query_box, slider, rerank_property_dropdown, submit_button, status_output)
    
    # Create individual vertical containers for each label and output
    vbox1 = widgets.VBox([label1, output1], layout={'width': '30%'})
    vbox2 = widgets.VBox([label2, output2], layout={'width': '30%'})
    vbox3 = widgets.VBox([label3, output3], layout={'width': '30%'})
    vbox4 = widgets.VBox([label4, output4], layout={'width': '30%'})
    vbox5 = widgets.VBox([label5, output5], layout={'width': '30%'})
    
    # HBoxes to arrange VBoxes in rows
    hbox_outputs1 = widgets.HBox([vbox1, vbox4, vbox2], layout={'justify_content': 'space-between'})
    hbox_outputs2 = widgets.HBox([vbox3, vbox5], layout={'justify_content': 'space-between'})
    
    def style_outputs(*outputs):
        for output in outputs:
            output.layout.margin = '5px'
            output.layout.height = '300px'
            output.layout.padding = '10px'
            output.layout.overflow = 'auto'
            output.add_class("custom-output")
            
    style_outputs(output1, output2, output3, output4, output5)
    
    # Display rows with outputs
    display(hbox_outputs1)
    display(hbox_outputs2)

def make_url():
    #lab_id = x
    url = f"https://app.phoenix.arize.com/s/aamini8118"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    print(f"{BOLD}FOLLOW THIS URL TO OPEN THE UI: {url}{RESET}")