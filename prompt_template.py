from jinja2 import Template
from task_prompt import prompt


def render_prompt_from_string(syllogism):
    template_string = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a logic engine. Analyze the syllogism below.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Syllogism: {{syllogism}}

    Instructions:
    1. In the 'thought_process' field, I want you to: {{prompt}}
    2. Then, determine the final 'validity' (true/false).
    3. Finally, provide a concise 'reason'.
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    
    template = Template(template_string)
    
    # Render the template with the provided context variables
    rendered_prompt = template.render(syllogism=syllogism, prompt=prompt)
        
    return rendered_prompt

