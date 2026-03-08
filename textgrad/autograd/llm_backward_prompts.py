GLOSSARY_TEXT_BACKWARD = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable."""

### Backward engine prompts

# System prompt to the backward engine.
# BACKWARD_SYSTEM_PROMPT = (
#     "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. "
#     "Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
#     "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
#     "Pay attention to the role description of the variable, and the context in which it is used. You should assume that the variable will be used in a similar context in the future. "
#     "Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients'). "
#     "For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\n"
#     "If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.\n"
#     "IMPORTANT: Be extremely concise. Do not use filler words. Limit response to 4-5 sentences.\n"
#     f"{GLOSSARY_TEXT_BACKWARD}")

# BACKWARD_SYSTEM_PROMPT = (
#     "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. "
#     "Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
#     "The variables may be solutions to problems, code, or frequently, **prompts to language models**. "
#     "Pay attention to the type of variable. **If the variable is a Prompt**, you must go beyond general advice. Your feedback must target a **Grounded Heuristic** (Policy + Specific Content):\n"
#     "1. **Identify the Logic Gap:** What specific reasoning step or domain knowledge is missing?\n"
#     "2. **Propose a Content-Anchored Policy:** Do not just propose a rule; propose a rule *containing* specific features. (e.g., instead of 'Verify neighbors', say 'Policy: If Center is distinct, verify if neighbors contain specific features like *desmoplastic stroma* or *fatty tissue* to confirm boundary').\n"
#     "3. **Propose Injectable Knowledge:** Provide the specific 'memorizable' content—such as a concrete few-shot example or a list of high-value features (e.g., 'Add a few-shot example where the agent explicitly checks for *mitotic figures* in the UP tile')—that implements your policy.\n"
#     "For non-prompt variables, provide strategies to fix specific failure modes relative to the objective.\n"
#     "DO NOT propose a full new version of the variable; that is the job of the optimizer. Your job is to compute the 'gradient' (the specific logic gap + the concrete features to fix it). "
#     "IMPORTANT: Be extremely concise. Limit response to 4-5 sentences. Structure: Diagnosis -> Grounded Policy -> Injectable Knowledge.\n"
#     f"{GLOSSARY_TEXT_BACKWARD}")


BACKWARD_SYSTEM_PROMPT = """You are the Gradient Feedback Engine in an advanced optimization system. Your objective is to provide deep, structural feedback (textual gradients) to improve a target variable based on an evaluation.

You will be evaluating either an [AGENT OUTPUT] or a [SYSTEM PROMPT]. You must identify which one you are evaluating based on its role description and apply the correct logic path below.

=========================================
PATH A: CRITIQUING AN [AGENT OUTPUT]
=========================================
If the variable is an Agent's Output:
1. Focus on the mistake it make when referencing the neighbors, The main objective is not to blindly comform to the ground truth label but to correctly infer the information in the neighbors' context.
2. Convey the message passing label in the graident(e.g. over-interpretation, under-interpretation, wrong referencing... etc)
3. Identify exactly *what* the output missed, hallucinated, or misclassified when referencing the neighboring tiles. 
4. Do not rewrite the output. Simply state the exact logical or factual failure in 200 words.

=========================================
PATH B: CRITIQUING A [SYSTEM PROMPT]
=========================================
If the variable is a System Prompt, you must perform a rigorous Root Cause Analysis using the following structured thinking process. 

First, determine the direction of the gradient:
[POSITIVE GRADIENT (Success Case)]
If the evaluation shows the prompt succeeded: Identify exactly which rule, policy, or phrasing in the prompt successfully handled the input. Your feedback must state: "Keep and reinforce [Specific Rule] because it correctly handled [Specific Context]."

[NEGATIVE GRADIENT (Failure Case)]
If the evaluation shows a failure, you MUST output a <scratchpad> thinking block followed by your <final_gradient>. 

Inside your <scratchpad>, complete these exact steps:
1. FAULT ATTRIBUTION: Is the error solely caused by the Prompt? Is it a combination of a tricky Input + the Prompt? Or is the Input fundamentally flawed/impossible?
2. MISTAKE CLASSIFICATION: If the Prompt is at fault, categorize the mistake:
   - Policy Mistake: Is the methodology/rule of thumb fundamentally wrong or too simplistic to extract the right info?
   - Phrasing Mistake: Is the instruction ambiguous, causing the LLM to misunderstand?
   - Edge Case/Few-Shot: Is the prompt lacking an example for this specific, rare input type?
3. DRAFT FIX: Propose a specific, concrete change to the prompt's logic (e.g., "Refine the spatial policy to explicitly check for X when Y occurs").
4. SIMULATION: Role-play the Agent. Read the original input, apply your DRAFT FIX methodology, and simulate the output. 
5. EVALUATION: Does your simulated output fix the original error? If YES, proceed. If NO, write a <revised_fix> that addresses the gap.

=========================================
FINAL OUTPUT FORMAT
=========================================
After your reasoning, provide your feedback. 
- DO NOT propose a complete rewritten version of the variable (the optimizer does that).
- Your gradient must be highly specific, actionable, and grounded in the input context.
- Limit the final gradient to 4-5 concise sentences.
- Limit the total output to 800 words

<scratchpad>
(Your step-by-step reasoning for Path B failures goes here)
</scratchpad>

<final_gradient>
(Your concise, actionable feedback goes here)
</final_gradient>
""" + f"\n{GLOSSARY_TEXT_BACKWARD}"

# First part of the prompt for the llm backward function
CONVERSATION_TEMPLATE = (
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_INPUT> {prompt} </LM_INPUT>\n\n"
    "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
)

# Has the gradient on the output.
CONVERSATION_START_INSTRUCTION_CHAIN = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is a conversation with a language model (LM):\n\n"
    "{conversation}"
)
OBJECTIVE_INSTRUCTION_CHAIN = (
    "This conversation is part of a larger system. The <LM_OUTPUT> was later used as {response_desc}.\n\n"
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable to address the following feedback on the LM_OUTPUT: {response_gradient} </OBJECTIVE_FUNCTION>\n\n"
)

# Does not have gradient on the output
CONVERSATION_START_INSTRUCTION_BASE = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is an evaluation of the variable using a language model:\n\n"
    "{conversation}"
)

OBJECTIVE_INSTRUCTION_BASE = (
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. "
    "Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
)

# Third part of the prompt for the llm backward function.
# Asks the user to evaluate a variable in the conversation.
EVALUATE_VARIABLE_INSTRUCTION = (
    "We are interested in giving feedback to the {variable_desc} "
    "for this conversation. Specifically, give feedback to the following span "
    "of text:\n\n<VARIABLE> "
    "{variable_short} </VARIABLE>\n\n"
    "Given the above history, describe how the {variable_desc} "
    "could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, intelligent and concise.\n\n"
)

SEARCH_QUERY_BACKWARD_INSTRUCTION = (
    "Here is a query and a response from searching with {engine_name}:\n"
    "<QUERY> {query} </QUERY>\n"
    "<RESULTS> {results} </RESULTS>\n\n"
)


GRADIENT_OF_RESULTS_INSTRUCTION = (
    "For the search results from {engine_name} we got the following feedback:\n\n"
    "<FEEDBACK>{results_gradient}</FEEDBACK>\n\n"
)

IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = (
    "You must base on the following examples when give feedback and criticism to the variable:\n\n"
    "<EXAMPLES>{in_context_examples}</EXAMPLES>\n\n"
)
