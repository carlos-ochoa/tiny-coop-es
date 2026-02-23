"""Custom metrics for the LLM as a judge evaluation approach
"""

from mlflow.genai import make_judge
from typing import Literal
from src.utils import ConfigManager

config = ConfigManager("config.yaml").config

judge_model = config["evaluation"]["judge_model"]

is_gramatically_correct = make_judge(
    name="is_gramatically_correct",
    instructions=("Review whether the provided text is written following grammar quality in Spanish."
        "Evaluate using a scale of 1 to 5, being 1 : really bad grammar and 5 : excellent expert level grammar." \
        "Text to evaluate : {{ outputs }}" 
        "Provide only the numeric score (1-5)."
    ),
    model=judge_model,
    feedback_value_type=Literal[1,2,3,4,5]
)

is_understandable = make_judge(
    name="is_understandable",
    instructions=("Review the provided text to check the story in Spanish is actually understandable "
                  "by a 3-4 year-old kid. The text must be using words that a kid would use and understand. "
                  "Evaluate using a scale of 1 to 5, being 1: difficult to understand by a kid and 5: easily understandable."
                 "Text to evaluate : {{ outputs }}" 
                "Provide only the numeric score (1-5)."
    ),
    model=judge_model,
    feedback_value_type=Literal[1,2,3,4,5]
)

uses_the_required_setup = make_judge(
    name="uses_the_required_setup",
    instructions=("Review the provided text to check the story in Spanish is written following the given setup {{ inputs }}"
                  "Evaluate using a scale of 1 to 3, being 1: does not use the setup at all and 3: implements the setup correctly."
                  "Text to evaluate : {{ outputs }}" 
                "Provide only the numeric score (1-5)."
    ),
    model=judge_model,
    feedback_value_type=Literal[1,2,3]
)

uses_the_vocabulary = make_judge(
    name="uses_the_vocabulary",
    instructions=("Review the provided text to check the story in Spanish uses the words in the vocabulary "
                  "verb: {{ inputs }}, noun : {{ inputs }}, adjective : {{ inputs }}"
                  "Evaluate with a Yes or No"
                  "Text to evaluate : {{ outputs }}" 
    ),
    model=judge_model,
    feedback_value_type=Literal["yes", "no"]
)

scorers = {
    'is_gramatically_correct' : is_gramatically_correct,
    'is_understandable' : is_understandable,
    'uses_the_required_setup' : uses_the_required_setup,
    'uses_the_vocabulary' : uses_the_vocabulary
}