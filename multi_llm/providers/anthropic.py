import datetime
import json
from abc import ABC
from typing import Dict, List, Optional

import xmltodict
from openai.types import chat

from anthropic import Anthropic, AnthropicVertex
from pydantic import BaseModel

from multi_llm.providers import base_provider


def get_finish_reason(stop_reason, stop_sequence):
    if stop_reason == "end_turn":
        return "stop"
    elif stop_reason == "stop_sequence" and stop_sequence == "</function_calls>":
        return "tool_calls"
    else:
        return "stop"


def extract_tool_calls(completion):
    content = completion.content[0].text
    tool_calls = None
    if completion.stop_sequence == "</function_calls>":
        tool_calls_dict = xmltodict.parse(completion.content[0].text + "</function_calls>")

        if isinstance(tool_calls_dict['function_calls'], dict):
            tool_calls_dict['function_calls'] = [tool_calls_dict['function_calls']]

        content = None
        tool_calls = []
        for function_call in tool_calls_dict['function_calls']:
            tool_calls.append(chat.ChatCompletionMessageToolCall(
                id=completion.id,
                function={
                    "arguments": json.dumps(function_call["invoke"]["parameters"]),
                    "name": function_call["invoke"]["tool_name"],
                },
                type="function",
            ))
    return content, tool_calls


def add_tools_to_system(system, tools):
    system = system + f"""
In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
"""

    def function_parameter_xml(parameter_dict):
        out = ""
        for key, value in parameter_dict.items():
            out += "<parameter>\n"
            out += f"<name>{key}</name>\n"
            out += f"<type>{value['type']}</type>\n"
            out += f"<description>{value['description']}</description>\n"
            out += "</parameter>\n"
        return out

    for tool in tools:
        system = system + f"""<tool_description>
<tool_name>{tool["function"]["name"]}</tool_name>
<description>{tool["function"]["description"]}</description>
<parameters>
{function_parameter_xml(tool["function"]["parameters"]["properties"])}</parameters>
</tool_description>
"""
    system += "</tools>\n"
    return system


class BaseAnthropicProvider(base_provider.BaseProvider, ABC):
    def count_tokens(self, messages, model_name) -> int:
        pass

    def create_completion(self,
                          messages,
                          model_name,
                          tools: List[chat.ChatCompletionToolParam] = None,
                          **kwargs) -> chat.ChatCompletion:
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
        else:
            system = ''

        input_messages = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                msg = msg.dict(exclude_none=True)

            if msg.get("role", None) == "system":
                continue
            elif msg.get("role", None) == "tool":
                func_results = """
                <function_results>
<result>
<tool_name>function_name</tool_name>
<stdout>
function result goes here
</stdout>
</result>
</function_results>
                """
                input_messages[-1].content += func_results
            else:
                if msg.get('tool_calls') is not None:
                    del msg['tool_calls']
                input_messages.append(msg)

        stop_sequences = ["\n\nHuman:", "\n\nAssistant"]

        if tools is not None:
            system = add_tools_to_system(system, tools)
            stop_sequences.append("</function_calls>")

        completion = self.client.messages.create(
            max_tokens=4096,
            system=system,
            messages=input_messages,
            model=model_name,
            stop_sequences=stop_sequences,
        )

        content, tool_calls = extract_tool_calls(completion)

        return chat.ChatCompletion(
            id=completion.id,
            choices=[
                {
                    "message": {"content": content, "role": completion.role, "tool_calls": tool_calls},
                    "finish_reason": get_finish_reason(completion.stop_reason, completion.stop_sequence),
                    "index": 0,
                }
            ],
            created=datetime.datetime.now().timestamp(),
            model=model_name,
            object="chat.completion",
            usage={
                "completion_tokens": completion.usage.output_tokens,
                "prompt_tokens": completion.usage.input_tokens,
                "total_tokens": completion.usage.input_tokens + completion.usage.output_tokens,
            }
        )


class AnthropicProvider(BaseAnthropicProvider):
    def __init__(self):
        self.client = Anthropic()

    def get_models(self) -> Dict[str, Dict[str, float]]:
        return {
            "claude-3-haiku-20240307": {"price": 0.25},
            "claude-3-sonnet-20240229": {"price": 3.0},
        }


class AnthropicVertexProvider(BaseAnthropicProvider):
    def __init__(self):
        LOCATION = "europe-west4"
        self.client = AnthropicVertex(region=LOCATION, project_id="smooth-guru-350515")

    def get_models(self) -> Dict[str, Dict[str, float]]:
        return {
            "claude-3-haiku@20240307": {"price": 0.25},
            "claude-3-sonnet@20240229": {"price": 3.0},
        }
