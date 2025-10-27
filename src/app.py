import re
import json

import chainlit as cl
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, ToolCallRequestEvent

from src.agents.agents import DataAnalysis

import logging
from autogen_core import EVENT_LOGGER_NAME

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.ERROR)


@cl.on_chat_start
async def on_chat_start():
    init_message = cl.Message(content="Please upload your data, and I'll help you analyze it.")
    await init_message.send()

    assistant = DataAnalysis()
    await assistant.start()
    cl.user_session.set("assistant", assistant)

@cl.on_stop
async def on_chat_stop():
    print("chat stop")

@cl.on_chat_end
async def on_chat_end():
    assistant = cl.user_session.get("assistant")
    await assistant.stop()


@cl.on_message
async def on_message(message: cl.Message):

    input_msg = message.content
    file_path = None
    file_name = None
    if len(message.elements)>0:
        file_path = message.elements[0].path
        file_name = message.elements[0].name
    assistant: DataAnalysis = cl.user_session.get("assistant")

    output_str = ''
    output_imgs = None
    output_files = None
    output_msg = cl.Message(content='')
    async for event in assistant.run_stream(
        task=input_msg,
        file_name=file_name,
        file_path=file_path,
    ):
        if isinstance(event, ModelClientStreamingChunkEvent):
            output_str += event.content
            output_files = extract_file_paths(output_str)
            await output_msg.stream_token(event.content)
        elif isinstance(event, ToolCallRequestEvent):
            for call in event.content:
                codeblock = f"\n```python\n{json.loads(call.arguments)['code']}\n```\n"
                await output_msg.stream_token(codeblock)


    if output_files:
        for file in output_files:
            output_msg.elements.append(
                cl.File(path="temp/" + file, name=file, display="inline")
            )
    await output_msg.update()


def extract_file_paths(text: str) -> list[str]:
    """
    Extract all file paths from the text within [file]...[/file] tags.
    Args:
        text (str): Input text, which may contain zero or more [file] tags.
    Returns:
        list[Path]: The extracted list of file paths, each element is a pathlib.Path object.
    """
    pattern = r'\[file\](.*?)\[/file\]'
    matches = re.findall(pattern, text)
    return [str(match.strip()) for match in matches]