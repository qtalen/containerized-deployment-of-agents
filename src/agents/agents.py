import asyncio
from typing import Sequence, AsyncGenerator
from pathlib import Path
import shutil
import os

from dotenv import load_dotenv
from autogen_core import CancellationToken
from autogen_agentchat.base import TaskResult
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseChatMessage, BaseAgentEvent, TextMessage
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.docker_jupyter import DockerJupyterCodeExecutor
from autogen_ext.code_executors.docker_jupyter._jupyter_server import JupyterConnectionInfo

from src.utils.openai_like import OpenAILikeChatCompletionClient
from src.utils.project_path import get_project_root

from src.agents.prompts import SYS_PROMPT

load_dotenv(get_project_root() / ".env")

BINDING_DIR = Path(get_project_root()/"temp")


class DataAnalysis:
    def __init__(
            self,
            model_name: str = "qwen3-max-preview",
    ):
        self._model_name = model_name
        self._model_client = None
        self._jupyter_server = None
        self._executor = None
        self._agent = None

        BINDING_DIR.mkdir(parents=True, exist_ok=True)
        self._init_jupyter_server()
        self._init_assistant()

    async def run(
            self,
            *,
            task: str | BaseChatMessage | Sequence[BaseChatMessage] | None = None,
            cancellation_token: CancellationToken | None = None,
            file_name: str | None = None,
            file_path: Path | str | None = None,
    ) -> TaskResult:
        async for message in self.run_stream(
            task=task,
            cancellation_token=cancellation_token,
            file_name=file_name,
            file_path=file_path,
        ):
            if isinstance(message, TaskResult):
                return message
        raise ValueError("No task result output.")

    async def run_stream(
            self,
            *,
            task: str | BaseChatMessage | Sequence[BaseChatMessage] | None = None,
            cancellation_token: CancellationToken | None = None,
            file_name: str | None = None,
            file_path: Path | str | None = None,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]:
        file_name = await self._copy_file(file_name, file_path)

        input_messages = []
        if isinstance(task, str):
            input_messages.append(TextMessage(
                source="user",
                content=task
            ))
        elif isinstance(task, BaseChatMessage):
            input_messages.append(task)

        if file_name is not None:
            print(file_name)
            input_messages.append(TextMessage(
                source="user",
                content=f"The input file is `{file_name}`"
            ))

        async for message in self._agent.run_stream(
                task=input_messages,
                cancellation_token=cancellation_token):
            yield message

    async def start(self):
        await self._executor.start()

    async def stop(self):
        await self._model_client.close()
        await self._executor.stop()

    async def __aenter__(self) -> "DataAnalysis":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def _init_jupyter_server(self) -> None:
        self._executor = DockerJupyterCodeExecutor(
            jupyter_server=JupyterConnectionInfo(
                host=os.getenv("JUPYTER_HOST", "127.0.0.1"),
                use_https=False,
                port=8888,
                token='UNSET',
            ),
            output_dir=BINDING_DIR,
            timeout=300,
        )

    def _init_assistant(self) -> None:
        self._model_client = OpenAILikeChatCompletionClient(
            model=self._model_name,
            temperature=0.5,
            top_p=0.85,
        )

        tool = PythonCodeExecutionTool(self._executor)

        self._agent = AssistantAgent(
            'assistant',
            model_client=self._model_client,
            tools=[tool],
            model_client_stream=True,
            system_message=SYS_PROMPT,
            max_tool_iterations=30,
        )

    @staticmethod
    async def _copy_file(
        file_name: str | None = None,
        file_path: Path | str | None = None,
    ) -> Path | str | None:
        if file_path is None:
            return None

        if file_name is None:
            file_name = Path(file_path).name
        dst_path = BINDING_DIR / file_name
        await asyncio.to_thread(shutil.copy2, file_path, dst_path)
        return file_name
