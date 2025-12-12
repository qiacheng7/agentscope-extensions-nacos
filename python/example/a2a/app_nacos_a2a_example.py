# -*- coding: utf-8 -*-
"""
A2A Protocol with Nacos Service Discovery Example

This example demonstrates A2A protocol with Nacos service discovery
using A2AFastAPIDefaultAdapter with NacosRegistry.

This example showcases important A2A configuration options including:
- AgentCard configuration (name, version, transport, skills, etc.)
- Task timeout settings
- Wellknown endpoint configuration
- Transport configurations for multiple interfaces
- Registry integration with Nacos

Usage:
    python app_nacos_a2a.py

Environment Variables:
    DASHSCOPE_API_KEY: API key for DashScope model
    NACOS_SERVER_ADDR: Nacos server address (default: localhost:8848)
    NACOS_USERNAME: Nacos username (optional)
    NACOS_PASSWORD: Nacos password (optional)
"""
import asyncio
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import stream_printing_messages
from agentscope.tool import Toolkit, execute_python_code
from a2a.types import AgentSkill
from v2.nacos import ClientConfigBuilder

from agentscope_runtime.adapters.agentscope.memory import (
    AgentScopeSessionHistoryMemory,
)
from agentscope_runtime.engine.app import AgentApp
from agentscope_runtime.engine.deployers.adapter.a2a import (
    A2AFastAPIDefaultAdapter,
    A2AConfig,
    AgentCardConfig,
    TaskConfig,
    WellknownConfig,
    TransportsConfig,
    NacosRegistry,
)
from agentscope_runtime.engine.deployers.local_deployer import (
    LocalDeployManager,
)
from agentscope_runtime.engine.schemas.agent_schemas import (
    AgentRequest,
)
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)

# Setup project paths
_file_path = Path(__file__).resolve()
# example/a2a/app_nacos_a2a_example.py -> example/a2a -> example -> python
project_root = _file_path.parent.parent.parent

# Load environment variables from .env file
env_file = project_root / ".env"
print(f"Looking for .env at: {env_file}")
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"✓ Loaded .env from: {env_file}")
    # Verify loaded values
    print(f"  NACOS_SERVER_ADDR={os.getenv('NACOS_SERVER_ADDR', 'NOT SET')}")
    print(f"  NACOS_USERNAME={os.getenv('NACOS_USERNAME', 'NOT SET')}")
else:
    print(f"✗ Warning: .env file not found at {env_file}")
    print(f"  Will use default values or system environment variables")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Configuration constants
DEFAULT_PORT = 8099
DEFAULT_HOST = "0.0.0.0"
AGENT_NAME = "Friday"
AGENT_DESCRIPTION = "A helpful assistant"


def create_agent_skills() -> List[AgentSkill]:
    """Create and return list of agent skills.

    Returns:
        List of AgentSkill objects
    """
    return [
        AgentSkill(
            id="dialog",
            name="Natural Language Dialog Skill",
            description=(
                "Enables natural language conversation "
                "and dialogue with users"
            ),
            tags=["natural language", "dialog", "conversation"],
            examples=[
                "Hello, how are you?",
                "Can you help me with something?",
            ],
        ),
        AgentSkill(
            id="code_execution",
            name="Code Execution Skill",
            description="Can execute Python code to solve problems",
            tags=["code", "python", "execution"],
            examples=[
                "Calculate 2+2",
                "Write a function to sort a list",
            ],
        ),
    ]


# Create NacosRegistry with explicit configuration
# Read from environment variables loaded from .env file
nacos_server_addr = os.getenv("NACOS_SERVER_ADDR", "localhost:8848")
nacos_username = os.getenv("NACOS_USERNAME")
nacos_password = os.getenv("NACOS_PASSWORD")
nacos_namespace_id = os.getenv("NACOS_NAMESPACE_ID", "public")

print(f"Nacos configuration:")
print(f"  Server: {nacos_server_addr}")
print(f"  Namespace: {nacos_namespace_id}")
print(f"  Auth: {'enabled' if nacos_username else 'disabled'}")

# Build Nacos client config
builder = ClientConfigBuilder().server_address(nacos_server_addr)
if nacos_namespace_id:
    builder.namespace_id(nacos_namespace_id)
if nacos_username and nacos_password:
    builder.username(nacos_username).password(nacos_password)

nacos_client_config = builder.build()
nacos_registry = NacosRegistry(nacos_client_config=nacos_client_config)

# Configure A2A protocol with a2a_config
a2a_config = A2AConfig(
    registry=nacos_registry,
    agent_card=AgentCardConfig(
        card_name=AGENT_NAME,
        card_version="1.0.0",
        preferred_transport="JSONRPC",
        skills=create_agent_skills(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        provider={
            "organization": "AgentScope Runtime",
            "url": "https://runtime.agentscope.io",
        },
        document_url="https://runtime.agentscope.io",
    ),
    task=TaskConfig(
        task_timeout=60,
        task_event_timeout=10,
    ),
    wellknown=WellknownConfig(
        wellknown_path="/.wellknown/agent-card.json",
    ),
    transports=TransportsConfig(
        transports=[
            {
                "name": "JSONRPC",
                "url": f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/a2a",
                "rootPath": "/",
                "subPath": "/a2a",
                "tls": False,
            },
        ],
    ),
)

agent_app = AgentApp(
    app_name=AGENT_NAME,
    app_description=AGENT_DESCRIPTION,
    a2a_config=a2a_config,
)


@agent_app.init
async def init_func(self):
    """Initialize services."""
    self.state_service = InMemoryStateService()
    self.session_service = InMemorySessionHistoryService()

    await self.state_service.start()
    await self.session_service.start()


@agent_app.shutdown
async def shutdown_func(self):
    """Shutdown services."""
    await self.state_service.stop()
    await self.session_service.stop()


@agent_app.query(framework="agentscope")
async def query_func(
    self,
    msgs,
    request: AgentRequest = None,
    **kwargs,
):
    """Query handler for agent requests."""
    assert kwargs is not None, "kwargs is required for query_func"
    session_id = request.session_id
    user_id = request.user_id

    state = await self.state_service.export_state(
        session_id=session_id,
        user_id=user_id,
    )

    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)

    agent = ReActAgent(
        name=AGENT_NAME,
        model=DashScopeChatModel(
            "qwen-turbo",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            enable_thinking=True,
            stream=True,
        ),
        sys_prompt=f"You're a helpful assistant named {AGENT_NAME}.",
        toolkit=toolkit,
        memory=AgentScopeSessionHistoryMemory(
            service=self.session_service,
            session_id=session_id,
            user_id=user_id,
        ),
        formatter=DashScopeChatFormatter(),
    )

    if state:
        agent.load_state_dict(state)

    async for msg, last in stream_printing_messages(
        agents=[agent],
        coroutine_task=agent(msgs),
    ):
        yield msg, last

    state = agent.state_dict()
    await self.state_service.save_state(
        user_id=user_id,
        session_id=session_id,
        state=state,
    )


async def main():
    """Main deployment function."""
    # AgentApp is already initialized with a2a_config
    # The A2A adapter is automatically created by AgentApp
    # We need to set the base_url for the adapter so it can
    # build correct agent card URL and register with the correct
    # host/port in Nacos

    a2a_adapter = agent_app.protocol_adapters[0]
    if isinstance(a2a_adapter, A2AFastAPIDefaultAdapter):
        # Set base_url for the adapter to build correct
        # agent card URL. This is used when registering with
        # Nacos registry
        a2a_adapter._base_url = (  # pylint: disable=protected-access
            f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
        )

    deploy_manager = LocalDeployManager(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
    )
    await agent_app.deploy(deploy_manager)


if __name__ == "__main__":
    BANNER = (
        "\n"
        + "=" * 60
        + "\n"
        + "A2A Protocol with Nacos Service Discovery Example\n"
        + "=" * 60
        + "\n"
        + "\nEnvironment Variables:\n"
        + "  DASHSCOPE_API_KEY: API key for DashScope model\n"
        + "  NACOS_SERVER_ADDR: Nacos server address "
        + "(default: localhost:8848)\n"
        + "  NACOS_USERNAME: Nacos username (optional)\n"
        + "  NACOS_PASSWORD: Nacos password (optional)\n"
        + "=" * 60
        + "\n"
    )
    print(BANNER)

    try:
        asyncio.run(main())
        print("\nServer is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)
