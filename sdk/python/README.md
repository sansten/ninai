# Ninai Python SDK

Python SDK for the Ninai Cognitive OS platform.

## Installation

```bash
pip install ninai
```

## Quick Start

```python
import asyncio
from ninai import NinaiClient

async def main():
    # Initialize the client
    client = NinaiClient(api_key="your-api-key")
    
    # Access resources
    async with client:
        # Get organization details
        org = await client.organizations.get("org-id")
        print(org)
        
        # Create a memory
        memory = await client.memories.create(
            organization_id="org-id",
            content="Important information",
            memory_type="episodic"
        )
        print(memory)

# Run the example
asyncio.run(main())
```

## Features

- **Cognitive Resources**: Access organizational cognitive models and reasoning capabilities
- **Memory Management**: Create, retrieve, and manage episodic and semantic memories
- **Goal Tracking**: Define and monitor goals within the cognitive system
- **Webhook Integration**: Real-time event streaming from the cognitive system
- **Type-Safe**: Full type annotations with Pydantic models
- **Async/Await**: Built on asyncio for high-concurrency applications

## Resources

- **Memories**: Create and manage memories (episodic, semantic)
- **Cognition**: Access cognitive models and reasoning chains
- **Goals**: Define organizational and personal goals
- **Webhooks**: Register event listeners for real-time updates
- **Organizations**: Manage multi-tenant organization contexts

## Authentication

Get your API key from the [Ninai Dashboard](https://dashboard.ninai.dev).

```python
client = NinaiClient(api_key="your-api-key", base_url="https://api.ninai.dev")
```

## Documentation

Full documentation available at [https://docs.ninai.dev/python-sdk](https://docs.ninai.dev/python-sdk)

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [https://github.com/sansten/ninai/issues](https://github.com/sansten/ninai/issues)
- Email: support@ninai.dev
- Documentation: [https://docs.ninai.dev](https://docs.ninai.dev)
