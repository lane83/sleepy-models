# Sleepy-Models

A brain-inspired LLM system that implements dream states for efficient memory consolidation and knowledge processing. This system manages LLM interactions with built-in rest periods, similar to human sleep cycles, to optimize performance and resource usage.

## Features

- Dream state for memory consolidation
- Multi-model support (OpenAI, Anthropic, HuggingFace)
- Usage tracking and cost management
- Performance monitoring and optimization
- Knowledge graph integration
- Automated memory management
- Rate limit handling
- User-friendly interface

## Installation

```bash
git clone https://github.com/yourusername/sleepy-models.git
cd sleepy-models
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_huggingface_key
```

## Usage

```python
from sleepy_models import ModelManager

# Initialize the system
manager = ModelManager()

# Start a conversation
response = manager.process_message("Your prompt here")

# System will automatically manage dream states and memory consolidation
```

## Dream State System

The system implements a brain-inspired dream state mechanism that:
- Consolidates memories during idle periods
- Optimizes knowledge graph connections
- Manages resource usage
- Improves response quality over time

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.