# n8n Customer Support Workflow

## Setup

1. Open n8n at http://localhost:5678
2. Create a new workflow with these nodes:

### Workflow Structure

```
Webhook (POST /webhook/support)
  → HTTP Request: GET http://lfx-server:8400/state
  → Set: system_prompt = {{ $json.system_prompt }}
  → AI Agent (OpenAI Chat Model)
      ├─ Model: gpt-4o-mini
      ├─ Temperature: 0.2
      └─ System Message: {{ $json.system_prompt }}
  → HTTP Request: POST http://lfx-server:8400/ingest
      Body: {
        "messages": [
          {"role": "system", "content": "{{ system_prompt }}"},
          {"role": "user", "content": "{{ $json.message }}"},
          {"role": "assistant", "content": "{{ $json.output }}"}
        ],
        "metadata": {
          "conversation_id": "{{ $json.execution_id }}",
          "model": "gpt-4o-mini"
        }
      }
  → Respond to Webhook: {{ $json.output }}
```

### Credentials Needed
- OpenAI API key (for the AI Agent node)

### Testing
Send a test message:
```bash
curl -X POST http://localhost:5678/webhook/support \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order #1234?"}'
```
