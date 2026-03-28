You have access to the following function:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"}
    },
    "required": ["city"]
  }
}
```

Call `get_weather` with `city` set to `"London"`.

Write your response as a JSON array to `/app/result.json`.
