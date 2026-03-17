You have access to the following function:

```json
{
  "name": "search_books",
  "description": "Search for books by query string",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "limit": {"type": "integer", "description": "Max results", "default": 10}
    },
    "required": ["query"]
  }
}
```

Call `search_books` with `query` set to `"python programming"`.

Write your response as a JSON array to `/app/result.json`.
