# curelynx-backend

## Useful debugging commands
1. Curl the backend service to get trails:
```curl -X POST "https://curelynx-backend.vercel.app/api/get_trials" -H "Content-Type: application/json" -d '{"question":"hello i have this disease"}'```