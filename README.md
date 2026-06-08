# Taiwan Transportation LINE Bot

AI-powered transportation planning assistant for Taiwan, deployed on Render.

## Architecture

```
LINE App → LINE Platform → Render (FastAPI)
                                ↓
                          Groq LLM (Llama 4 Scout)
                                ↓
                        交通建議回覆 → LINE Reply
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your actual keys
```

### 3. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test with ngrok

```bash
ngrok http 8000
# Set the ngrok URL as webhook in LINE Developer Console
```

## Deploy to Render

1. Push this repo to GitHub
2. Connect repo in [Render Dashboard](https://dashboard.render.com/)
3. Set environment variables in Render
4. Set webhook URL in LINE Developer Console: `https://<your-app>.onrender.com/webhook`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LINE_CHANNEL_SECRET` | LINE Messaging API channel secret |
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Messaging API channel access token |
| `GROQ_API_KEY` | Groq API key for LLM |

## Accounts Needed

1. [LINE Developers](https://developers.line.biz/) - Messaging API Channel
2. [Groq Console](https://console.groq.com/) - API Key
3. [Render](https://render.com/) - Deployment
4. [UptimeRobot](https://uptimerobot.com/) - Keep-alive (free)
