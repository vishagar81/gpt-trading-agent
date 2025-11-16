# X (Twitter) Sentiment Analysis - Quick Setup

## 1. Get Twitter API Credentials (2 minutes)

Go to [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard):

1. Log in or create account
2. Create a new app
3. Go to "Keys and tokens" tab
4. Copy:
   - **API Key** → `TWITTER_API_KEY`
   - **API Secret** → `TWITTER_API_SECRET`
   - **Access Token** → `TWITTER_ACCESS_TOKEN`
   - **Access Secret** → `TWITTER_ACCESS_SECRET`
   - **Bearer Token** → `TWITTER_BEARER_TOKEN`

## 2. Update .env File (1 minute)

```
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret
TWITTER_BEARER_TOKEN=your_bearer_token
```

## 3. Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

## 4. Test (30 seconds)

Run the app:
```bash
streamlit run trading-agent-claude.py
```

Type in chat:
```
get x sentiment AAPL
```

## Usage

```
get x sentiment AAPL          # By ticker
get x sentiment #Tesla        # By hashtag
get x sentiment NVDA stock    # By keywords
twitter sentiment MSFT        # Alternative syntax
```

## Output

```
📈 Sentiment: BULLISH
📊 Score: 0.72 (-1 to +1 scale)
💬 Tweets Analyzed: 87
⏱️ Period: Last 7 days
Summary: [Claude's analysis]
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Twitter API credentials not set` | Add `TWITTER_BEARER_TOKEN` to .env |
| `ModuleNotFoundError: tweepy` | `pip install tweepy>=4.14.0` |
| `No recent tweets found` | Try different ticker format |
| Rate limit | Wait 15 minutes, try again |

## What It Does

- Fetches last 7 days of tweets mentioning the stock
- Analyzes top 20 tweets by engagement
- Uses Claude to generate BULLISH/NEUTRAL/BEARISH sentiment
- Shows sentiment score from -1 (bearish) to +1 (bullish)
- Only analyzes original tweets (no retweets)

## Done! 🚀

Your X sentiment analysis is ready to use.