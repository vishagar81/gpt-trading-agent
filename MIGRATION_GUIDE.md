# Migration Guide: Original → Claude Version

This guide helps you switch from the original trading agent to the Claude-powered version.

## Quick Start (5 minutes)

### Step 1: Get Claude API Key (2 minutes)
1. Visit https://console.anthropic.com/
2. Click "Create API Key"
3. Copy the key starting with `sk-ant-`

### Step 2: Add to .env File (1 minute)
Open `.env` and add:
```
CLAUDE_API_KEY=sk-ant-your-key-here
```

### Step 3: Install SDK (1 minute)
```bash
pip install anthropic
```

### Step 4: Run Claude Version (1 minute)
```bash
streamlit run trading-agent-claude.py
```

Done! 🎉

## Side-by-Side Comparison

### Original Version
```bash
streamlit run trading-agent.py
```
- Uses: Perplexity (stock discovery) + OpenAI GPT-5 & GPT-4 (analysis)
- Requires: 3 API keys (Perplexity, OpenAI, QuantiQ)
- Best for: If you already have OpenAI credits

### Claude Version
```bash
streamlit run trading-agent-claude.py
```
- Uses: Claude Sonnet 4.5 for all AI tasks
- Requires: 1 API key (Claude)
- Best for: Better analysis, simpler setup, larger context window

## What's the Same?

✅ **Identical features:**
- Portfolio tracking (same CSV format)
- Technical analysis charts
- Backtesting engine
- Event calendar
- Macro indicators
- Reddit sentiment analysis
- News fetching
- Alpaca trading integration
- All keyboard commands

✅ **Shared data:**
- `portfolio.csv` is used by both versions
- Trading history carries over when switching versions

## What's Different?

| Aspect | Original | Claude |
|--------|----------|--------|
| Stock discovery prompt | Perplexity's algorithm | Claude's reasoning |
| Analysis quality | Good | Better |
| Response length | Concise | More detailed |
| Reasoning depth | Standard | Superior |
| Available context | Standard | Much larger (200K tokens) |
| Learning curve | Moderate | Easy (simpler setup) |

## API Key Comparison

### Original Version (.env)
```
PERPLEXITY_API_KEY=ppl-xxx
OPENAI_API_KEY=sk-xxx
QUANTIQ_API=xxx
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
```

### Claude Version (.env)
```
CLAUDE_API_KEY=sk-ant-xxx
QUANTIQ_API=xxx
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
```

**Better**: Fewer API keys to manage!

## Cost Comparison

### Original Version
- Perplexity: ~$0.001-0.005 per request
- OpenAI GPT-5: ~$0.0005-0.005 per 1K tokens
- Total for typical analysis: $0.01-0.02

### Claude Version
- Claude Sonnet 4.5: ~$0.003 per 1K input + ~$0.015 per 1K output
- Typical analysis: $0.005-0.01
- **With Batch API**: ~50% discount on costs

**Winner**: Claude, especially with Batch API

## Performance Comparison

### Stock Discovery ("find stocks")
- **Original**: Perplexity returns quick, basic list
- **Claude**: Returns thoughtful list with brief reasoning

### Analysis ("analyze AAPL")
- **Original**: GPT-5 provides recommendation + justification
- **Claude**: More detailed analysis with better financial reasoning

### Multi-factor Analysis (auto-pilot)
- **Original**: Analyzes financials + news + sentiment
- **Claude**: Same + better integration of multiple data sources due to larger context

### Hedging ("hedge")
- **Original**: GPT-5 proposes hedge asset
- **Claude**: Better reasoning about correlations and market dynamics

## Testing the Migration

Before fully switching, test the Claude version:

```bash
# Keep original running for comparison
streamlit run trading-agent.py

# In another terminal, test Claude version
streamlit run trading-agent-claude.py
```

Try the same commands in both versions and compare outputs.

## Rollback (If Needed)

If you want to go back to the original:

```bash
# Stop Claude version (Ctrl+C)
# Run original again
streamlit run trading-agent.py

# Your portfolio.csv is safe - it works with both versions
```

## Common Questions

### Q: Will I lose my trading history?
**A:** No! Both versions use the same `portfolio.csv` file. Your trades are preserved.

### Q: Can I use both versions simultaneously?
**A:** Yes, but they'll both access/write to the same `portfolio.csv`. It's safe but confusing. Better to use one at a time.

### Q: Is Claude better for all use cases?
**A:** Yes, especially for:
- Multi-factor analysis (auto-pilot mode)
- Complex financial reasoning
- Detailed explanations
- High-volume analysis (batch API)

The original might be better if:
- You prefer shorter responses
- You have leftover OpenAI credits
- You like Perplexity's specific approach to stock discovery

### Q: What's Claude Sonnet 4.5?
**A:** It's Anthropic's latest model - a good balance of speed and capability. Perfect for financial analysis.

### Q: Can I customize the prompts?
**A:** Yes! Edit the functions in `trading-agent-claude.py`:
- `get_small_cap_stocks()` - customize stock discovery
- `get_stock_recommendation()` - customize analysis
- `hedge_portfolio()` - customize hedging strategy
- `generate_google_trends_queries()` - customize trend queries

### Q: What about rate limits?
**A:** Claude's rate limits are generous. For auto-pilot mode (multiple stocks), you might hit limits. Add small `time.sleep(0.5)` delays between requests if needed.

### Q: How do I know which version is running?
**A:** Look at the top of the page:
- Original: No special indicator
- Claude: Shows "🤖 Powered by Claude Sonnet 4.5"

## Next Steps

1. ✅ Get Claude API key
2. ✅ Add to .env
3. ✅ Install anthropic SDK
4. ✅ Run `streamlit run trading-agent-claude.py`
5. ✅ Test with "find stocks" command
6. ✅ Test with "analyze AAPL" command
7. ✅ If happy, make Claude your main version!

## Support

If you encounter issues:

1. Check CLAUDE_VERSION.md for detailed setup
2. Verify API key in .env file
3. Ensure anthropic package is installed: `pip show anthropic`
4. Check Claude Console for API usage/errors: https://console.anthropic.com/

## Summary

| Aspect | Time | Complexity | Benefit |
|--------|------|-----------|---------|
| Setup | 5 min | Very Easy | Immediate |
| Learning | 0 min | None | Identical interface |
| Testing | 10 min | Easy | Confidence to switch |
| Benefits | Ongoing | - | Better analysis quality |

**Total time to migration: ~15 minutes**

Enjoy your Claude-powered trading agent! 🚀
