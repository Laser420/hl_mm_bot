# Market Maker Analysis Tool

## Usage

### Interactive Mode (Recommended)
```bash
source venv/bin/activate
python analyze_session.py
```
This will show a menu of available sessions to analyze.

### Direct Session Analysis
```bash
source venv/bin/activate
python analyze_session.py <session_id>
```

Example:
```bash
python analyze_session.py paper_20250816_174948
```

## What It Shows

### Session Information
- Trading mode (Paper/Live)
- Asset traded
- Session duration
- Data availability

### Trading Performance
- Total orders placed
- Completed round-trip trades
- Win/loss statistics
- PnL breakdown (realized, fees, net)
- Volume traded

### Position Analysis
- Maximum position sizes
- Time spent in positions vs flat

### Recent Trade History
- Last 5 trades with details

## Session Data Location
Sessions are stored in: `logs/sessions/`
- Paper trading: `paper_YYYYMMDD_HHMMSS/`
- Live trading: `YYYYMMDD_HHMMSS/`

Each session contains:
- `trades.csv` - Individual trade records
- `positions.csv` - Position updates
- `metadata.json` - Session configuration