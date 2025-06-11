# Geoff 2.0 - A Hyperliquid Market Making Bot

Geoff 1.0 was a Feed-Forward neural network made in my connectionist computing class in university named after Old Top Gear's infamous homemade EV. Geoff is the name for things which are ambitious but rubbish.

I originally started market making using [Hummingbot](https://hummingbot.org/).
It is a far larger, better made, more proven market making framework that offers so so so so much more. 

That said, I decided in my naivety that I should make my own project which would be smaller and specific to Hyperliquid so that I slowly expand functionality and get a handle on market making.

I also asked...what if market maker but AI? So I built in a LTSM Neural Network that is trained on previous trading data fetched from the Hyperliquid perpetuals market in order to look at sequences of trading candles and use its training to predict the optimal spreads to place the take-profit and stop-loss orders. 

**Geoff 2.0 offers market-making for a single perpetual asset on the Hyperliquid exchange with NN-powered spread prediction. Geoff is not containerized at this time. It comes with a basic TUI, analytics and paper trading.**

This was my first foray into vibe coding. I used Cursor (the $20 a month subscription) and found that it could fill out the bulk of code but fell at many logical hurdles, ignored context of instructions, over-engineered solutions and imported nonexistent or unneccessary packages. However, the worst of the problems was found when debugging API calls and using the Hyperliquid python SDK. My boy cursor got stuck in recursive loops of problem and solution.
It was still extremely useful and allowed me to put this together in a day or two. 

Geoff 2.0 does not self-train and currently isn't a particularly advanced model. I haven't done enough analytics to determine if Geoff 2.0 even makes a difference in market-making prediction. Currently Geoff 2.0 is trained on a specified timeframe packaged into sequences.

In the current example config.yaml youll see:
  timeframe: 60  - This means we are fetching the 1 min/60 second candle data for training off of
  sequence_length: 10 - This means we will be looking at the trends over the course of 10, 1 minute candles or 10 minute periods with ten points of data
  lookback_period: 2880 - This is the number of seconds to lookback for and collect sequences. Given this configuration this means we are training off the last 
  48 hours or (60 * 48). This is the number of minutes in 48 hours. 
If you changed the timeframe, sequence and lookback period you could train the model on hourly, daily, weekly, monthly, etc trends and sequences. Up until whenever Hyperliquid launched that is. Also remember the longer the sequence and lookbackperiod the more strain you put on the rate-limited Hyperliquid API.


## Project Structure
```
hl_mm_bot/
│
├── analysis/           # Storage of the analysis charts and csv's after exported
├── logs/               # Logging of each bot trading session. 
    ├── sessions/       # Each session is labeled and timestamped then stored here
    └── bot.log         # Log file with the console outputs of the current/most recent running bot. This is gitignored but will appear at runtime
├── models/             # The AI model(s) used in the project. In this case an LSTM Spread Predictor 
├── tests/              # Test files - these are not formal tests but instead POC files
├── utils/              # Utility files - various utility python files
├── env.example/        # Example .env file
├── .gitignore/         # Gitignore
├── config.yaml         # The core config file with the parameters for training and trading 
├── market_maker.py     # The file with live market making capabilities
├── paper_trader.py     # Uses live markets and function to test but not execute trade configs
├── README.md           # This file
├── requirements.txt    # Python Project dependencies
├── run_bot_gui.py      # Python file that contains the main TUI and bot runnning interface
└── run_bot.sh          # Shell file for running the bot through the TUI
```

## Setup

0. Make sure you have python3 installed. 

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env.local` file in the root directory with your configuration:
```
HYPERLIQUID_API_KEY=your_api_key
HYPERLIQUID_API_SECRET=your_api_secret
```

3. Be sure to configure the config.yaml file with your market-making and NN parameters

4. Option A: Permission and run the shell command provided. 
Instructions provided for Unix/macOS. Too lazy to open the windows laptop right now
```bash
chmod +x run_bot.sh # Permission the shell file 
# and then
./run_bot.sh
```

4. Option B: Run market_maker.py or paper_trader.py as regular python files 
First, create and activate the virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows - But I wouldnt know, I didnt try windows
```
Then run the python file:
```bash
python market_maker.py
# OR
python paper_trader.py #untested because Im lazy
```

## Use
Geoff 2.0 will open with a number of options to select. 
- "Start Live Trading",         # Begins live trading on Hyperliquid with the given configuration
- "Start Paper Trading",        # Begins paper trading simulating Hyperliquid with the given configuration 
- "Stop Bot",                   # Only appears when bot is running, stops the bot
- "View Bot Output",            # Opens up the console and shows raw bot console output. Press ctrl + c to exit
- "Analyze Trading Session",    # Opens a menu where you select a particular trade session to be analyzed. Graphs will appear one at a time
- "Generate Test Session",      # Debug option that creates a fake session to check the analysis feature
- "Exit"                       # Exit the bot, will ask to quit the bot if it is still running

Note 1: If you exit and determine to leave the bot running in the background, you will have to restart the app to select "Stop Bot" to stop the bot.
Or if you have the PID then you can terminate it manually.

Note 2: If you exit with a position still open, the bot should automatically place a matching set of take-profit and stop-loss orders to minimize loss if you forget to close the position manually on the Hyperliquid GUI. 

## License
MIT
