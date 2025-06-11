#!/usr/bin/env python3
import os
import subprocess
import questionary
import psutil
import signal
from datetime import datetime
import glob
import time
from dotenv import load_dotenv
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
import sys
import numpy as np
import yaml
from market_maker import MarketMaker
from paper_trader import PaperTrader
from utils.tui import TradingTUI
from utils.trade_logger import TradeLogger
from utils.analyze_trades import TradeAnalyzer

# Load environment variables from .env.local
load_dotenv('.env.local')

# Global variables
console = Console()
bot_process = None
PID_FILE = "bot.pid"

# This doesnt work as intended but it doesnt stop the use of ctrl z, so bug > feature
def handle_suspend(signum, frame):
    """Handle terminal suspension (Ctrl+Z)"""
    if check_bot_process():
        console.print("\n[yellow]Warning: Bot is still running in the background![/yellow]")
        console.print("[yellow]Use 'fg' to bring the terminal back to foreground[/yellow]")
        console.print("[yellow]Or use the menu to stop the bot properly[/yellow]\n")

# Register signal handler for SIGTSTP (Ctrl+Z)
signal.signal(signal.SIGTSTP, handle_suspend)

def get_available_sessions():
    """Get list of available trading sessions from logs directory"""
    sessions_dir = os.path.join("logs", "sessions")
    if not os.path.exists(sessions_dir):
        console.print("[yellow]No sessions directory found![/yellow]")
        return []
        
    sessions = []
    for session_id in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, session_id)
        if os.path.isdir(session_path):
            # Check for any data files in the session directory
            has_data = any(
                os.path.exists(os.path.join(session_path, f))
                for f in ["trades.csv", "positions.csv", "performance.csv", "metadata.json"]
            )
            
            if has_data:
                # Extract trading type from session ID
                trading_type = "Paper" if session_id.startswith("paper_") else "Live"
                # Format the display name
                display_name = f"{trading_type} Trading - {session_id}"
                sessions.append((display_name, session_id))
            else:
                console.print(f"[yellow]Warning: Session {session_id} has no data files[/yellow]")
    
    if not sessions:
        console.print("\n[yellow]No trading sessions with data found![/yellow]")
        console.print("[yellow]Please run the bot or generate a test session first.[/yellow]")
    else:
        console.print(f"\n[green]Found {len(sessions)} trading sessions[/green]")
    
    return sorted(sessions, key=lambda x: x[1], reverse=True)  # Most recent first

def check_bot_process():
    """Check if bot process is still running and valid"""
    global bot_process
    
    # First check the PID file
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process exists
            try:
                os.kill(pid, 0)  # This will raise an error if process doesn't exist
                console.print(f"[green]Bot process {pid} is running[/green]")
                return True
            except (OSError, ProcessLookupError):
                # Process doesn't exist, clean up
                console.print(f"[yellow]Bot process {pid} not found, cleaning up[/yellow]")
                if os.path.exists(PID_FILE):
                    os.remove(PID_FILE)
                return False
        except (ValueError, FileNotFoundError):
            console.print("[yellow]Invalid PID file, cleaning up[/yellow]")
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
            return False
    else:
        console.print("[yellow]No PID file found[/yellow]")
    
    return False

def ensure_logs_directory():
    """Ensure logs directory exists"""
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.path.join("logs", "sessions"), exist_ok=True)

def run_bot():
    """Run the trading bot in a separate process"""
    print("\nStarting trading bot...")
    
    # Check if bot is already running
    if check_bot_process():
        print("Bot is already running.")
        return None
    
    # Check for required environment variables
    required_env_vars = ["HYPERLIQUID_ACCOUNT_ADDRESS", "HYPERLIQUID_SECRET_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease create a .env.local file with these variables:")
        print("HL_API_KEY=your_api_key")
        print("HL_API_SECRET=your_api_secret")
        return None
    
    # Clean up any stale PID file
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    
    # Ensure logs directory exists
    ensure_logs_directory()
    
    # Open log file for writing
    log_file = open("logs/bot.log", "w")
    
    try:
        # Use Popen to run the bot in background
        process = subprocess.Popen(
            ["python", "market_maker.py"],
            stdout=log_file,
            stderr=log_file,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            preexec_fn=os.setpgrp,  # Create new process group
            env=os.environ.copy()  # Pass current environment
        )
        
        # Store the process ID and process object for later reference
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))
        
        # Store the process object in a global variable
        global bot_process
        bot_process = process
        
        # Wait a moment to check if process started successfully
        time.sleep(1)
        if process.poll() is not None:
            # Process exited immediately, read the log file
            log_file.close()
            with open("logs/bot.log", "r") as f:
                error_output = f.read()
            print("Bot failed to start. Error output:")
            print(error_output)
            return None
        
        print(f"Bot started with PID: {process.pid}")
        print("Bot output is being written to logs/bot.log")
        return process
        
    except Exception as e:
        print(f"Error starting bot: {e}")
        log_file.close()
        return None

def stop_bot():
    """Stop the running bot"""
    if not check_bot_process():
        console.print("[yellow]No bot process is running![/yellow]")
        return
        
    try:
        # Read PID from file
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
            
        # Try SIGTERM first
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait a moment for process to terminate
            time.sleep(2)
            
            # Check if process is still running
            try:
                os.kill(pid, 0)
                # If we get here, process is still running, use SIGKILL
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                # Process has terminated
                pass
                
        except (OSError, ProcessLookupError):
            # Process already terminated
            pass
            
        # Cleanup
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        global bot_process
        bot_process = None
            
        console.print("[green]Bot stopped successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error stopping bot: {str(e)}[/red]")
        # Cleanup on error
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        bot_process = None

def analyze_session(session_id=None):
    """Analyze a specific trading session"""
    sessions = get_available_sessions()
    if not sessions:
        console.print("\n[yellow]No trading sessions found! Please run the bot first to generate some data.[/yellow]")
        return
        
    if session_id:
        selected_session = session_id
    else:
        # Show session selection menu
        choices = [s[0] for s in sessions]  # Show display names
        if not choices:
            console.print("\n[yellow]No trading sessions found! Please run the bot first to generate some data.[/yellow]")
            return
            
        selected_session = questionary.select(
            "Select a session to analyze:",
            choices=choices,
            use_indicator=True,
            style=questionary.Style([
                ('selected', 'fg:cyan bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
            ])
        ).ask()
        
        if not selected_session:  # User cancelled selection or pressed Enter without selection
            return
            
        # Get the actual session ID from the display name
        selected_session = next((s[1] for s in sessions if s[0] == selected_session), None)
        if not selected_session:
            console.print("[red]Invalid session selection![/red]")
            return
    
    console.print(f"\n[blue]Analyzing session: {selected_session}[/blue]")
    try:
        cmd = ["python", "utils/analyze_trades.py", "--report", "--session", selected_session]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error analyzing session: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")

def view_logs(session_id):
    """View logs for a specific session"""
    session_dir = os.path.join("logs", "sessions", session_id)
    log_files = {
        "Trades": os.path.join(session_dir, "trades.csv"),
        "Positions": os.path.join(session_dir, "positions.csv"),
        "Performance": os.path.join(session_dir, "performance.csv"),
        "Metadata": os.path.join(session_dir, "metadata.json")
    }
    
    while True:
        choice = questionary.select(
            f"View logs for session {session_id}:",
            choices=list(log_files.keys()) + ["Back"]
        ).ask()
        
        if choice == "Back":
            break
            
        file_path = log_files[choice]
        if os.path.exists(file_path):
            if file_path.endswith('.json'):
                # For JSON files, use jq if available, otherwise cat
                try:
                    subprocess.run(["jq", ".", file_path])
                except FileNotFoundError:
                    subprocess.run(["cat", file_path])
            else:
                # For CSV files, use head to show first few lines
                subprocess.run(["head", "-n", "10", file_path])
        else:
            print(f"Log file not found: {file_path}")


def view_bot_output():
    """Show raw bot output from log file"""
    if not check_bot_process():
        console.print("[yellow]Bot is not running![/yellow]")
        return
        
    log_file = "logs/bot.log"
    if not os.path.exists(log_file):
        console.print("[red]No log file found![/red]")
        return
        
    try:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Create a layout with main content and footer
        layout = Layout()
        layout.split_column(
            Layout(name="main", size=os.get_terminal_size().lines - 3),  # Leave space for footer
            Layout(name="footer")
        )
        
        # Create footer with instructions
        footer = Panel(
            "[bold blue]Press Ctrl+C to return to menu[/bold blue]",
            border_style="blue"
        )
        layout["footer"].update(footer)
        
        # Initialize output buffer
        output_buffer = []
        max_lines = os.get_terminal_size().lines - 4  # Leave space for footer and borders
        
        # Start the live display
        with Live(layout, refresh_per_second=1) as live:
            # Use tail -f to follow the log file
            process = subprocess.Popen(
                ["tail", "-f", log_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            try:
                while True:
                    # Read output from tail
                    output = process.stdout.readline()
                    if output:
                        # Add new line to buffer
                        output_buffer.append(output.strip())
                        # Keep only the last max_lines
                        if len(output_buffer) > max_lines:
                            output_buffer = output_buffer[-max_lines:]
                        # Update main content with all lines
                        layout["main"].update("\n".join(output_buffer))
                    time.sleep(0.1)
            except KeyboardInterrupt:
                process.terminate()
                console.print("\n[yellow]Returning to menu...[/yellow]")
            except Exception as e:
                process.terminate()
                console.print(f"[red]Error viewing output: {str(e)}[/red]")
            finally:
                process.terminate()
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Returning to menu...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error viewing output: {str(e)}[/red]")

def list_sessions():
    """List all available trading sessions"""
    sessions_dir = os.path.join("logs", "sessions")
    if not os.path.exists(sessions_dir):
        console.print("[yellow]No sessions found![/yellow]")
        return
        
    sessions = []
    for session_id in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, session_id)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "metadata.json")):
            sessions.append(session_id)
            
    if not sessions:
        console.print("[yellow]No sessions found![/yellow]")
        return
        
    # Sort sessions by date (newest first)
    sessions.sort(reverse=True)
    
    # Display sessions
    console.print("\nAvailable Sessions:")
    for i, session_id in enumerate(sessions, 1):
        console.print(f"{i}. {session_id}")
        
    # Let user select a session to analyze
    choice = questionary.select(
        "Select a session to analyze:",
        choices=sessions
    ).ask()
    
    if choice:
        analyze_session(choice)

def generate_test_session():
    """Generate a test session with sample trading data"""
    # Create session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("logs", "sessions", "test_session_" + session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Generate sample trades
    trades_data = {
        'timestamp': [],
        'order_id': [],
        'asset': [],
        'side': [],
        'size': [],
        'price': [],
        'position_after': [],
        'pnl': [],
        'pnl_after': []
    }
    
    # Generate 20 sample trades
    base_price = 2000.0  # Starting price
    current_position = 0
    base_timestamp = int(time.time())
    
    for i in range(20):
        timestamp = base_timestamp + i * 60  # 1 minute apart
        side = 'buy' if i % 2 == 0 else 'sell'
        size = 0.1  # Fixed size
        price = base_price + (i * 10)  # Price increases by $10 each trade
        
        # Calculate position after trade
        if side == 'buy':
            current_position += size
        else:
            current_position -= size
        
        # Calculate PnL
        pnl = 1.0 if side == 'sell' else 0.0  # Fixed $1.0 profit per completed trade
        
        trades_data['timestamp'].append(timestamp)
        trades_data['order_id'].append(f"order_{i}")
        trades_data['asset'].append("ETH")
        trades_data['side'].append(side)
        trades_data['size'].append(size)
        trades_data['price'].append(price)
        trades_data['position_after'].append(current_position)
        trades_data['pnl'].append(pnl)
        trades_data['pnl_after'].append(pnl)
    
    # Save trades
    trades_df = pd.DataFrame(trades_data)
    trades_df.to_csv(os.path.join(session_dir, "trades.csv"), index=False)
    
    # Generate positions data
    positions_data = {
        'timestamp': [],
        'asset': [],
        'size': [],
        'value': [],
        'entry_price': []
    }
    
    # Generate position updates
    current_position = 0
    for i in range(20):
        timestamp = base_timestamp + i * 60
        if i % 2 == 0:
            current_position += 0.1
        else:
            current_position -= 0.1
        
        price = base_price + (i * 10)
        value = current_position * price
        
        positions_data['timestamp'].append(timestamp)
        positions_data['asset'].append("ETH")
        positions_data['size'].append(current_position)
        positions_data['value'].append(value)
        positions_data['entry_price'].append(price if current_position > 0 else 0)
    
    # Save positions
    positions_df = pd.DataFrame(positions_data)
    positions_df.to_csv(os.path.join(session_dir, "positions.csv"), index=False)
    
    # Generate performance data
    performance_data = {
        'timestamp': [],
        'total_trades': [],
        'winning_trades': [],
        'losing_trades': [],
        'win_rate': [],
        'total_pnl': [],
        'current_position': [],
        'current_price': []
    }
    
    # Calculate performance metrics
    total_trades = 0
    winning_trades = 0
    total_pnl = 0
    
    for i in range(20):
        timestamp = base_timestamp + i * 60
        if i % 2 == 1:  # Only increment on sell trades
            total_trades += 1
            winning_trades += 1
            total_pnl += 1.0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        current_position = 0.1 if i % 2 == 0 else 0
        current_price = base_price + (i * 10)
        
        performance_data['timestamp'].append(timestamp)
        performance_data['total_trades'].append(total_trades)
        performance_data['winning_trades'].append(winning_trades)
        performance_data['losing_trades'].append(total_trades - winning_trades)
        performance_data['win_rate'].append(win_rate)
        performance_data['total_pnl'].append(total_pnl)
        performance_data['current_position'].append(current_position)
        performance_data['current_price'].append(current_price)
    
    # Save performance
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(os.path.join(session_dir, "performance.csv"), index=False)
    
    # Save metadata
    metadata = {
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "config": {
            "asset": "ETH",
            "leverage": 1.0,
            "position_size": 0.1,
            "profit_target": 0.02,
            "stop_loss": 0.01
        }
    }
    
    with open(os.path.join(session_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    console.print(f"[green]Test session generated: {session_id}[/green]")
    return session_id

def start_trading_bot(config: dict, is_paper_trading: bool = False) -> None:
    """Start the trading bot with the given configuration"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create separate directories for paper and live trading
        trading_type = "paper" if is_paper_trading else "live"
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join("logs", "sessions", f"{trading_type}_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Clean up any stale PID file
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        
        # Open log file for writing
        log_file = open("logs/bot.log", "w")
        
        # Determine which script to run
        script_name = "paper_trader.py" if is_paper_trading else "market_maker.py"
        
        # Set environment variable to indicate trading type and session directory
        env = os.environ.copy()
        env["TRADING_TYPE"] = trading_type
        env["SESSION_DIR"] = session_dir
        
        console.print(f"[blue]Starting {trading_type} trading bot...[/blue]")
        
        # Use Popen to run the bot in background
        process = subprocess.Popen(
            ["python", script_name],
            stdout=log_file,
            stderr=log_file,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            preexec_fn=os.setpgrp,  # Create new process group
            env=env  # Pass environment with trading type
        )
        
        # Store the process ID
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))
        
        # Store the process object in a global variable
        global bot_process
        bot_process = process
        
        # Wait a moment to check if process started successfully
        time.sleep(1)
        if process.poll() is not None:
            # Process exited immediately, read the log file
            log_file.close()
            with open("logs/bot.log", "r") as f:
                error_output = f.read()
            console.print("[red]Bot failed to start. Error output:[/red]")
            console.print(error_output)
            return
        
        console.print(f"[green]Bot started with PID: {process.pid}[/green]")
        console.print(f"[green]Bot output is being written to logs/bot.log[/green]")
        console.print(f"[green]Session data will be saved in: {session_dir}[/green]")
        
        # Verify the process is running
        if check_bot_process():
            console.print("[green]Bot process verified as running[/green]")
        else:
            console.print("[red]Bot process failed to start properly[/red]")
        
    except Exception as e:
        console.print(f"[red]Error starting bot: {e}[/red]")
        if 'log_file' in locals():
            log_file.close()
        return

def main_menu():
    """Display the main menu and handle user input"""
    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold blue]Geoff 2.0 Hyperliquid Market Maker Bot[/bold blue]\n"
            "[yellow]Select an option:[/yellow]",
            border_style="blue"
        ))
        
        # Check if bot is running to show appropriate options
        is_bot_running = check_bot_process()
        
        # Define all possible choices
        all_choices = [
            "Start Live Trading Bot",
            "Start Paper Trading Bot",
            "Stop Bot",
            "View Bot Output",
            "Analyze Trading Session",
            "Generate Test Session",
            "Exit"
        ]
        
        # Filter choices based on bot state
        choices = [choice for choice in all_choices if choice != "Stop Bot" or is_bot_running]
        
        try:
            choice = questionary.select(
                "What would you like to do?",
                choices=choices,
                use_indicator=True,
                style=questionary.Style([
                    ('selected', 'fg:cyan bold'),
                    ('pointer', 'fg:cyan bold'),
                    ('highlighted', 'fg:cyan bold'),
                ])
            ).ask()
            
            if not choice:  # Handle case where user presses Ctrl+C or Enter without selection
                continue
                
            if choice == "Start Live Trading Bot":
                if not check_required_env_vars():
                    console.print("[red]Missing required environment variables. Please check your .env.local file.[/red]")
                    time.sleep(2)
                    continue
                    
                config = load_config()
                if config:
                    start_trading_bot(config, is_paper_trading=False)
                    # Force a menu refresh after starting the bot
                    time.sleep(1)  # Give the process time to start
                    continue
            
            elif choice == "Start Paper Trading Bot":
                config = load_config()
                if config:
                    start_trading_bot(config, is_paper_trading=True)
                    # Force a menu refresh after starting the bot
                    time.sleep(1)  # Give the process time to start
                    continue
            
            elif choice == "Stop Bot":
                stop_bot()
                # Force a menu refresh after stopping the bot
                time.sleep(1)  # Give the process time to stop
                continue
            
            elif choice == "View Bot Output":
                view_bot_output()
            
            elif choice == "Analyze Trading Session":
                analyze_session()
                # Add a pause to let user see the analysis results
                input("\nPress Enter to continue...")
            
            elif choice == "Generate Test Session":
                generate_test_session()
                # Add a pause to let user see the generation results
                input("\nPress Enter to continue...")
            
            elif choice == "Exit":
                if check_bot_process():
                    if questionary.confirm("Bot is still running. Stop it before exiting?").ask():
                        stop_bot()
                console.print("[yellow]Goodbye![/yellow]")
                break
                
        except KeyboardInterrupt:
            continue
        except Exception as e:
            console.print(f"[red]Error in menu: {str(e)}[/red]")
            time.sleep(2)
            continue

def load_config() -> dict:
    """Load configuration from config.yaml"""
    try:
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            console.print("[red]Error: config.yaml not found![/red]")
            return None
            
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            
        # Extract trading configuration
        config = {
            'asset': full_config['universal']['asset'],
            'max_position_size': full_config['trading']['max_position_size'],
            'leverage': full_config['trading']['leverage'],
            'profit_target': full_config['trading']['profit_target'],
            'stop_loss': full_config['trading']['stop_loss'],
            'order_size': full_config['trading']['max_position_size'] * 0.2,  # Use 20% of max position size for orders
            'spread': full_config['trading']['min_volatility_threshold'],  # Use min volatility threshold as spread
            'update_interval': full_config['trading']['update_interval']
        }
            
        return config
        
    except Exception as e:
        console.print(f"[red]Error loading config: {str(e)}[/red]")
        return None

def check_required_env_vars() -> bool:
    """Check if all required environment variables are set"""
    required_vars = ["HYPERLIQUID_ACCOUNT_ADDRESS", "HYPERLIQUID_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        console.print("[red]Missing required environment variables:[/red]")
        for var in missing_vars:
            console.print(f"[red]  - {var}[/red]")
        console.print("\n[yellow]Please create a .env.local file with these variables:[/yellow]")
        console.print("HYPERLIQUID_ACCOUNT_ADDRESS=your_account_address")
        console.print("HYPERLIQUID_SECRET_KEY=your_secret_key")
        return False
        
    return True

if __name__ == "__main__":
    main_menu() 