import json
import requests
import time
import sys
import argparse
from datetime import datetime

# Official Polymarket Metadata API
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"

def get_total_closed_markets():
    """Fetches the total number of closed markets to initialize the progress bar."""
    try:
        resp = requests.get(f"{GAMMA_API_URL}/count", params={"closed": "true"}, timeout=10)
        if resp.ok:
            return resp.json()
        return None
    except:
        return None

def draw_progress_bar(current, total, found_count, start_time):
    """Draws a text-based progress bar in the terminal."""
    bar_length = 30
    # Avoid division by zero
    if total == 0:
        total = 1
        
    percent = min(1.0, float(current) / total)
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    # Calculate ETA
    elapsed = time.time() - start_time
    if percent > 0:
        rate = current / elapsed  # items per second
        remaining_items = total - current
        eta_seconds = remaining_items / rate
        eta_str = f"{int(eta_seconds)}s"
    else:
        eta_str = "..."

    sys.stdout.write(f"\r[{arrow + spaces}] {int(percent * 100)}% | Scanned: {current}/{total} | Found: {found_count} | ETA: {eta_str}")
    sys.stdout.flush()

def count_single_markets(start_date, end_date, limit=500):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 1. Get Total Count for Progress Bar
    print("Initializing... fetching total market count...")
    total_markets = get_total_closed_markets()
    
    if not total_markets:
        print("Could not fetch total count. Running without progress bar.")
        total_markets = 60000 # Fallback estimate
    
    print(f"Total Closed Markets to Scan: {total_markets}")
    print(f"Filtering for Single Markets between {start_date} and {end_date}")
    print(f"Batch Size: {limit}\n")
    
    offset = 0
    total_single_markets = 0
    start_time = time.time()
    
    while True:
        try:
            # Fetch batch
            params = {
                "limit": limit,
                "offset": offset,
                "closed": "true" 
            }
            
            resp = requests.get(GAMMA_API_URL, params=params, timeout=10)
            if not resp.ok:
                print(f"\nAPI Error {resp.status_code}: {resp.text}")
                # If 500 fails, retry once with 100 just in case
                if limit > 100:
                    print("Retrying with limit=100...")
                    limit = 100
                    continue
                break
                
            markets = resp.json()
            if not markets:
                break # End of list
            
            # Process Batch
            for m in markets:
                # 1. Check Date
                end_date_str = m.get('endDate')
                if not end_date_str:
                    continue
                    
                try:
                    # Parse "2024-05-01T12:00:00Z"
                    market_end = datetime.strptime(end_date_str[:10], "%Y-%m-%d")
                except:
                    continue
                
                # 2. Check Range & Type
                if start_dt <= market_end <= end_dt:
                    # Filter for Single/Binary Markets (2 outcomes)
                    outcomes_str = m['outcomes']
                    outcomes = json.loads(outcomes_str)
                    assert len(outcomes) == 2, f"Outcomes are {outcomes}"
                    # Also check question text to avoid "Group" markets if necessary
                    if len(outcomes) == 2:
                        total_single_markets += 1

            # Update Progress
            offset += len(markets)
            draw_progress_bar(min(offset, total_markets), total_markets, total_single_markets, start_time)
            
            # Rate limiting (Gamma is sensitive)
            time.sleep(0.05)
            
            if len(markets) < limit:
                break
            
        except KeyboardInterrupt:
            print("\nScan interrupted by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break
            
    return total_single_markets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count Single Markets on Polymarket")
    parser.add_argument("--start", type=str, default="2024-04-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-04-01", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=500, help="API Batch Limit (Default: 500)")
    
    args = parser.parse_args()
    
    count = count_single_markets(args.start, args.end, args.limit)
    print(f"\n\n{'='*40}")
    print(f"FINAL COUNT")
    print(f"{'='*40}")
    print(f"Single Markets Resolved: {count}")
