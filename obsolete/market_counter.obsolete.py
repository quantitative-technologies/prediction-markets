import os
import requests
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.environ.get("ALCHEMY_API_KEY")
if not API_KEY:
    raise ValueError("ALCHEMY_API_KEY not found in .env")

# Use the key exactly as provided, just stripping whitespace
API_KEY = API_KEY.strip().split('/')[-1]

RPC_URL = f"https://polygon-mainnet.g.alchemy.com/v2/{API_KEY}"
CTF_EXCHANGE_ADDR = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
TOKEN_REGISTERED_TOPIC = "0x3c672951e70e300994f275e74c867201c10d7a049171e2175949d211029c1352"

# Dates
START_DATE = "2024-04-01"
END_DATE = "2025-04-01"

def make_rpc_request(method, params, retries=3):
    """
    Makes an RPC request. 
    Returns (result, error_message, status_code).
    """
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }
    
    for attempt in range(retries):
        try:
            resp = requests.post(RPC_URL, json=payload, timeout=15)
            
            if resp.status_code == 429: # Rate limit
                time.sleep(2 * (attempt + 1))
                continue
            
            # If 400, return the error body so we can handle "Too Wide" logic
            if not resp.ok:
                return None, resp.text, resp.status_code

            data = resp.json()
            if "error" in data:
                return None, data['error'].get('message', 'Unknown RPC Error'), 200
                
            return data.get('result'), None, 200
            
        except requests.exceptions.RequestException as e:
            print(f"  Network error: {e}")
            time.sleep(1)
            
    return None, "Max Retries Exceeded", 0

def get_block_for_timestamp(target_ts):
    # Get current block to set high bound
    current_block_hex, err, status = make_rpc_request("eth_blockNumber", [])
    if err:
        raise ValueError(f"Could not init: {err} (Status {status})")
    
    current_block = int(current_block_hex, 16)
    low, high = 0, current_block
    
    print(f"Finding block for timestamp {target_ts}...")
    
    while low <= high:
        mid = (low + high) // 2
        block_data, err, _ = make_rpc_request("eth_getBlockByNumber", [hex(mid), False])
        
        if not block_data:
            low = mid + 1 
            continue
            
        mid_ts = int(block_data['timestamp'], 16)
        
        if abs(mid_ts - target_ts) < 300: # 5 min precision
            return mid
            
        if mid_ts < target_ts:
            low = mid + 1
        else:
            high = mid - 1
            
    return mid

def fetch_markets_in_range(start_date_str, end_date_str):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    try:
        start_block = get_block_for_timestamp(int(start_dt.timestamp()))
        end_block = get_block_for_timestamp(int(end_dt.timestamp()))
    except ValueError as e:
        print(f"Fatal Error during setup: {e}")
        return set()

    single_conditions = set()
    
    # Alchemy Free Tier only allows 10 blocks per eth_getLogs request (range must be ≤9)
    chunk_size = 9

    print(f"\n--- CONFIGURATION ---")
    print(f"RPC Endpoint: {RPC_URL[:50]}...")
    print(f"Block Range: {start_block} to {end_block}")
    print(f"Total Blocks: {end_block - start_block}")
    print(f"Chunk Size: {chunk_size} blocks/request")
    
    print(f"\nScanning blockchain for Single Markets...")
    
    current = start_block
    while current < end_block:
        end = min(current + chunk_size, end_block)
        
        params = [{
            "address": CTF_EXCHANGE_ADDR,
            "topics": [TOKEN_REGISTERED_TOPIC],
            "fromBlock": hex(current),
            "toBlock": hex(end)
        }]
        
        logs, err, status = make_rpc_request("eth_getLogs", params)
        
        # ERROR HANDLING STRATEGY
        if err or status != 200:
            err_lower = str(err).lower()
            
            # Debug: Print actual error on first occurrence
            if not hasattr(fetch_markets_in_range, '_err_printed'):
                print(f"\n  [DEBUG] First error response (status {status}):\n  {str(err)[:500]}\n")
                fetch_markets_in_range._err_printed = True
            
            # Check if it's a "too many results/blocks" error
            is_size_error = (
                "limit" in err_lower or 
                "exceeded" in err_lower or
                "too many" in err_lower or
                "range" in err_lower or
                "block" in err_lower
            )
            
            if is_size_error:
                new_size = chunk_size // 2
                if new_size < 1:
                    print(f"\n  [!] Block {current} fails even alone. Skipping.")
                    current += 1
                    chunk_size = 9
                    continue
                    
                print(f"  [!] Reducing chunk: {chunk_size} -> {new_size}")
                chunk_size = new_size
                continue
            
            # Not a size error - something else is wrong
            print(f"\n  [!] Error at block {current} (status {status}): {str(err)[:200]}")
            current += chunk_size
            continue

        # SUCCESS
        if logs:
            for log in logs:
                data_hex = log['data'][2:]
                if len(data_hex) >= 192:
                    condition_id = "0x" + data_hex[128:192]
                    single_conditions.add(condition_id)
        
        # Progress
        progress = (current - start_block) / (end_block - start_block) * 100
        print(f"  {progress:.1f}% | Block {current} | Found {len(single_conditions)} markets", end="\r")
        
        # Keep chunk_size at 9 (max for Alchemy free tier: toBlock - fromBlock ≤ 9)
        chunk_size = 9
            
        current = end + 1

    return single_conditions

if __name__ == "__main__":
    markets = fetch_markets_in_range(START_DATE, END_DATE)
    print(f"\n\nDone. Total Single Markets: {len(markets)}")
    
    with open("single_markets_list.txt", "w") as f:
        for m in markets:
            f.write(f"{m}\n")
    print("Saved ID list to single_markets_list.txt")
    