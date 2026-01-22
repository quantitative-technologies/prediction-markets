import requests
import json
import pandas as pd

# Official Polymarket Subgraph (hosted by Goldsky for free public access)
# See: https://docs.polymarket.com/developers/subgraph/overview
URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/polymarket-mainnet/1.0.0/gn"

def fetch_single_markets():
    # GraphQL query to find markets created between your dates
    # We look for 'fixedProductMarketMakers' (the technical name for these markets)
    # We filter for only 2 outcome tokens (Single/Binary markets)
    query = """
    query {
      fixedProductMarketMakers(
        where: {
          creationTimestamp_gte: "1711929600", 
          creationTimestamp_lte: "1743465600",
          outcomeTokenAmounts_len: 2
        }
        first: 1000
        orderBy: creationTimestamp
        orderDirection: asc
      ) {
        id
        creationTimestamp
        title
        question {
          id
          title
        }
      }
    }
    """
    
    response = requests.post(URL, json={'query': query})
    if response.status_code == 200:
        data = response.json()
        if 'errors' in data:
            print("Error:", data['errors'])
            return []
        return data['data']['fixedProductMarketMakers']
    else:
        raise Exception(f"Query failed: {response.status_code}")

if __name__ == "__main__":
    markets = fetch_single_markets()
    print(f"Found {len(markets)} Single Markets via Subgraph")
    
    # Optional: Save to CSV to inspect
    if markets:
        df = pd.DataFrame(markets)
        print(df.head())
        