from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from config import (
    TRANSPORT_COST_MIN, TRANSPORT_COST_MAX,
    MIN_PROFIT_PERCENT, MIN_PROFIT_DOLLARS,
    PURCHASE_PRICE_MIN_PCT, PURCHASE_PRICE_MAX_PCT
)

def load_and_preprocess_data():
    # Fetching dataset
    dataset = fetch_ucirepo(id=352)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # Basic cleaning
    df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate', 'Quantity', 'UnitPrice'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df.rename(columns={'UnitPrice': 'selling_price', 'InvoiceDate': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Temporal features
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    # Simulated fields
    df['transportation_cost'] = np.random.uniform(TRANSPORT_COST_MIN, TRANSPORT_COST_MAX, len(df))
    df['purchase_price'] = df['selling_price'] * np.random.uniform(PURCHASE_PRICE_MIN_PCT, PURCHASE_PRICE_MAX_PCT, len(df))
    base_cost = df['purchase_price'] + df['transportation_cost']
    df['min_profit'] = np.maximum(base_cost * MIN_PROFIT_PERCENT, MIN_PROFIT_DOLLARS)
    df['target_price'] = base_cost + df['min_profit']

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    df.to_csv("data/preprocessed_online_retail.csv", index=False)
    print("Data preprocessing completed and saved to data/preprocessed_online_retail.csv")
