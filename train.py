import argparse
import os
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from utils.data import fetch_data, build_features, train_test_split_time, compute_metrics

def main():
    parser = argparse.ArgumentParser(description="Train and save stock price prediction model.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol, e.g., AAPL or INFY.NS")
    parser.add_argument("--period", type=str, default="8y", help="yfinance period, e.g., 5y, 8y, max")
    parser.add_argument("--interval", type=str, default="1d", help="yfinance interval, e.g., 1d, 1h")
    parser.add_argument("--lookback_lags", type=int, default=5, help="Number of lag features")
    parser.add_argument("--model", type=str, default="rf", choices=["rf"], help="Model type")
    parser.add_argument("--n_estimators", type=int, default=400, help="RF trees")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    args = parser.parse_args()

    print(f"Fetching {args.ticker} ...")
    df = fetch_data(args.ticker, period=args.period, interval=args.interval)
    X, y, prev_close, feature_names = build_features(df, lookback_lags=args.lookback_lags)
    X_train, X_test, y_train, y_test, prev_train, prev_test = train_test_split_time(
        X, y, prev_close, test_size=args.test_size
    )

    print("Training RandomForest...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, prev_test)
    print("Metrics:", metrics)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{args.ticker}_rf.joblib")
    meta_path = os.path.join("models", f"{args.ticker}_meta.json")

    import json
    joblib.dump({"model": model, "feature_names": feature_names, "lookback_lags": args.lookback_lags}, model_path)
    with open(meta_path, "w") as f:
        json.dump({"ticker": args.ticker, "period": args.period, "interval": args.interval, "metrics": metrics}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()