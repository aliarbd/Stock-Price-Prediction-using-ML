import uuid
import shutil
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from webapp.db import init_db, get_db_connection, SAVED_MODELS_DIR
from webapp.model_registry import (
    save_model_record,
    record_prediction,
    get_model_predictions,
    update_actual_values,
)
from webapp.prediction_service import (
    get_post_save_predictions_analysis,
    get_post_save_strategy_analysis,
    get_current_signal_analysis,
)

def run_test():
    init_db()
    
    test_model_id = "test_model_" + uuid.uuid4().hex[:8]
    live_boundary = "2026-08-09"

    # Create dummy artifact directory
    model_dir = SAVED_MODELS_DIR / test_model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_model = LinearRegression()
    dummy_model.fit([[1.0]], [1.0])
    joblib.dump(dummy_model, model_dir / "model.joblib")

    dummy_scaler = MinMaxScaler()
    dummy_scaler.fit([[1.0], [2.0]])
    joblib.dump(dummy_scaler, model_dir / "scaler.joblib")
    
    # 1. Save dummy model record with live boundary = 2026-08-09
    model_data = {
        "id": test_model_id,
        "name": "TEST_AAPL_LinearRegression_V1",
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "timeframe": "1D",
        "model_type": "Linear Regression",
        "model_path": str(model_dir / "model.joblib"),
        "scaler_path": str(model_dir / "scaler.joblib"),
        "metadata_path": str(model_dir / "metadata.json"),
        "feature_config": ["close"],
        "timestep": 60,
        "train_start": "2024-01-01",
        "train_end": live_boundary,
        "train_split": 0.8,
        "created_at": "2026-08-09T20:00:00",
        "status": "Active",
        "is_active": 1,
        "auto_update": 0,
        "version": 1,
    }
    save_model_record(model_data)
    print("Step 1: Model saved with train_end =", live_boundary)

    # Prepare dummy market dataframe (df)
    dates = pd.date_range("2026-08-01", "2026-08-12", freq="B")
    close_prices = [150.0 + i for i in range(len(dates))]
    df = pd.DataFrame({"close": close_prices}, index=dates)

    # 2. Immediately after save: check post-save predictions analysis
    res1 = get_post_save_predictions_analysis(test_model_id, df)
    print("Step 2 (Immediate post-save):")
    print("  has_data:", res1["has_data"])
    print("  message:", res1.get("message"))
    print("  post_save_samples:", res1.get("post_save_samples"))
    assert res1["has_data"] is False, "Expected has_data=False immediately after save"
    assert res1["post_save_samples"] == 0, "Expected 0 samples immediately after save"

    # 3. Check post-save strategy analysis immediately after save
    strat1 = get_post_save_strategy_analysis(test_model_id, df)
    print("Step 3 (Immediate strategy):")
    print("  has_data:", strat1["has_data"])
    print("  message:", strat1.get("message"))
    assert strat1["has_data"] is False, "Expected strat has_data=False immediately after save"

    # 4. Check current signal immediately after save
    sig1 = get_current_signal_analysis(test_model_id, df)
    print("Step 4 (Immediate current signal):")
    print("  signal:", sig1["signal"])
    print("  position_status:", sig1["position_status"])
    print("  rationale:", sig1["rationale"])
    assert sig1["signal"] == "HOLD", "Expected signal=HOLD when no live records exist"

    # 5. Simulate Day 1 live prediction: Target 2026-08-10 (generated on 2026-08-09)
    df_aug9 = df.loc[:"2026-08-09"].copy()
    pred_id1 = record_prediction(
        model_id=test_model_id,
        symbol="AAPL",
        input_data_timestamp="2026-08-09",
        prediction_date="2026-08-10",
        predicted_value=157.5,
        prediction_horizon=1,
        signal="BUY",
    )
    print("\nStep 5: Generated prediction for 2026-08-10 (Target date)")
    
    # Check duplicate prevention: attempt inserting prediction for same (test_model_id, 2026-08-10)
    pred_id1_dup = record_prediction(
        model_id=test_model_id,
        symbol="AAPL",
        input_data_timestamp="2026-08-09",
        prediction_date="2026-08-10",
        predicted_value=157.5,
        prediction_horizon=1,
        signal="BUY",
    )
    assert pred_id1 == pred_id1_dup, "Duplicate prevention failed! Expected same prediction ID."
    print("  Duplicate prevention verified: prediction IDs match (ID:", pred_id1, ")")

    # Before actual 2026-08-10 close is evaluated
    res_pending = get_post_save_predictions_analysis(test_model_id, df_aug9)
    print("  Before 2026-08-10 close available -> has_data:", res_pending["has_data"])

    # 6. Now 2026-08-10 close becomes available (156.0)
    df_aug10 = df.loc[:"2026-08-10"].copy()
    update_actual_values(test_model_id, df_aug10)
    res_eval1 = get_post_save_predictions_analysis(test_model_id, df_aug10)
    print("Step 6: After 2026-08-10 close evaluated:")
    print("  has_data:", res_eval1["has_data"])
    print("  post_save_samples:", res_eval1["post_save_samples"])
    print("  metrics:", res_eval1["metrics"])
    assert res_eval1["has_data"] is True, "Expected has_data=True after 1 evaluated point"
    assert res_eval1["post_save_samples"] == 1, "Expected 1 post_save_sample"

    # 7. Record Day 2 prediction: Target 2026-08-11 (generated on 2026-08-10)
    record_prediction(
        model_id=test_model_id,
        symbol="AAPL",
        input_data_timestamp="2026-08-10",
        prediction_date="2026-08-11",
        predicted_value=159.0,
        prediction_horizon=1,
        signal="BUY",
    )
    df_aug11 = df.loc[:"2026-08-11"].copy()
    update_actual_values(test_model_id, df_aug11)

    res_eval2 = get_post_save_predictions_analysis(test_model_id, df_aug11)
    strat_eval2 = get_post_save_strategy_analysis(test_model_id, df_aug11)

    print("\nStep 7: After 2026-08-11 close evaluated:")
    print("  Graph #1 samples:", res_eval2["post_save_samples"])
    print("  Graph #2 has_data:", strat_eval2["has_data"])
    print("  Graph #2 metrics:", strat_eval2.get("metrics"))

    assert res_eval2["post_save_samples"] == 2, "Expected 2 post_save_samples"
    assert strat_eval2["has_data"] is True, "Expected Graph #2 has_data=True with 2 evaluated predictions"

    # Cleanup test model & directory
    with get_db_connection() as conn:
        conn.execute("DELETE FROM saved_models WHERE id = ?", (test_model_id,))
        conn.commit()
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
    print("\nALL VERIFICATION TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    run_test()
