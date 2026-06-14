"""
Check if ML model is loaded and using Elo features
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from NHL.AppState import get_app_state

def main():
    print("=" * 60)
    print("ML MODEL DIAGNOSTIC")
    print("=" * 60)
    
    try:
        state = get_app_state()
        ml = state['ml_model']
        
        print(f"\n✓ ML Model Loaded: {ml is not None}")
        print(f"✓ Model Trained: {ml.is_trained}")
        print(f"✓ Model ID: {ml.model_id}")
        print(f"✓ Current Season: {state.get('current_season', 'unknown')}")
        
        # Check performance
        perf = ml.get_recent_performance(n=100)
        if perf and perf.get('n_predictions', 0) > 0:
            print(f"\n📊 Recent Performance (last {perf['n_predictions']} predictions):")
            print(f"   Accuracy: {perf['accuracy']:.1%}")
            print(f"   Brier Score: {perf['avg_brier']:.4f}")
            print(f"   Log Loss: {perf['avg_logloss']:.4f}")
        else:
            print("\n⚠️  No performance data yet (model hasn't made predictions)")
        
        # Check feature importance
        importance = ml.get_feature_importance()
        if importance:
            print(f"\n🎯 Top 10 Most Important Features:")
            sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feat, imp) in enumerate(sorted_feats, 1):
                indicator = "🔥" if "elo" in feat.lower() else "  "
                print(f"   {i:2d}. {indicator} {feat:30s} {imp:.4f}")
            
            # Check if Elo features are being used
            elo_features = [f for f, _ in sorted_feats if "elo" in f.lower()]
            if len(elo_features) >= 3:
                print(f"\n✅ Good: {len(elo_features)} Elo features in top 10")
            else:
                print(f"\n⚠️  Warning: Only {len(elo_features)} Elo features in top 10")
                print("   Model may not be using Elo effectively")
        else:
            print("\n⚠️  No feature importance data available")
        
        # Check config
        print(f"\n⚙️  Model Config:")
        print(f"   Elo Feature Weight: {ml.config.elo_feature_weight}")
        print(f"   Model Type: {ml.config.model_type}")
        print(f"   Learning Rate: {ml.config.learning_rate}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())