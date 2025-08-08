# main.py

from src import preprocessing, train, evaluate, visualize, config

def main():
    """Runs the main project pipeline."""
    
    # 1. Preprocess the data
    print("--- 1. Starting Data Preprocessing ---")
    data = preprocessing.prepare_data()
    
    # 2. Train the model
    print("\n--- 2. Starting Model Training ---")
    model = train.train_model(data['X_train'], data['y_train'])
    
    # 3. Save the model and scalers for future use
    train.save_model_and_scalers(model, data['scaler_X'], data['scaler_y'])
    
    # 4. Evaluate the model on the test set
    predictions_scaled = evaluate.get_predictions(model, data['X_test'])
    eval_results = evaluate.calculate_metrics(
        data['y_test'], 
        predictions_scaled, 
        data['scaler_y'],
        num_features=len(data['feature_columns'])
    )
    
    # 5. Visualize the results
    print("\n--- 5. Generating Visualizations ---")
    visualize.plot_actual_vs_predicted(
        eval_results['y_true_original'], 
        eval_results['y_pred_original']
    )
    
    visualize.plot_residuals(data['y_test'], predictions_scaled)
    
    visualize.plot_feature_importance(model, data['feature_columns'])
    
    print("\n--- Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main()