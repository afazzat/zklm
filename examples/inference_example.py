"""
Example of using a trained medical diagnosis model for inference.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import argparse

from zkml.models import MedicalDiagnosisModel

def parse_args():
    parser = argparse.ArgumentParser(description="Medical diagnosis model inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model file')
    parser.add_argument('--scaler_path', type=str, required=True, help='Path to saved scaler file')
    parser.add_argument('--input_dim', type=int, default=33, help='Input dimension (feature count)')
    return parser.parse_args()

def generate_test_patient(age=None, blood_pressure=None, glucose=None, bmi=None, cholesterol=None):
    """Generate a test patient with specified or random values."""
    # Default values if not provided
    age = age if age is not None else np.random.normal(50, 15)
    blood_pressure = blood_pressure if blood_pressure is not None else np.random.normal(120, 20)
    glucose = glucose if glucose is not None else np.random.normal(100, 25)
    bmi = bmi if bmi is not None else np.random.normal(25, 5)
    cholesterol = cholesterol if cholesterol is not None else np.random.normal(200, 40)
    heart_rate = np.random.normal(75, 12)
    
    # Generate additional clinical indicators
    crp = np.random.gamma(2, 3)  # C-reactive protein (inflammation)
    wbc = np.random.normal(7.5, 2)  # White blood cell count
    temp = np.random.normal(37, 0.7)  # Body temperature
    
    # Random features for remaining dimensions
    random_features = np.random.randn(24)
    
    # Create a feature vector
    features = np.concatenate([
        [age, blood_pressure, glucose, bmi, cholesterol, heart_rate, crp, wbc, temp],
        random_features
    ])
    
    return features, {
        'Age': age,
        'Blood Pressure': blood_pressure,
        'Glucose': glucose,
        'BMI': bmi,
        'Cholesterol': cholesterol,
        'Heart Rate': heart_rate,
        'CRP': crp,
        'WBC': wbc,
        'Temperature': temp
    }

def predict(model, scaler, features):
    """Make a prediction using the model."""
    # Preprocess features
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, confidence, probabilities.numpy()

def main():
    args = parse_args()
    
    # Load model and scaler
    print(f"Loading model from {args.model_path}")
    model = MedicalDiagnosisModel(input_dim=args.input_dim, hidden_dim=1024)
    model.load_state_dict(torch.load(args.model_path))
    
    print(f"Loading scaler from {args.scaler_path}")
    scaler = torch.load(args.scaler_path)
    
    # Generate some test patients
    print("\nGenerating test patients...")
    
    # 1. Healthy patient
    features_healthy, info_healthy = generate_test_patient(
        age=35, 
        blood_pressure=110, 
        glucose=85, 
        bmi=22, 
        cholesterol=170
    )
    
    # 2. High-risk patient
    features_high_risk, info_high_risk = generate_test_patient(
        age=65, 
        blood_pressure=150, 
        glucose=140, 
        bmi=32, 
        cholesterol=240
    )
    
    # 3. Random patient
    features_random, info_random = generate_test_patient()
    
    # Make predictions
    patients = [
        ("Healthy Patient", features_healthy, info_healthy),
        ("High-Risk Patient", features_high_risk, info_high_risk),
        ("Random Patient", features_random, info_random)
    ]
    
    print("\nPredictions:")
    print("-" * 80)
    
    for name, features, info in patients:
        prediction, confidence, probs = predict(model, scaler, features)
        disease_status = "Positive" if prediction == 1 else "Negative"
        
        print(f"\n{name}:")
        print(f"  Key vitals: Age={info['Age']:.1f}, BP={info['Blood Pressure']:.1f}, "
              f"Glucose={info['Glucose']:.1f}, BMI={info['BMI']:.1f}, "
              f"Cholesterol={info['Cholesterol']:.1f}")
        print(f"  Prediction: {disease_status} (Confidence: {confidence:.2%})")
        print(f"  Class probabilities: Negative={probs[0,0]:.4f}, Positive={probs[0,1]:.4f}")
    
    print("\nInteractive mode - predict with custom values:")
    while True:
        try:
            age = float(input("Age (or press Enter for random): ") or "random")
            age = age if age != "random" else None
            
            bp = float(input("Blood Pressure (or press Enter for random): ") or "random")
            bp = bp if bp != "random" else None
            
            glucose = float(input("Glucose (or press Enter for random): ") or "random")
            glucose = glucose if glucose != "random" else None
            
            bmi = float(input("BMI (or press Enter for random): ") or "random")
            bmi = bmi if bmi != "random" else None
            
            cholesterol = float(input("Cholesterol (or press Enter for random): ") or "random")
            cholesterol = cholesterol if cholesterol != "random" else None
            
            # Generate patient with these values
            features, info = generate_test_patient(age, bp, glucose, bmi, cholesterol)
            prediction, confidence, probs = predict(model, scaler, features)
            disease_status = "Positive" if prediction == 1 else "Negative"
            
            print("\nCustom Patient:")
            print(f"  Key vitals: Age={info['Age']:.1f}, BP={info['Blood Pressure']:.1f}, "
                  f"Glucose={info['Glucose']:.1f}, BMI={info['BMI']:.1f}, "
                  f"Cholesterol={info['Cholesterol']:.1f}")
            print(f"  Prediction: {disease_status} (Confidence: {confidence:.2%})")
            print(f"  Class probabilities: Negative={probs[0,0]:.4f}, Positive={probs[0,1]:.4f}")
            
            another = input("\nTry another? (y/n): ")
            if another.lower() != 'y':
                break
                
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main() 