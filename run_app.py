#!/usr/bin/env python3
"""
Script to run the Streamlit application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app."""
    try:
        # Check if models exist
        if not os.path.exists('models/linear_regression.pkl'):
            print("Models not found. Training models first...")
            subprocess.run([sys.executable, 'train_models.py'], check=True)
            print("Models trained successfully!")
        
        # Run the Streamlit app
        print("Starting Streamlit app...")
        print("The app will be available at: http://localhost:8501")
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app/main.py',
            '--server.port', '8501'
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running the application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()