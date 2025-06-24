import sys
import os
import subprocess

def check_dependencies():
    """Ensure required packages are installed"""
    print("Checking dependencies...")
    required = ['pandas', 'numpy', 'matplotlib', 'seaborn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Attempting to install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("Installation completed. Please restart the script.")
        sys.exit(1)

def main():
    """Main EDA execution function"""
    print("Starting EDA visualization generation...")
    
    # Check environment
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Ensure dependencies are installed
    check_dependencies()
    
    # Now import visualization
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    
    from visualization import generate_eda
    
    # Generate visualizations
    generate_eda()
    print("EDA process completed!")

if __name__ == '__main__':
    main()