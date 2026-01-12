#!/usr/bin/env python3
"""
Quick setup verification script for ArXiv Trends Analysis.
Checks if all dependencies and configurations are correct.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed."""
    print("Checking Python dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'kagglehub', 'umap',
        'nltk', 'matplotlib', 'seaborn', 'wordcloud', 'yaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    print("\nChecking Kaggle credentials...")
    
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if kaggle_json.exists():
        print(f"  ✓ Found kaggle.json at {kaggle_json}")
        
        # Check permissions (should be 600)
        import stat
        mode = kaggle_json.stat().st_mode
        if mode & stat.S_IRWXG == 0 and mode & stat.S_IRWXO == 0:
            print("  ✓ Permissions are correct (600)")
        else:
            print("  ⚠ Permissions should be 600")
            print(f"    Run: chmod 600 {kaggle_json}")
        return True
    else:
        print(f"  ✗ Kaggle credentials not found at {kaggle_json}")
        print("\nTo set up Kaggle credentials:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print(f"  3. Place kaggle.json in {kaggle_json.parent}")
        print(f"  4. Run: chmod 600 {kaggle_json}")
        return False


def check_project_structure():
    """Check if project directories exist."""
    print("\nChecking project structure...")
    
    required_dirs = [
        'src',
        'src/config',
        'src/utils',
        'data',
        'output',
        'output/plots',
        'output/data',
        'output/reports'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/")
            all_exist = False
    
    if all_exist:
        print("\n✓ Project structure is correct")
    else:
        print("\n⚠ Some directories missing (will be auto-created)")
    
    return all_exist


def check_config_file():
    """Check if configuration file exists."""
    print("\nChecking configuration...")
    
    config_path = Path('src/config/config.yaml')
    if config_path.exists():
        print(f"  ✓ Configuration file found: {config_path}")
        return True
    else:
        print(f"  ✗ Configuration file not found: {config_path}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("ArXiv Trends Analysis - Setup Verification")
    print("=" * 60)
    
    checks = [
        check_dependencies(),
        check_project_structure(),
        check_config_file(),
        check_kaggle_credentials()
    ]
    
    print("\n" + "=" * 60)
    if all(checks[:3]):  # First 3 checks are critical
        print("✓ Setup is complete!")
        if not checks[3]:
            print("\n⚠ Note: Kaggle credentials not configured.")
            print("  The pipeline will attempt to download data automatically.")
            print("  If download fails, set up credentials as shown above.")
        print("\nYou can now run the analysis:")
        print("  python src/main.py")
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
