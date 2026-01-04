import musdb
import os

def check(root):
    print(f"Checking root: {root}")
    try:
        db = musdb.DB(root=root, subsets=['train'])
        print(f"Found {len(db.tracks)} tracks.")
    except Exception as e:
        print(f"Error: {e}")

# Check 1: Explicit path
check(r"C:\Users\vamsi\MUSDB18\MUSDB18-7")

# Check 2: Parent
check(r"C:\Users\vamsi\MUSDB18")

# Check 3: None (Env var)
check(None)

# Check 4: Download=True with None
print("Checking with download=True")
db = musdb.DB(download=True)
print(f"Found {len(db.tracks)} tracks in {db.root}")
