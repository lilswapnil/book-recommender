import json

# Read the file
with open('notebook/sentiment_analysis.ipynb', 'r') as f:
    lines = f.readlines()

# Process lines to remove merge conflict markers
output = []
i = 0
while i < len(lines):
    if lines[i].strip().startswith('<<<<<<< HEAD'):
        # Found start of conflict, keep HEAD version
        i += 1
        conflict_head = []
        while i < len(lines) and not lines[i].strip().startswith('======='):
            conflict_head.append(lines[i])
            i += 1
        
        # Skip the separator and the other version
        i += 1  # skip =======
        while i < len(lines) and not lines[i].strip().startswith('>>>>>'):
            i += 1
        i += 1  # skip the closing >>>>>
        
        # Add the HEAD version
        output.extend(conflict_head)
    else:
        output.append(lines[i])
        i += 1

# Write back
with open('notebook/sentiment_analysis.ipynb', 'w') as f:
    f.writelines(output)

# Validate JSON
try:
    with open('notebook/sentiment_analysis.ipynb', 'r') as f:
        json.load(f)
    print("✓ Merge conflicts resolved and JSON is valid")
except json.JSONDecodeError as e:
    print(f"✗ JSON validation failed: {e}")
    import traceback
    traceback.print_exc()
