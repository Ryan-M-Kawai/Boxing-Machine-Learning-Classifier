import json
from collections import Counter

with open('training_data.json') as f:
    data = json.load(f)

labels = [d['label'] for d in data]
counts = Counter(labels)

print(f"Total reps    : {len(data)}")
print(f"Feature size  : {len(data[0]['frames'][0])}")
print(f"Frame window  : {len(data[0]['frames'])}")
print(f"\nReps per class:")
for label, count in sorted(counts.items()):
    print(f"  {label}: {count}")