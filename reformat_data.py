import json
with open('training_data.json') as f:
    data = json.load(f)

class CompactFramesEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            yield '[\n'
            for i, rep in enumerate(obj):
                yield '  {\n'
                yield f'    "label": "{rep["label"]}",\n'
                yield '    "frames": [\n'
                for j, frame in enumerate(rep['frames']):
                    comma = ',' if j < len(rep['frames']) - 1 else ''
                    yield f'      {json.dumps([round(v, 3) for v in frame])}{comma}\n'
                yield '    ]\n'
                yield '  }' + (',' if i < len(obj) - 1 else '') + '\n'
            yield ']\n'
        else:
            yield from super().iterencode(obj, _one_shot)

# overwrite original
# with open('training_data.json', 'w') as f:
#     f.write(CompactFramesEncoder().encode(data))

#or write to a new file instead
with open('training_data_formatted.json', 'w') as f:
     f.write(CompactFramesEncoder().encode(data))
  
print(f"Done — {len(data)} reps reformatted.")