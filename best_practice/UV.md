### to add library
- `uv add ___`
- (X; this will not update pyproject) `uv pip install ___` 

### to check dependency issues
- `uv pip install deptry`
- `uv run deptry .`

### form requirement.txt
- `uv pip compile pyproject.toml -o requirements.txt`