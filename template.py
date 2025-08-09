import os

# Create directories
os.makedirs("src", exist_ok=True)
os.makedirs("research", exist_ok=True)

# Create files inside src
open("src/__init__.py", "a").close()
open("src/helper.py", "a").close()
open("src/prompt.py", "a").close()

# Create other files
open(".env", "a").close()
open("setup.py", "a").close()
open("app.py", "a").close()
open("research/trails.ipynb", "a").close()
open("requirements.txt", "a").close()

print("Directory and files created successfully")

