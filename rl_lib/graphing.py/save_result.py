import os, shutil, sys

src_dir = str(sys.argv[1])
destination_dir = str(sys.argv[2])

if os.path.exists(src_dir):
    os.makedirs(destination_dir, exist_ok=True)
    shutil.move(src_dir, destination_dir)
else:
    print(f"Source directory {src_dir} does not exist.")