import os
import shutil

ROOT_DIR = "dataset"
OUTPUT_DIR = "files"

TSV_EXT = ".tsv"
TXT_EXT = ".txt"

def safe_copy(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dst = os.path.join(dst_dir, base)

    counter = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.copy2(src, dst)


def extract_files(base_dir, extension, out_dir):
    for root, dirs, files in os.walk(base_dir):

        dirs[:] = [d for d in dirs if d.lower() != "logs"]

        for file in files:
            if file.lower().endswith(extension):
                src_path = os.path.join(root, file)
                safe_copy(src_path, out_dir)


def main():
    paths = {
        "CNV": TSV_EXT,
        "RNA": TSV_EXT,
        "methylation": TXT_EXT
    }

    for folder, ext in paths.items():
        base_path = os.path.join(ROOT_DIR, folder)
        out_path = os.path.join(OUTPUT_DIR, folder)

        if os.path.isdir(base_path):
            extract_files(base_path, ext, out_path)
        else:
            print(f"Directory not found: {base_path}")

    print("Completed")


if __name__ == "__main__":
    main()