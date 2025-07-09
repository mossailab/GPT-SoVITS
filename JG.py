import os

def print_tree(root_dir, indent=""):
    items = os.listdir(root_dir)
    items.sort()
    for i, name in enumerate(items):
        path = os.path.join(root_dir, name)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(indent + connector + name)
        if os.path.isdir(path):
            next_indent = indent + ("    " if i == len(items) - 1 else "│   ")
            print_tree(path, next_indent)

if __name__ == "__main__":
    root_path = r"E:\Python\GPT-SoVITS-main"  # ← 你可以改成任何目录
    print(root_path + "\\")
    print_tree(root_path)

with open("tree.txt", "w", encoding="utf-8") as f:
    f.write(root_path + "\\\n")
    def print_tree_to_file(dir_path, indent=""):
        items = os.listdir(dir_path)
        items.sort()
        for i, name in enumerate(items):
            path = os.path.join(dir_path, name)
            connector = "└── " if i == len(items) - 1 else "├── "
            f.write(indent + connector + name + "\n")
            if os.path.isdir(path):
                next_indent = indent + ("    " if i == len(items) - 1 else "│   ")
                print_tree_to_file(path, next_indent)
    print_tree_to_file(root_path)
