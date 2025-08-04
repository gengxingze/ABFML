import os

def print_tree(root_dir, file=None, prefix=''):
    items = sorted(os.listdir(root_dir))
    for i, name in enumerate(items):
        path = os.path.join(root_dir, name)
        is_last = (i == len(items) - 1)
        branch = '└── ' if is_last else '├── '
        line = f"{prefix}{branch}{name}"
        print(line)
        if file:
            file.write(line + '\n')
        if os.path.isdir(path):
            extension = '    ' if is_last else '│   '
            print_tree(path, file=file, prefix=prefix + extension)

if __name__ == "__main__":
    root = r"D:\Work\PyCharm\ABFML\abfml"  # 可改为你的项目路径，如 "./my_project"
    output_file = "tree.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        print_tree(root, file=f)

    print(f"\n目录结构已保存至 {output_file}")
