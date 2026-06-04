import os
import glob


def delete_png_files_recursive(folder_path):
    """
    递归删除文件夹及其子文件夹中所有以 '_P.png' 结尾的图片文件
    """
    deleted_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('_P.png'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)

                    deleted_count += 1
                except Exception as e:
                    print(f"删除文件 '{file_path}' 时出错: {e}")

    if deleted_count == 0:
        print(f"在 '{folder_path}' 及其子文件夹中没有找到以 '_P.png' 结尾的文件")
    else:
        print(f"\n完成！共删除了 {deleted_count} 个文件")


# 使用示例
folder_to_clean = "/home/yassine/projects/UACANet-main/dataset/patches"  # 替换为你的实际路径
delete_png_files_recursive(folder_to_clean)