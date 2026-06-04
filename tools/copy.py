import os
import shutil
from pathlib import Path


def copy_images_with_suffix(source_folder, target_folder, suffix="_P"):
    """
    复制源文件夹中的所有图片到目标文件夹，并在文件名后添加后缀

    Args:
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
        suffix: 要添加的后缀
    """
    # 创建目标文件夹（如果不存在）
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    # 遍历源文件夹中的所有文件
    copied_count = 0
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        # 检查是否是文件且是图片格式
        if os.path.isfile(file_path):
            file_ext = Path(filename).suffix.lower()
            if file_ext in image_extensions:
                # 分离文件名和扩展名
                name_without_ext = Path(filename).stem
                new_filename = f"{name_without_ext}{suffix}{file_ext}"
                target_path = os.path.join(target_folder, new_filename)

                # 复制文件
                shutil.copy2(file_path, target_path)
                copied_count += 1

    print(f"\n复制完成！共复制 {copied_count} 个文件")
    return copied_count


# 使用示例
if __name__ == "__main__":
    source_dir = "../results_cl/pranet-results/PraNet"  # 修改为您的源文件夹路径
    target_dir = "../dataset/pranet-traindataset/pranet-testdataset"  # 修改为您的目标文件夹路径
    for x in ['CVC-300','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','Kvasir']:
        source = os.path.join(source_dir,x)
        target = os.path.join(target_dir,'masks')
        copy_images_with_suffix(source, target, x)