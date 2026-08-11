import os
import shutil
import zipfile
import random
from pathlib import Path

def get_class_folders(source_path):
    """Recursively unwrap folders to find the actual food item class directories."""
    src = Path(source_path)
    if not src.exists():
        return []
    
    subdirs = [d for d in src.iterdir() if d.is_dir()]
    split_names = {'train', 'val', 'test', 'validation'}
    
    # Case 1: Contains pre-existing split folders (train, val, test)
    if any(d.name.lower() in split_names for d in subdirs):
        class_dirs = []
        for split_dir in subdirs:
            if split_dir.name.lower() in split_names:
                class_dirs.extend([d for d in split_dir.iterdir() if d.is_dir()])
        return class_dirs
    
    # Case 2: Wrapped in a single parent folder (e.g., 'indian_food_images')
    if len(subdirs) == 1 and not any(src.glob("*.*")):
        inner_subdirs = [d for d in subdirs[0].iterdir() if d.is_dir()]
        # Check if the inner folder contains split folders
        if any(d.name.lower() in split_names for d in inner_subdirs):
            class_dirs = []
            for split_dir in inner_subdirs:
                if split_dir.name.lower() in split_names:
                    class_dirs.extend([d for d in split_dir.iterdir() if d.is_dir()])
            return class_dirs
        return inner_subdirs
        
    # Case 3: Subdirectories are directly the food item classes
    return subdirs

def split_and_organize_datasets(indian_food_path, fruit_veg_zip_path, output_dir, train_ratio=0.7, val_ratio=0.15):
    output_path = Path(output_dir)
    temp_extract_path = output_path / "temp_extracted"
    
    # 1. Extract the fruit and vegetable dataset zip file
    if Path(fruit_veg_zip_path).exists():
        print("Extracting fruit and vegetable dataset zip...")
        with zipfile.ZipFile(fruit_veg_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
    
    # Gather all individual food class directories from both sources safely
    class_folders = []
    class_folders.extend(get_class_folders(indian_food_path))
    if temp_extract_path.exists():
        class_folders.extend(get_class_folders(temp_extract_path))

    # Create train, val, and test split directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Collect images grouped by normalized food class name
    class_images = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    for folder in class_folders:
        if not folder.is_dir():
            continue
        class_name = folder.name.lower().strip().replace(' ', '_')
        images = [p for p in folder.glob("**/*.*") if p.suffix.lower() in valid_extensions]
        
        if images:
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].extend(images)

    print(f"Found {len(class_images)} total unique food/item classes.")

    # 2. Shuffle and split images for each class
    for class_name, images in class_images.items():
        random.shuffle(images)
        total = len(images)
        
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        # Copy images into their respective split folders
        for split_name, split_images in splits.items():
            split_class_dir = output_path / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                dest_path = split_class_dir / img_path.name
                # Avoid filename collisions by appending a random suffix if needed
                if dest_path.exists():
                    dest_path = split_class_dir / f"{img_path.stem}_{random.randint(1000, 9999)}{img_path.suffix}"
                shutil.copy(img_path, dest_path)

    # 3. Clean up temporary extracted files
    if temp_extract_path.exists():
        shutil.rmtree(temp_extract_path)
        
    print("Dataset successfully separated into clean train, val, and test folders!")

# --- Execution ---
if __name__ == "__main__":
    indian_food_folder = r"C:\Users\srikanth\OneDrive\Desktop\NutriVision-AI\training\raw_datasets\indian_food_dataset"
    fruit_veg_zip_file = r"C:\Users\srikanth\OneDrive\Desktop\NutriVision-AI\training\raw_datasets\fruit_vegetable_dataset.zip"
    target_dataset_dir = r"C:\Users\srikanth\OneDrive\Desktop\nutrivisionV1\datasets"
    
    split_and_organize_datasets(indian_food_folder, fruit_veg_zip_file, target_dataset_dir)