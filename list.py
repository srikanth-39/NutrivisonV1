import os

base_path = "C:/Users/srikanth/OneDrive/Desktop/NutriVision-AI/training/dataset/train"

# List all directories in the train folder
food_items = sorted([
    name for name in os.listdir(base_path) 
    if os.path.isdir(os.path.join(base_path, name))
])

print(f"Total food items: {len(food_items)}")
print(food_items)