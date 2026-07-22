import os
import shutil

def organize_5class_dataset(target_dir: str = "./data/weather_dataset"):
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if target dataset directory is already populated
    existing_images = 0
    for cls_name in ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Foggy']:
        cls_path = os.path.join(target_dir, cls_name)
        if os.path.exists(cls_path):
            existing_images += len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
    if existing_images > 1000:
        print(f"[+] Target dataset directory '{target_dir}' is already populated with {existing_images} images across 5 classes!")
        return

    # Source directories to check
    source_candidates = [
        "./data set 3/data",
        "../data set 3/data",
        "./data set 2",
        "../data set 2",
        "./data sets/dataset2"
    ]
    
    source_dir = None
    for cand in source_candidates:
        if os.path.exists(cand):
            source_dir = cand
            break
            
    if not source_dir:
        print(f"[!] No raw dataset source folder found. Skipping organizing (assuming dataset already pre-extracted).")
        return
        
    mapping = {
        'sunny': 'Sunny',
        'cloudy': 'Cloudy',
        'rainy': 'Rainy',
        'snowy': 'Snowy',
        'foggy': 'Foggy'
    }
    
    total_copied = 0
    for src_cls, dst_cls in mapping.items():
        src_path = os.path.join(source_dir, src_cls)
        dst_path = os.path.join(target_dir, dst_cls)
        os.makedirs(dst_path, exist_ok=True)
        
        if os.path.exists(src_path):
            files = os.listdir(src_path)
            for f in files:
                shutil.copy2(os.path.join(src_path, f), os.path.join(dst_path, f))
                total_copied += 1
            print(f"[+] {dst_cls}: Copied {len(files)} images")
            
    print(f"[+] Total 5-class dataset organized: {total_copied} images in {target_dir}")

if __name__ == "__main__":
    organize_5class_dataset()
