import os
import shutil

def organize_real_dataset(raw_dir: str = "./data sets/dataset2", target_dir: str = "./data/weather_dataset"):
    os.makedirs(target_dir, exist_ok=True)
    
    # Class mapping from raw prefix to target class directory
    class_map = {
        'cloudy': 'Cloudy',
        'rain': 'Rainy',
        'shine': 'Sunny',
        'sunrise': 'Sunny',
        'snow': 'Snowy',
        'fog': 'Foggy'
    }
    
    for cname in ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Foggy']:
        os.makedirs(os.path.join(target_dir, cname), exist_ok=True)
        
    if not os.path.exists(raw_dir):
        print(f"[!] Directory {raw_dir} does not exist.")
        return
        
    count = 0
    for fname in os.listdir(raw_dir):
        lower_name = fname.lower()
        matched_class = None
        for prefix, target_cls in class_map.items():
            if lower_name.startswith(prefix):
                matched_class = target_cls
                break
                
        if matched_class:
            src_path = os.path.join(raw_dir, fname)
            dst_path = os.path.join(target_dir, matched_class, fname)
            shutil.copy2(src_path, dst_path)
            count += 1
            
    print(f"[+] Successfully organized {count} real weather images into {target_dir}")

if __name__ == "__main__":
    organize_real_dataset()
