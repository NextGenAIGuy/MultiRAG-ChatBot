from datasets import load_dataset
import os 

dataset = load_dataset("detection-datasets/fashionpedia")

dataset_folder = 'Data/image_data'
os.makedirs(dataset_folder, exist_ok=True)

def save_images(dataset, dataset_folder, num_images=1000):
    for i in range(num_images):
        image = dataset['train'][i]['image']
        
        image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))

save_images(dataset, dataset_folder, num_images=1000)

print(f"Saved the first 1000 images to {dataset_folder}")