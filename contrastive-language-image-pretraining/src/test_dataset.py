import kagglehub
import os
import pandas as pd
path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")

#print("Path to dataset files:", path)
#print(type(path))
#print(dir(path))
#print(os.listdir(path))
flicker_data_path = os.path.join(path, "flickr30k_images")
print(os.listdir(flicker_data_path))
print("Flickr data path:", flicker_data_path)
flickr_images_count = len(os.listdir(os.path.join(flicker_data_path, 'flickr30k_images')))
print("Number of images in flickr30k_images folder:", flickr_images_count)


flickr_csv_path = os.path.join(flicker_data_path, 'results.csv')
#print("Flickr CSV path:", flickr_csv_path)
#df = pd.read_csv(flickr_csv_path)
#print(df.head())
#print(df.columns)

'''with open(flickr_csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print("First 10 lines of results.csv:")
    for line in lines[:10]:
        print(line.strip())'''

df = pd.read_csv(
    flickr_csv_path,
    sep='|',
    engine='python',
)
df.columns = [col.strip() for col in df.columns]
print(df.columns)
df['image_name'] = df['image_name'].str.strip()
df['comment_number'] = df['comment_number'].str.strip()
df['comment'] = df['comment'].str.strip()
print(df.head(20))