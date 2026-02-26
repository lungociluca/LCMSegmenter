import os

voc_path = '/Users/lungociluca/Downloads/copy_voc'

subdirs = [
    'JPEGImages',
    'SegmentationObject'
]

ids_file_path = './dataset/voc12/val_id.txt'
with open(ids_file_path) as f:
    file_ids = f.read().split('\n')

for i, current_dir in enumerate(subdirs):
    current_dir_path = os.path.join(voc_path, current_dir)
    for current_file in os.listdir(current_dir_path):
        extension ='.jpg' if current_dir == subdirs[0] else '.png'
        current_file_id = current_file.replace(extension, '')
        if current_file_id not in file_ids:
            current_file_path = os.path.join(current_dir_path, current_file)
            os.remove(current_file_path)