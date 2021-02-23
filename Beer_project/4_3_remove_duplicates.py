import hashlib
import os

dirs = 'C:/data/image/beer_selenium/hite/' # cass, filgood, filite, hite
os.chdir(dirs)
file_list = os.listdir()

duplicates = []
hash_keys = dict()
for index, filename in  enumerate(os.listdir('.')):
    if os.path.isfile(filename):
        # Removing Duplicate Images Using Hashing
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys: 
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash]))

print(len(duplicates))
# Delete Files After Printing
for index in duplicates:
    os.remove(file_list[index[0]])
    print("deleted ", index)