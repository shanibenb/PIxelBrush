import os
import numpy as np
import h5py
import pickle

train = True
if train:
    images_path = 'data/oxford/images'
    embedding_path = 'data/oxford/skip-thought-embeddings-train.npy'
    text_path = 'data/oxford/text-train.npy'
    datasetDir = 'data/oxford/oxford_train.hdf5'
    f = h5py.File(datasetDir, 'w')
    train = f.create_group('train')
    split = train

    # Load embedding - skip - thoughts
    my_embedding = np.load(embedding_path)
    txt_file = np.load(text_path)
    filepath = os.path.join('data/oxford/filenames-test.pickle')
else:
    images_path = 'data/oxford/images'
    embedding_path = 'data/oxford/skip-thought-embeddings-test.npy'
    text_path = 'data/oxford/text-test.npy'
    datasetDir = 'data/oxford/oxford_test.hdf5'
    f = h5py.File(datasetDir, 'w')
    test = f.create_group('test')
    split = test

    # Load embedding - skip-thoughts
    my_embedding = np.load(embedding_path)
    txt_file = np.load(text_path)
    filepath = os.path.join('data/oxford/filenames-test.pickle')



with open(filepath, 'rb') as f:
    filenames = pickle.load(f)
i = 0


for i, filename in enumerate(filenames):
    img_name = filenames[i][:-4]
    img_path = img_name + '.jpg'
    embeddings = my_embedding[i]
    example_name = img_path.split('/')[-1][:-4]

    txt = txt_file[i]

    img_path = os.path.join(images_path, img_path)
    img = open(img_path, 'rb').read()

    dt = h5py.special_dtype(vlen=str)

    for c, e in enumerate(embeddings):
        ex = split.create_group(example_name + '_' + str(c))
        ex.create_dataset('name', data=example_name)
        ex.create_dataset('img', data=np.void(img))
        ex.create_dataset('embeddings', data=e)
        ex.create_dataset('class', data=img_path.split('/')[-2])
        ex.create_dataset('txt', data=txt.astype(object), dtype=dt)

    print(example_name)

print(i)

