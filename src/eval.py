from utils.metrics import compute_metrics
import pickle
import os

features = os.listdir("models/features/")
offset = len(features) // 2

for i in range(offset):
    print("evaluating:", features[i].split("_")[-1].split(".")[0])
    img_features, ids = pickle.load(open(f"models/features/{features[i]}", 'rb'))
    text_features, ids = pickle.load(open(f"models/features/{features[i+offset]}", 'rb'))

    img2text = compute_metrics(queries=img_features, database=text_features, ids=ids, metric='cosine', recall_klist=(1, 5, 10), return_raw=False, return_idx=False)
    text2img = compute_metrics(queries=text_features, database=img_features, ids=ids, metric='cosine', recall_klist=(1, 5, 10), return_raw=False, return_idx=False)
    print("Number of correct recall @ 1:", img2text['recall_1']*0.01*len(img_features))
    print(img2text)
    print(text2img)