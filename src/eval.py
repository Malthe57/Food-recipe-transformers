from utils.metrics import compute_metrics
import pickle

# for i in range(1,3+1):

#     img_features, ids = pickle.load(open(f"models/test_img_features_{i}.pkl", 'rb'))
#     text_features, ids = pickle.load(open(f"models/test_text_features_{i}.pkl", 'rb'))

#     metrics1 = compute_metrics(img_features, text_features, metric='cosine', recall_klist=(1, 5, 10), return_raw=False)
#     # metrics2 = compute_metrics(text_features, img_features, metric='cosine', recall_klist=(1, 5, 10), return_raw=False)
#     print(metrics1)
#     # print(metrics2)

# print(len(img_features))

img_features, ids = pickle.load(open(f"models/test_img_features_clip.pkl", 'rb'))
text_features, ids = pickle.load(open(f"models/test_text_features_clip.pkl", 'rb'))

metrics1 = compute_metrics(img_features, text_features, ids, metric='cosine', recall_klist=(1, 5, 10), return_raw=False, return_idx=False)
# metrics2 = compute_metrics(text_features, img_features, metric='cosine', recall_klist=(1, 5, 10), return_raw=False)
print(metrics1['recall_1']*len(img_features))
print(metrics1)