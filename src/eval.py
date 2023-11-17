from utils.metrics import compute_metrics
import pickle

img_features, ids = pickle.load(open("models/test_img_features.pkl", 'rb'))
text_features, ids = pickle.load(open("models/test_text_features.pkl", 'rb'))

metrics1 = compute_metrics(img_features, text_features, metric='cosine', recall_klist=(1, 5, 10), return_raw=False)
metrics2 = compute_metrics(text_features, img_features, metric='cosine', recall_klist=(1, 5, 10), return_raw=False)

print(len(ids))

print(metrics1)
print(metrics2)