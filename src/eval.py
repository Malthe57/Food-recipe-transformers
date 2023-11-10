from utils.metrics import compute_metrics
import pickle

img_features = pickle.load(open("models/test_img_features.pkl", 'rb'))
text_features = pickle.load(open("models/test_text_features.pkl", 'rb'))

metrics = compute_metrics(img_features, text_features, metric='cosine', recall_klist=(1, 5, 10), return_raw=False)

