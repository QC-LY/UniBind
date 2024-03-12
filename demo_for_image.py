import torch
from imagebind.imagebind_model import ModalityType
import argparse
from utils.utils import load_centre_embeddings
from model import UniBind
from utils.data_transform import load_and_transform_vision_data

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")
    parser.add_argument("--pretrain_weights", type=str, default='./ckpts/pretrained_weights.pt', required=False)
    parser.add_argument("--modality", type=str, default='image', required=False)
    parser.add_argument("--centre_embeddings_path", type=str, default='./centre_embs/image_in_center_embeddings.pkl', required=False)
    args = parser.parse_args()

    model = UniBind(args)
    model.to(device)
    centre_embeddings, centre_labels = load_centre_embeddings(args.centre_embeddings_path, device)
    centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)
    images = ["./assets/image_chair.jpg", "./assets/image_dog.jpg"]
    inputs = {
        ModalityType.VISION: load_and_transform_vision_data(images, device),
    }
    visual_embeddings = model.encode_vision(inputs)
    logic = (visual_embeddings @ centre_embeddings.t()).softmax(dim=-1)
    predictions = []
    for i in range(logic.shape[0]):
        _, index = logic[i].topk(1)
        predictions.append(centre_labels[int(index[0])])
    print(f"The categories are: {predictions}")