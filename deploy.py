import torchvision.transforms as transforms
import cv2 
import torch
from PIL import Image


def transform_image(input_size):
    mean, std = get_imagenet_mean_std()
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def get_imagenet_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return (mean,std)

def preprocess_image(image, input_size, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = transform_image(input_size)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    return image_tensor 

if __name__ == "__main__":
    image = cv2.imread(r'C:\Users\ACER\OneDrive\Desktop\PAOLO\MSU-IIT\BS COM ENG (1st Sem_2024-2025)\COE190\Yolo\vit_rice_classification\archive\extra_resized_raw_images\extra_resized_raw_images\bakanae\Bakanae (1).jpeg')
    if image is None:
        raise FileNotFoundError("Image not found or path is incorrect!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = preprocess_image(image, input_size=(224,224), device=device)
    model = torch.load(r'C:\Users\ACER\OneDrive\Desktop\PAOLO\MSU-IIT\BS COM ENG (1st Sem_2024-2025)\COE190\Yolo\vit_rice_classification\ViT_Rice_Disease_Classification\runs\train_1\best_model.pt')
    model.to(device)
    model.eval()
    labels = {
        0: 'bacterial_leaf_blight',
        1: 'bacterial_leaf_streak',
        2: 'bakanae',
        3: 'brown_spot',
        4: 'grassy_stunt_virus',
        5: 'healthy_rice_plant',
        6: 'narrow_brown_spot',
        7: 'rice_blast',
        8: 'rice_false_smut',
        9: 'sheath_blight',
        10: 'sheath_rot',
        11: 'stem_rot',
        12: 'tungro_virus'
    }
    
    prediction = model(image_tensor)
    pred_label_index = torch.argmax(prediction.logits, dim=1).item()
    pred_label = labels[pred_label_index]
    print(f'The image is predicted as {pred_label}')


    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destoryAllWindows()