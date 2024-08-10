import torch
from Model_architecture import ResNet9
from Preprocess import preprocess_image

def Predict(image) :
            # Initialize the model
            model = ResNet9(in_channels=3, num_diseases=38)  # Adjust num_diseases based on your specific case

            # Load the trained weights
            state_dict = torch.load('model_state_dict.pth', map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)

            # Set the model to evaluation mode
            model.eval()

            # Example prediction (using a sample image)
            with torch.no_grad():
                
                predictions = model(image)
                _, predicted_class = torch.max(predictions, dim=1)



            key_of_prediction = predicted_class.item()

            dict = {0: 'Corn_(maize) --> Northern_Leaf_Blight',
            1: 'Tomato --> Bacterial_spot',
            2: 'Tomato --> Leaf_Mold',
            3: 'Cherry_(including_sour) --> Powdery_mildew',
            4: 'Pepper_bell --> Bacterial_spot',
            5: 'Squash --> Powdery_mildew',
            6: 'Grape --> healthy',
            7: 'Corn_(maize) --> Common_rust_',
            8: 'Potato --> healthy',
            9: 'Apple --> Black_rot',
            10: 'Raspberry --> healthy',
            11: 'Orange --> Haunglongbing_(Citrus_greening)',
            12: 'Apple --> Apple_scab',
            13: 'Tomato --> Tomato_Yellow_Leaf_Curl_Virus',
            14: 'Cherry_(including_sou --> __healthy',
            15: 'Soybean --> healthy',
            16: 'Peach --> healthy',
            17: 'Grape --> Esca_(Black_Measles)',
            18: 'Blueberry --> healthy',
            19: 'Pepper_bell --> healthy',
            20: 'Tomato --> Late_blight',
            21: 'Corn_(maize) --> healthy',
            22: 'Tomato --> Tomato_mosaic_virus',
            23: 'Strawberry --> Leaf_scorch',
            24: 'Apple --> healthy',
            25: 'Grape --> Leaf_blight_(Isariopsis_Leaf_Spot)',
            26: 'Tomato --> Spider_mites Two-spotted_spider_mite',
            27: 'Potato --> Late_blight',
            28: 'Tomato --> Early_blight',
            29: 'Grape --> Black_rot',
            30: 'Corn_(maize) --> Cercospora_leaf_spot Gray_leaf_spot',
            31: 'Potato --> Early_blight',
            32: 'Strawberry --> healthy',
            33: 'Tomato --> Septoria_leaf_spot',
            34: 'Tomato --> healthy',
            35: 'Peach --> Bacterial_spot',
            36: 'Tomato --> Target_Spot',
            37: 'Apple --> Cedar_apple_rust'}


            return dict[key_of_prediction]

