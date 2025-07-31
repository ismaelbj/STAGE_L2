# Étude d’un modèle de Graph Neural Network attentionnel pour la prédiction d’interactions médicamenteuses
Code développé pendant le stage de L2 à l'Université d'Uppsala, complémetns aux travaux de Ma Mei.
## Note
Quelques commentaires utilses à la compréhension du fonctionnement du modèle ont été ajoutés dans certains fichiers par le créateur du modèle.
## Requirements  
numpy ==1.22.3          
pandas  == 1.4.3           
python  == 3.8.13             
pytorch   == 1.12.0           
rdkit   == 2020.09.1     
scikit-learn  ==1.1.1                   
torch-geometric ==2.0.4                   
torch-scatter == 2.0.9                  
torch-sparse == 0.6.14                 
tqdm  ==   4.64.0  
## Step-by-step running:  

- Commencer par run data_preprocessing.py via la commande  ' data_preprocessing.py -d drugbank -o all`  
  data_preprocessing.py convertit les données pures en graphes.

- Puis, run train.py via ' train.py --fold 0 --save_model' 
