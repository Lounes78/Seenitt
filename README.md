# Piece Classifier

This model takes as input a cropped bounding box containing a single chess piece and predicts which type of piece it is (pawn, rook, knight, bishop, queen, or king) and its color (white or black).

## Datasets

The training data is a combination of:
- A public chess dataset from [Roboflow](https://public.roboflow.com/object-detection/chess-full).
- A custom dataset collected and annotated by us.

Both datasets are available here:  
https://drive.google.com/drive/folders/1wiMKGxKQ8bVzOorNOcTy3BU3tsk-4Fbh?usp=sharing

## Model

The exported ONNX model can be found at:  
https://drive.google.com/drive/folders/16cg1maDqDJtmiLoDn6I1TfvoktAMp938?usp=sharing

The numbers associated to each piece are:
0. 'black-bishop'
1. 'black-king' 
2. 'black-knight'
3. 'black-pawn'
4. 'black-queen' 
5. 'black-rook',
6. 'white-bishop' 
7. 'white-king' 
8. 'white-knight' 
9. 'white-pawn' 
10. 'white-queen'
11. 'white-rook'




![Result](frame_10638.jpg)